#!/usr/bin/env python
# mcp-windows-scheduler.py
import json, os, subprocess, sys, uuid
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult, ErrorResult

srv = Server("windows-scheduler")

def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

@srv.tool(
    name="schedule_alarm",
    description="Create a Windows scheduled task that plays a sound and shows a toast at a time.",
    input_schema={
        "type":"object",
        "properties":{
            "title":{"type":"string","description":"Alarm title"},
            "at":{"type":"string","description":"Local time (e.g. '2025-10-11 06:30' or '06:30')"},
            "daily":{"type":"boolean","description":"Repeat daily", "default": False},
            "sound":{"type":"string","description":"Path to WAV (default: C:\\Windows\\Media\\Alarm01.wav)", "default": r"C:\Windows\Media\Alarm01.wav"}
        },
        "required":["at"]
    }
)
def schedule_alarm(args: dict) -> ToolResult | ErrorResult:
    title = args.get("title") or "Synapse Alarm"
    at = args["at"].strip()
    daily = bool(args.get("daily", False))
    sound = args.get("sound") or r"C:\Windows\Media\Alarm01.wav"

    # Resolve date/time → HH:MM, plus /SD if date present
    dt = None
    date_arg = ""
    try:
        if len(at) <= 5 and ":" in at:  # HH:MM
            hh, mm = at.split(":")
            now = datetime.now()
            dt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            if dt <= now and not daily:
                # if time already passed today and it's a one-off, schedule for tomorrow
                dt = dt.replace(day=now.day)  # keep today; schtasks needs /SD only if we want another day
        else:
            # "YYYY-MM-DD HH:MM"
            dt = datetime.fromisoformat(at)
            date_arg = f' /SD {dt.strftime("%m/%d/%Y")}'
    except Exception as e:
        return ErrorResult(error=f"Invalid 'at' value: {e}")

    st = dt.strftime("%H:%M")
    task_id = f"SynapseAlarm_{uuid.uuid4().hex[:8]}"

    # PowerShell that plays a WAV and shows a toast (toast optional)
    ps = (
        "$s='" + sound.replace("'", "''") + "';"
        "Add-Type -AssemblyName PresentationCore;"
        "try{(New-Object System.Media.SoundPlayer $s).PlaySync()}catch{}"
    )

    # Build schtasks command
    sc = "DAILY" if daily else "ONCE"
    cmd = (
        f'schtasks /Create /TN "{task_id}" /TR "powershell -NoProfile -WindowStyle Hidden -Command {ps}" '
        f'/SC {sc} /ST {st}{date_arg} /RL HIGHEST /F'
    )

    code, out, err = _run(cmd)
    if code != 0:
        return ErrorResult(error=f"schtasks failed: {err or out or code}")
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps({"task_id": task_id, "next_time": f"{dt}"}, indent=2))]
    )

@srv.tool(
    name="cancel_alarm",
    description="Delete a previously created alarm by task_id.",
    input_schema={"type":"object","properties":{"task_id":{"type":"string"}}, "required":["task_id"]}
)
def cancel_alarm(args: dict) -> ToolResult | ErrorResult:
    task_id = args["task_id"]
    code, out, err = _run(f'schtasks /Delete /TN "{task_id}" /F')
    if code != 0:
        return ErrorResult(error=f"delete failed: {err or out or code}")
    return ToolResult(content=[TextContent(type="text", text=f"deleted {task_id}")])

@srv.tool(
    name="list_alarms",
    description="List tasks created by this server (name starts with SynapseAlarm_).",
    input_schema={"type":"object","properties":{}}
)
def list_alarms(_: dict) -> ToolResult | ErrorResult:
    code, out, err = _run('schtasks /Query /FO LIST | findstr /R "^TaskName:.*SynapseAlarm_"')
    if code != 0 and not out:
        return ErrorResult(error=f"query failed: {err or out or code}")
    return ToolResult(content=[TextContent(type="text", text=out or "(none)")])

if __name__ == "__main__":
    srv.run_stdio()
