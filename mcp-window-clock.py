#!/usr/bin/env python
# mcp-window-clock.py
import json
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult, ErrorResult

srv = Server("windows-clock")

@srv.tool(
    name="get_current_time",
    description="Get the current system time and date.",
    input_schema={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Time format (iso, readable, timestamp)",
                "default": "readable"
            }
        }
    }
)
def get_current_time(args: dict) -> ToolResult | ErrorResult:
    try:
        now = datetime.now()
        format_type = args.get("format", "readable")
        
        if format_type == "iso":
            time_str = now.isoformat()
        elif format_type == "timestamp":
            time_str = str(int(now.timestamp()))
        else:  # readable
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        return ToolResult(
            content=[TextContent(
                type="text", 
                text=json.dumps({
                    "current_time": time_str,
                    "timezone": str(now.astimezone().tzinfo),
                    "unix_timestamp": int(now.timestamp())
                }, indent=2)
            )]
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to get current time: {e}")

@srv.tool(
    name="get_timezone_info",
    description="Get information about the current timezone.",
    input_schema={
        "type": "object",
        "properties": {}
    }
)
def get_timezone_info(args: dict) -> ToolResult | ErrorResult:
    try:
        now = datetime.now()
        tz = now.astimezone().tzinfo
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "timezone_name": str(tz),
                    "utc_offset": str(tz.utcoffset(now)),
                    "is_dst": tz.dst(now) is not None
                }, indent=2)
            )]
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to get timezone info: {e}")

@srv.tool(
    name="format_time",
    description="Format a specific time or date.",
    input_schema={
        "type": "object",
        "properties": {
            "time": {
                "type": "string",
                "description": "Time string to format (ISO format or timestamp)"
            },
            "format": {
                "type": "string",
                "description": "Output format string (e.g., '%Y-%m-%d %H:%M:%S')",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        },
        "required": ["time"]
    }
)
def format_time(args: dict) -> ToolResult | ErrorResult:
    try:
        time_input = args["time"]
        format_str = args.get("format", "%Y-%m-%d %H:%M:%S")
        
        # Try to parse as ISO format first
        try:
            dt = datetime.fromisoformat(time_input)
        except ValueError:
            # Try as timestamp
            try:
                dt = datetime.fromtimestamp(float(time_input))
            except ValueError:
                return ErrorResult(error="Invalid time format. Use ISO format or timestamp.")
        
        formatted_time = dt.strftime(format_str)
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "original_time": time_input,
                    "formatted_time": formatted_time,
                    "iso_format": dt.isoformat()
                }, indent=2)
            )]
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to format time: {e}")

if __name__ == "__main__":
    srv.run_stdio()