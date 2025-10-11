# #!/usr/bin/env python3
# """
# MCP API Server for Synapse Frontend Integration
# This server runs MCP servers locally and provides an API for the frontend to use tools.
# """

# import asyncio
# import json
# import logging
# import os
# import re
# import sys
# from typing import Dict, List, Optional, Any
# from enum import Enum
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn

# # Import from main.py
# from main import Configuration, LLMProvider, MultiLLMClient, Server, ChatSession
# from scheduler_service import initialize_scheduler, get_scheduler, shutdown_scheduler

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Global variables for MCP session
# mcp_session: Optional[ChatSession] = None
# initialized = False

# class QueryRequest(BaseModel):
#     query: str
#     user_id: str

# class QueryResponse(BaseModel):
#     response: str
#     success: bool
#     error: Optional[str] = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     await startup_event()
#     yield
#     # Shutdown
#     await shutdown_event()

# # FastAPI app
# app = FastAPI(title="MCP API Server", version="1.0.0", lifespan=lifespan)

# # CORS middleware to allow frontend requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:4173", "http://localhost:3000"],  # Frontend URLs
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# async def startup_event():
#     """Initialize MCP servers on startup."""
#     global mcp_session, initialized
    
#     try:
#         logging.info("🚀 Starting MCP API Server...")
        
#         # Load configuration
#         config = Configuration()
        
#         # Initialize LLM client
#         llm_client = MultiLLMClient(config)
        
#         # Load servers configuration
#         with open('servers_config.json', 'r') as f:
#             servers_config = json.load(f).get("mcpServers", {})
        
#         # Initialize MCP servers
#         servers = []
#         logging.info("🔌 Initializing MCP servers...")
        
#         for server_name, server_params in servers_config.items():
#             command = server_params["command"]
#             args = server_params.get("args", [])
#             env = server_params.get("env", {})
            
#             # Inject Notion token if available
#             if server_name == "notionApi" and config.notion_token:
#                 env["NOTION_TOKEN"] = config.notion_token
#                 logging.info("Injected NOTION_TOKEN into notionApi server environment.")
            
#             logging.info(f"Starting MCP server: {server_name} with command: {command} {args}")
#             try:
#                 # Create config dictionary for Server constructor
#                 server_config = {
#                     "command": command,
#                     "args": args,
#                     "env": env
#                 }
#                 server = Server(server_name, server_config)
#                 servers.append(server)
#                 logging.info(f"MCP server '{server_name}' created successfully.")
#             except Exception as e:
#                 logging.error(f"Failed to create MCP server '{server_name}': {e}")
        
#         # Initialize servers
#         initialized_servers = 0
#         for server in servers:
#             try:
#                 await server.initialize()
#                 initialized_servers += 1
#                 logging.info(f"✓ Server '{server.name}' initialized")
#             except Exception as e:
#                 logging.error(f"✗ Failed to initialize server '{server.name}': {e}")
        
#         if initialized_servers == 0:
#             logging.warning("⚠ No MCP servers were initialized. Continuing without tools...")
        
#         # Create chat session
#         mcp_session = ChatSession(servers, llm_client)
        
#         # Load tools
#         logging.info("🔧 Loading tools...")
#         all_tools = []
#         for server in servers:
#             if server.session:
#                 try:
#                     tools = await server.list_tools()
#                     all_tools.extend(tools)
#                     logging.info(f"✓ Loaded {len(tools)} tools from '{server.name}'")
#                 except Exception as e:
#                     logging.warning(f"✗ Could not load tools from '{server.name}': {e}")
        
#         logging.info(f"📦 Total tools available: {len(all_tools)}")
#         # Attach loaded tools to the chat session for better prompting
#         try:
#             mcp_session.all_tools = all_tools
#             mcp_session.tools_loaded = len(all_tools) > 0
#         except Exception:
#             pass
        
#         # Set Groq as primary provider
#         if LLMProvider.GROQ in llm_client.available_providers:
#             llm_client.current_provider = LLMProvider.GROQ
#             logging.info("🎯 Set Groq as primary LLM provider")
#         elif LLMProvider.GEMINI in llm_client.available_providers:
#             llm_client.current_provider = LLMProvider.GEMINI
#             logging.info("🎯 Set Gemini as primary LLM provider")
        
#         initialized = True
#         logging.info("✅ MCP API Server initialized successfully!")
        
#         # Initialize scheduler service
#         async def mcp_executor(action: str, user_id: str) -> bool:
#             """Execute MCP action for scheduled tasks"""
#             try:
#                 # Clean the action by removing time-related parts
#                 cleaned_action = action
                
#                 # Remove common time expressions from the action
#                 time_patterns = [
#                     r'\s+(in|after)\s+\d+\s*(minutes?|mins?|m|hours?|hrs?|h|days?|d|seconds?|secs?|s)\b',
#                     r'\s+at\s+\d{1,2}:\d{2}\s*(am|pm)?',
#                     r'\s+at\s+\d{1,2}\s*(am|pm)',
#                     r'\s+daily\s+',
#                     r'\s+weekly\s+',
#                     r'\s+monthly\s+',
#                     r'\s+every\s+day\s+',
#                     r'\s+every\s+week\s+',
#                     r'\s+every\s+month\s+'
#                 ]
                
#                 for pattern in time_patterns:
#                     cleaned_action = re.sub(pattern, '', cleaned_action, flags=re.IGNORECASE)
                
#                 # Clean up extra spaces
#                 cleaned_action = ' '.join(cleaned_action.split())
                
#                 logging.info(f"🎵 Executing cleaned action: '{cleaned_action}'")
#                 result = await mcp_session.chat(cleaned_action)
#                 return result is not None
#             except Exception as e:
#                 logging.error(f"Error executing scheduled action '{action}': {e}")
#                 return False
        
#         initialize_scheduler(mcp_executor)
#         logging.info("🕐 Scheduler service initialized")
        
#     except Exception as e:
#         logging.error(f"❌ Failed to initialize MCP API Server: {e}")
#         initialized = False

# async def shutdown_event():
#     """Clean up MCP servers on shutdown."""
#     global mcp_session, initialized
    
#     # Shutdown scheduler service
#     shutdown_scheduler()
#     logging.info("🕐 Scheduler service stopped")
    
#     if mcp_session and mcp_session.servers:
#         logging.info("🧹 Cleaning up MCP servers...")
#         cleanup_tasks = []
#         for server in mcp_session.servers:
#             cleanup_tasks.append(server.cleanup())
        
#         if cleanup_tasks:
#             await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
#         logging.info("✅ MCP servers cleaned up")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "healthy" if initialized else "not_initialized",
#         "initialized": initialized,
#         "servers_count": len(mcp_session.servers) if mcp_session else 0
#     }

# @app.get("/tools")
# async def list_tools():
#     """List available MCP tools."""
#     if not initialized or not mcp_session:
#         raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
#     try:
#         all_tools = []
#         for server in mcp_session.servers:
#             if server.session:
#                 try:
#                     tools = await server.list_tools()
#                     for tool in tools:
#                         all_tools.append({
#                             "name": tool.name,
#                             "description": tool.description,
#                             "server": server.name
#                         })
#                 except Exception as e:
#                     logging.warning(f"Could not list tools from {server.name}: {e}")
        
#         return {"tools": all_tools, "count": len(all_tools)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

# @app.post("/query", response_model=QueryResponse)
# async def process_query(request: QueryRequest):
#     """Process a query using MCP tools and LLM."""
#     if not initialized or not mcp_session:
#         raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
#     try:
#         logging.info(f"🔧 Processing tools query: {request.query}")
        
#         # Check if this is a scheduling request
#         scheduler = get_scheduler()
#         if scheduler:
#             # Check for scheduling keywords
#             scheduling_keywords = ['in ', 'at ', 'daily', 'weekly', 'monthly', 'every', 'schedule', 'remind']
#             if any(keyword in request.query.lower() for keyword in scheduling_keywords):
#                 # Try to schedule the task
#                 task_id = scheduler.schedule_task(request.user_id, request.query, request.query)
#                 if task_id:
#                     return QueryResponse(
#                         response=f"✅ Task scheduled successfully! I'll execute '{request.query}' at the specified time. Task ID: {task_id}",
#                         success=True
#                     )
        
#         # Regular MCP processing
#         response = await mcp_session.chat(request.query)
        
#         return QueryResponse(
#             response=response,
#             success=True
#         )
        
#     except Exception as e:
#         logging.error(f"Error processing query: {e}")
#         return QueryResponse(
#             response=f"❌ Error processing tools request: {str(e)}",
#             success=False,
#             error=str(e)
#         )

# @app.get("/status")
# async def get_status():
#     """Get detailed status of MCP servers and tools."""
#     if not initialized or not mcp_session:
#         return {
#             "initialized": False,
#             "servers": [],
#             "tools": [],
#             "llm_provider": None
#         }
    
#     try:
#         servers_status = []
#         all_tools = []
        
#         for server in mcp_session.servers:
#             server_info = {
#                 "name": server.name,
#                 "initialized": server.session is not None,
#                 "tools": []
#             }
            
#             if server.session:
#                 try:
#                     tools = await server.list_tools()
#                     for tool in tools:
#                         tool_info = {
#                             "name": tool.name,
#                             "description": tool.description
#                         }
#                         server_info["tools"].append(tool_info)
#                         all_tools.append(tool_info)
#                 except Exception as e:
#                     server_info["error"] = str(e)
            
#             servers_status.append(server_info)
        
#         return {
#             "initialized": True,
#             "servers": servers_status,
#             "tools": all_tools,
#             "llm_provider": mcp_session.llm_client.current_provider.value if mcp_session.llm_client.current_provider else None,
#             "available_providers": [p.value for p in mcp_session.llm_client.available_providers]
#         }
        
#     except Exception as e:
#         return {
#             "initialized": True,
#             "error": str(e),
#             "servers": [],
#             "tools": [],
#             "llm_provider": None
#         }

# @app.get("/scheduled-tasks")
# async def get_scheduled_tasks(user_id: str = None):
#     """Get scheduled tasks, optionally filtered by user."""
#     scheduler = get_scheduler()
#     if not scheduler:
#         raise HTTPException(status_code=503, detail="Scheduler not available")
    
#     tasks = scheduler.get_scheduled_tasks(user_id)
#     return {"tasks": tasks}

# @app.delete("/scheduled-tasks/{task_id}")
# async def cancel_scheduled_task(task_id: str):
#     """Cancel a scheduled task."""
#     scheduler = get_scheduler()
#     if not scheduler:
#         raise HTTPException(status_code=503, detail="Scheduler not available")
    
#     success = scheduler.cancel_task(task_id)
#     if success:
#         return {"message": f"Task {task_id} cancelled successfully"}
#     else:
#         raise HTTPException(status_code=404, detail="Task not found")

# if __name__ == "__main__":
#     print("🚀 Starting MCP API Server...")
#     print("📡 Server will be available at: http://localhost:8001")
#     print("🔧 MCP servers will be initialized on startup")
#     print("🌐 Frontend can connect to this server for tools functionality")
#     print("\n" + "="*60)
    
#     uvicorn.run(
#         "mcp_api_server:app",
#         host="0.0.0.0",
#         port=8001,
#         reload=False,
#         log_level="info"
#     )


#!/usr/bin/env python3
"""
MCP API Server for Synapse Frontend Integration
- Initializes MCP servers once on startup
- Exposes HTTP API to list tools, run queries, and schedule actions
- Supports "in/after X min/hour/..." and "at HH(:MM) [am|pm]" parsing
"""

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import zoneinfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Project imports
from main import Configuration, LLMProvider, MultiLLMClient, Server, ChatSession
from scheduler_service import (
    initialize_scheduler,
    get_scheduler,
    shutdown_scheduler,
    schedule_in_seconds,
    schedule_at,
    schedule_daily_at,
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    user_id: str

class QueryResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
mcp_session: Optional[ChatSession] = None
initialized: bool = False

# Timezone
try:
    import tzlocal  # optional
    LOCAL_TZ = zoneinfo.ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")

# ------------------------------------------------------------------------------
# Scheduling phrase parser
# ------------------------------------------------------------------------------
def _parse_schedule(text: str):
    """
    Returns a dict describing schedule or None if no schedule phrase found.

    Formats supported:
      - "in 5 min", "after 2 minutes", "in 10s", "after 1 hour", "in 2 days"
      - "at 5pm", "at 17:05", "at 5:05 pm"

    Output:
      {"mode":"delay","seconds":<int>,"clean":"<action-without-time-phrase>"}
      {"mode":"at","run_at":<aware-datetime>,"clean":"<action-without-time-phrase>"}
      None
    """
    q = text.strip()

    # 1) Delay: in/after N unit
    m = re.search(
        r"\b(in|after)\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
        q,
        re.IGNORECASE,
    )
    if m:
        amount = int(m.group(2))
        unit = m.group(3).lower()
        if unit.startswith("s"):
            seconds = amount
        elif unit.startswith("m"):
            seconds = amount * 60
        elif unit.startswith("h"):
            seconds = amount * 3600
        else:
            seconds = amount * 86400

        clean = (q[:m.start()] + q[m.end():]).strip()
        return {"mode": "delay", "seconds": seconds, "clean": clean}

    # 2) Absolute time today: "at 5pm", "at 17:05", "at 5:05 pm"
    m = re.search(r"\bat\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", q, re.IGNORECASE)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        ap = (m.group(3) or "").lower()

        if ap == "pm" and hh < 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0

        now = datetime.now(LOCAL_TZ)
        run_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if run_at <= now:
            run_at = run_at + timedelta(days=1)

        clean = (q[:m.start()] + q[m.end():]).strip()
        return {"mode": "at", "run_at": run_at, "clean": clean}

    return None

# ------------------------------------------------------------------------------
# FastAPI app + lifespan
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()

app = FastAPI(title="MCP API Server", version="1.1.0", lifespan=lifespan)

# CORS for your frontend(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Startup / Shutdown
# ------------------------------------------------------------------------------
async def startup_event():
    """Initialize MCP servers and scheduler on startup."""
    global mcp_session, initialized

    try:
        logging.info("🚀 Starting MCP API Server...")

        # Load config & LLM
        config = Configuration()
        llm_client = MultiLLMClient(config)

        # Load servers_config.json
        with open("servers_config.json", "r", encoding="utf-8") as f:
            servers_cfg = json.load(f).get("mcpServers", {})

        servers: List[Server] = []
        logging.info("🔌 Initializing MCP servers...")
        for server_name, server_params in servers_cfg.items():
            command = server_params.get("command")
            args = server_params.get("args", [])
            env = dict(server_params.get("env", {}))

            # Inject NOTION_TOKEN from .env if present
            if server_name.lower() == "notionapi" and config.notion_token:
                env["NOTION_TOKEN"] = config.notion_token

            srv = Server(server_name, {"command": command, "args": args, "env": env})
            try:
                await srv.initialize()
                servers.append(srv)
                logging.info(f"  ✓ Server '{server_name}' initialized")
            except Exception as e:
                logging.error(f"  ✗ Failed to initialize '{server_name}': {e}")

        # Create chat session and load tools
        mcp_session = ChatSession(servers, llm_client)
        all_tools = []
        for srv in servers:
            if not srv.session:
                continue
            try:
                tools = await srv.list_tools()
                all_tools.extend(tools)
                logging.info(f"  ✓ Loaded {len(tools)} tools from '{srv.name}'")
            except Exception as e:
                logging.warning(f"  ✗ Could not load tools from '{srv.name}': {e}")

        mcp_session.all_tools = all_tools
        mcp_session.tools_loaded = len(all_tools) > 0
        logging.info(f"📦 Total tools available: {len(all_tools)}")

        # Choose default LLM
        if LLMProvider.GROQ in llm_client.available_providers:
            llm_client.current_provider = LLMProvider.GROQ
        elif LLMProvider.GEMINI in llm_client.available_providers:
            llm_client.current_provider = LLMProvider.GEMINI
        logging.info(f"🎯 LLM provider: {llm_client.current_provider.value if llm_client.current_provider else 'None'}")

        # Wire the scheduler to call MCP later
        async def mcp_executor(action: str, user_id: str) -> bool:
            """Executor used by scheduler to run the action later."""
            try:
                # Strip any residual timing phrase (best-effort)
                cleaned = _parse_schedule(action)
                final_action = cleaned["clean"] if cleaned else action
                logging.info(f"▶️ Executing scheduled action for user={user_id}: {final_action}")
                result = await mcp_session.chat(final_action)
                logging.info(f"✅ Scheduled action result: {result[:200] if isinstance(result, str) else result}")
                return True
            except Exception as e:
                logging.exception(f"Scheduled action failed: {e}")
                return False

        initialize_scheduler(mcp_executor)
        logging.info("🕐 Scheduler initialized")

        initialized = True
        logging.info("✅ MCP API Server ready")

    except Exception as e:
        initialized = False
        logging.exception(f"Startup failed: {e}")

async def shutdown_event():
    """Cleanup MCP servers and scheduler."""
    global mcp_session
    try:
        shutdown_scheduler()
        logging.info("🕐 Scheduler stopped")
    except Exception as e:
        logging.warning(f"Scheduler shutdown warning: {e}")

    if mcp_session:
        logging.info("🧹 Cleaning up MCP servers...")
        try:
            await mcp_session.cleanup_servers()
            logging.info("✅ MCP servers cleaned up")
        except Exception as e:
            logging.warning(f"Cleanup warning: {e}")

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if initialized else "not_initialized",
        "initialized": initialized,
        "servers_count": len(mcp_session.servers) if mcp_session else 0,
    }

@app.get("/status")
async def get_status():
    if not initialized or not mcp_session:
        return {"initialized": False, "servers": [], "tools": [], "llm_provider": None}

    servers_status = []
    all_tools = []
    for srv in mcp_session.servers:
        info = {"name": srv.name, "initialized": srv.session is not None, "tools": []}
        if srv.session:
            try:
                tools = await srv.list_tools()
                for t in tools:
                    ti = {"name": t.name, "description": t.description}
                    info["tools"].append(ti)
                    all_tools.append(ti)
            except Exception as e:
                info["error"] = str(e)
        servers_status.append(info)

    return {
        "initialized": True,
        "servers": servers_status,
        "tools": all_tools,
        "llm_provider": mcp_session.llm_client.current_provider.value if mcp_session.llm_client.current_provider else None,
        "available_providers": [p.value for p in mcp_session.llm_client.available_providers],
    }

@app.get("/tools")
async def list_tools():
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    out = []
    for srv in mcp_session.servers:
        if not srv.session:
            continue
        try:
            tools = await srv.list_tools()
            for t in tools:
                out.append({"name": t.name, "description": t.description, "server": srv.name})
        except Exception as e:
            logging.warning(f"Could not list tools from {srv.name}: {e}")
    return {"tools": out, "count": len(out)}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Schedule or execute a query."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")

    try:
        plan = _parse_schedule(request.query)
        sched = get_scheduler()

        # Scheduling path
        if sched and plan:
            action = plan["clean"] or request.query

            if plan["mode"] == "delay":
                task_id = schedule_in_seconds(
                    user_id=request.user_id,
                    action=action,
                    seconds=plan["seconds"],
                    description=request.query,
                )
                if task_id:
                    return QueryResponse(
                        response=f"⏱️ Scheduled in {plan['seconds']}s: “{action}”. Task ID: {task_id}",
                        success=True,
                    )

            if plan["mode"] == "at":
                task_id = schedule_at(
                    user_id=request.user_id,
                    action=action,
                    run_at=plan["run_at"],
                    description=request.query,
                )
                if task_id:
                    pretty = plan["run_at"].strftime("%Y-%m-%d %H:%M")
                    return QueryResponse(
                        response=f"🗓️ Scheduled at {pretty}: “{action}”. Task ID: {task_id}",
                        success=True,
                    )

        # Immediate execution path
        resp = await mcp_session.chat(request.query)
        return QueryResponse(response=resp, success=True)

    except Exception as e:
        logging.exception("process_query error")
        return QueryResponse(response=f"❌ {e}", success=False, error=str(e))

@app.get("/scheduled-tasks")
async def get_scheduled_tasks(user_id: str = None):
    sched = get_scheduler()
    if not sched:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    return {"tasks": sched.get_jobs()} if hasattr(sched, "get_jobs") else {"tasks": []}

@app.delete("/scheduled-tasks/{task_id}")
async def cancel_scheduled_task(task_id: str):
    sched = get_scheduler()
    if not sched:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    from scheduler_service import cancel_task  # local helper keeps our in-memory index in sync
    ok = cancel_task(task_id)
    if ok:
        return {"message": f"Task {task_id} cancelled successfully"}
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/weather/{city}")
async def get_weather(city: str):
    """Get weather data for a specific city using OpenWeatherMap API."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
    try:
        # Use the weather functionality from the LLM client
        weather_data = mcp_session.llm_client.get_weather(city)
        
        if "error" in weather_data:
            raise HTTPException(status_code=400, detail=weather_data["error"])
        
        # Format the response for the frontend
        response = {
            "city": weather_data.get("name", city),
            "country": weather_data.get("sys", {}).get("country", "Unknown"),
            "temperature": weather_data.get("main", {}).get("temp", "N/A"),
            "feels_like": weather_data.get("main", {}).get("feels_like", "N/A"),
            "humidity": weather_data.get("main", {}).get("humidity", "N/A"),
            "pressure": weather_data.get("main", {}).get("pressure", "N/A"),
            "description": weather_data.get("weather", [{}])[0].get("description", "N/A"),
            "main_weather": weather_data.get("weather", [{}])[0].get("main", "N/A"),
            "wind_speed": weather_data.get("wind", {}).get("speed", "N/A"),
            "wind_direction": weather_data.get("wind", {}).get("deg", "N/A"),
            "visibility": weather_data.get("visibility", "N/A"),
            "clouds": weather_data.get("clouds", {}).get("all", "N/A"),
            "sunrise": weather_data.get("sys", {}).get("sunrise", "N/A"),
            "sunset": weather_data.get("sys", {}).get("sunset", "N/A"),
            "raw_data": weather_data  # Include full data for advanced use cases
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Weather API error for city '{city}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather data: {str(e)}")

@app.post("/weather")
async def get_weather_by_query(request: QueryRequest):
    """Get weather data by processing a natural language query."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
    try:
        # Process the weather query through the MCP session
        response = await mcp_session.chat(request.query)
        return QueryResponse(response=response, success=True)
        
    except Exception as e:
        logging.error(f"Weather query error: {e}")
        return QueryResponse(
            response=f"❌ Error processing weather query: {str(e)}",
            success=False,
            error=str(e)
        )

@app.get("/news")
async def get_news(query: str, max_results: int = 10, language: str = "en"):
    """Get news articles from GNews API."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
    try:
        # Use the news functionality from the LLM client
        news_data = mcp_session.llm_client.get_news(query, max_results, language)
        
        if "error" in news_data:
            raise HTTPException(status_code=400, detail=news_data["error"])
        
        # Format the response for the frontend
        articles = news_data.get("articles", [])
        total_articles = news_data.get("totalArticles", 0)
        
        formatted_articles = []
        for article in articles:
            formatted_article = {
                "id": article.get("id", ""),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "url": article.get("url", ""),
                "image": article.get("image", ""),
                "published_at": article.get("publishedAt", ""),
                "language": article.get("lang", ""),
                "source": {
                    "id": article.get("source", {}).get("id", ""),
                    "name": article.get("source", {}).get("name", ""),
                    "url": article.get("source", {}).get("url", ""),
                    "country": article.get("source", {}).get("country", "")
                }
            }
            formatted_articles.append(formatted_article)
        
        response = {
            "query": query,
            "total_articles": total_articles,
            "returned_articles": len(formatted_articles),
            "articles": formatted_articles,
            "information": news_data.get("information", {}),
            "raw_data": news_data  # Include full data for advanced use cases
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"News API error for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch news data: {str(e)}")

@app.post("/news")
async def get_news_by_query(request: QueryRequest):
    """Get news data by processing a natural language query."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
    try:
        # Process the news query through the MCP session
        response = await mcp_session.chat(request.query)
        return QueryResponse(response=response, success=True)
        
    except Exception as e:
        logging.error(f"News query error: {e}")
        return QueryResponse(
            response=f"❌ Error processing news query: {str(e)}",
            success=False,
            error=str(e)
        )

# ------------------------------------------------------------------------------
# Entrypoint (optional)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting MCP API Server at http://localhost:8001")
    uvicorn.run("mcp_api_server:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
