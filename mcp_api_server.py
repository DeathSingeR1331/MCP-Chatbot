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
import os
import re
import signal
import socket
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import zoneinfo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Project imports
try:
    from main import Configuration, LLMProvider, MultiLLMClient, Server, ChatSession
    from scheduler_service import (
        initialize_scheduler,
        get_scheduler,
        shutdown_scheduler,
        schedule_in_seconds,
        schedule_at,
        schedule_daily_at,
    )
except Exception:
    # Stub imports when running outside the full project. These will satisfy
    # the Python parser but should be replaced by real modules in production.
    Configuration = object  # type: ignore
    LLMProvider = object  # type: ignore
    MultiLLMClient = object  # type: ignore
    Server = object  # type: ignore
    ChatSession = object  # type: ignore
    def initialize_scheduler(*args, **kwargs): return None  # type: ignore
    def get_scheduler(): return None  # type: ignore
    def shutdown_scheduler(): return None  # type: ignore
    def schedule_in_seconds(*args, **kwargs): return None  # type: ignore
    def schedule_at(*args, **kwargs): return None  # type: ignore
    def schedule_daily_at(*args, **kwargs): return None  # type: ignore

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
mcp_session: Optional["ChatSession"] = None
initialized: bool = False

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logging.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Timezone
try:
    import tzlocal  # optional

    LOCAL_TZ = zoneinfo.ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")

# Port utilities
def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            return True
    except OSError:
        return False

def find_available_port(start_port: int = 8001, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

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

        clean = (q[: m.start()] + q[m.end() :]).strip()
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

        clean = (q[: m.start()] + q[m.end() :]).strip()
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
        try:
            await shutdown_event()
        except Exception as e:
            logging.warning(f"Shutdown event error: {e}")
        # Ensure we don't leave any hanging tasks
        try:
            # Cancel any remaining tasks
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logging.warning(f"Task cleanup error: {e}")


app = FastAPI(title="MCP API Server", version="1.1.0", lifespan=lifespan)

# CORS for your frontend(s) - Updated for Railway deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4173", 
        "http://localhost:3000",
        "https://synapse-front-end.vercel.app",  # Vercel frontend
        "https://synapse-backend-production-7887.up.railway.app",  # Railway backend
    ],
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
            if server_name.lower() == "notionapi" and getattr(config, "notion_token", None):
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
        all_tools: List[Any] = []
        for srv in servers:
            # Skip servers that failed to initialize
            if not getattr(srv, "session", None):
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

        # Choose default LLM provider
        if hasattr(LLMProvider, "GROQ") and getattr(llm_client, "available_providers", None):
            if LLMProvider.GROQ in llm_client.available_providers:
                llm_client.current_provider = LLMProvider.GROQ
            elif LLMProvider.GEMINI in llm_client.available_providers:
                llm_client.current_provider = LLMProvider.GEMINI
        logging.info(
            f"🎯 LLM provider: {llm_client.current_provider.value if getattr(llm_client.current_provider, 'value', None) else 'None'}"
        )

        # Wire the scheduler to call MCP later
        async def mcp_executor(action: str, user_id: str) -> bool:
            """Executor used by scheduler to run the action later."""
            try:
                # Strip any residual timing phrase (best-effort)
                cleaned = _parse_schedule(action)
                final_action = cleaned["clean"] if cleaned else action
                logging.info(f"▶️ Executing scheduled action for user={user_id}: {final_action}")
                result = await mcp_session.chat(final_action)
                logging.info(
                    f"✅ Scheduled action result: {result[:200] if isinstance(result, str) else result}"
                )
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

    # Stop scheduler first
    try:
        shutdown_scheduler()
        logging.info("🕐 Scheduler stopped")
    except Exception as e:
        logging.warning(f"Scheduler shutdown warning: {e}")

    # Clean up MCP servers with better error handling
    if mcp_session:
        logging.info("🧹 Cleaning up MCP servers...")
        try:
            # Use asyncio.wait_for to prevent hanging
            await asyncio.wait_for(mcp_session.cleanup_servers(), timeout=3.0)
            logging.info("✅ MCP servers cleaned up")
        except asyncio.TimeoutError:
            logging.warning("⚠️ MCP server cleanup timed out, forcing shutdown")
        except asyncio.CancelledError:
            logging.info("ℹ️ MCP server cleanup was cancelled (normal during shutdown)")
        except Exception as e:
            logging.warning(f"Cleanup warning: {e}")

    # Don't sleep during shutdown as it can cause cancellation errors
    # The lifespan context will handle the final cleanup

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if initialized else "not_initialized",
        "initialized": initialized,
        "servers_count": len(getattr(mcp_session, "servers", [])) if mcp_session else 0,
    }


@app.get("/status")
async def get_status():
    if not initialized or not mcp_session:
        return {"initialized": False, "servers": [], "tools": [], "llm_provider": None}

    servers_status: List[Dict[str, Any]] = []
    all_tools: List[Dict[str, Any]] = []
    for srv in mcp_session.servers:
        info: Dict[str, Any] = {"name": srv.name, "initialized": srv.session is not None, "tools": []}
        if srv.session:
            try:
                tools = await srv.list_tools()
                for t in tools:
                    tool_info = {"name": t.name, "description": t.description}
                    info["tools"].append(tool_info)
                    all_tools.append(tool_info)
            except Exception as e:
                info["error"] = str(e)
        servers_status.append(info)

    return {
        "initialized": True,
        "servers": servers_status,
        "tools": all_tools,
        "llm_provider": getattr(
            mcp_session.llm_client.current_provider, "value", None
        ),
        "available_providers": [
            p.value for p in getattr(mcp_session.llm_client, "available_providers", [])
        ],
    }


@app.get("/tools")
async def list_tools():
    """List available MCP tools."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    out: List[Dict[str, Any]] = []
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
                        response=f"Scheduled at {pretty}: {action}. Task ID: {task_id}",
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
    """Get scheduled tasks, optionally filtered by user."""
    sched = get_scheduler()
    if not sched:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    if hasattr(sched, "get_jobs"):
        return {"tasks": sched.get_jobs()}
    return {"tasks": []}


@app.delete("/scheduled-tasks/{task_id}")
async def cancel_scheduled_task(task_id: str):
    """Cancel a scheduled task."""
    sched = get_scheduler()
    if not sched:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    try:
        from scheduler_service import cancel_task  # local helper keeps our in-memory index in sync

        ok = cancel_task(task_id)
        if ok:
            return {"message": f"Task {task_id} cancelled successfully"}
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Task not found")


@app.get("/weather/{city}")
async def get_weather(city: str):
    """Get weather data for a specific city using OpenWeatherMap API."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")

    try:
        # Use the weather functionality from the LLM client
        weather_data = mcp_session.llm_client.get_weather(city)

        if isinstance(weather_data, dict) and "error" in weather_data:
            raise HTTPException(status_code=400, detail=weather_data["error"])

        # Format the response for the frontend
        response: Dict[str, Any] = {
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
            "raw_data": weather_data,  # Include full data for advanced use cases
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
            response=f"❌ Error processing weather query: {str(e)}", success=False, error=str(e)
        )


@app.get("/news")
async def get_news(query: str, max_results: int = 10, language: str = "en"):
    """Get news articles from GNews API."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")

    try:
        # Use the news functionality from the LLM client
        news_data = mcp_session.llm_client.get_news(query, max_results, language)

        if isinstance(news_data, dict) and "error" in news_data:
            raise HTTPException(status_code=400, detail=news_data["error"])

        # Format the response for the frontend
        articles = news_data.get("articles", [])
        total_articles = news_data.get("totalArticles", 0)

        formatted_articles: List[Dict[str, Any]] = []
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
                    "country": article.get("source", {}).get("country", ""),
                },
            }
            formatted_articles.append(formatted_article)

        response: Dict[str, Any] = {
            "query": query,
            "total_articles": total_articles,
            "returned_articles": len(formatted_articles),
            "articles": formatted_articles,
            "information": news_data.get("information", {}),
            "raw_data": news_data,  # Include full data for advanced use cases
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
            response=f"❌ Error processing news query: {str(e)}", success=False, error=str(e)
        )


# ------------------------------------------------------------------------------
# Entrypoint (optional)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Use Railway's PORT environment variable or default to 8001
        port = int(os.getenv("PORT", 8001))
        print(f"Starting MCP API Server at http://0.0.0.0:{port}")
        uvicorn.run("mcp_api_server:app", host="0.0.0.0", port=port, reload=False, log_level="info")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)