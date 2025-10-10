#!/usr/bin/env python3
"""
MCP API Server for Synapse Frontend Integration
This server runs MCP servers locally and provides an API for the frontend to use tools.
"""

import asyncio
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Any
from enum import Enum
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import from main.py
from main import Configuration, LLMProvider, MultiLLMClient, Server, ChatSession
from scheduler_service import initialize_scheduler, get_scheduler, shutdown_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for MCP session
mcp_session: Optional[ChatSession] = None
initialized = False

class QueryRequest(BaseModel):
    query: str
    user_id: str

class QueryResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

# FastAPI app
app = FastAPI(title="MCP API Server", version="1.0.0", lifespan=lifespan)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def startup_event():
    """Initialize MCP servers on startup."""
    global mcp_session, initialized
    
    try:
        logging.info("🚀 Starting MCP API Server...")
        
        # Load configuration
        config = Configuration()
        
        # Initialize LLM client
        llm_client = MultiLLMClient(config)
        
        # Load servers configuration
        with open('servers_config.json', 'r') as f:
            servers_config = json.load(f).get("mcpServers", {})
        
        # Initialize MCP servers
        servers = []
        logging.info("🔌 Initializing MCP servers...")
        
        for server_name, server_params in servers_config.items():
            command = server_params["command"]
            args = server_params.get("args", [])
            env = server_params.get("env", {})
            
            # Inject Notion token if available
            if server_name == "notionApi" and config.notion_token:
                env["NOTION_TOKEN"] = config.notion_token
                logging.info("Injected NOTION_TOKEN into notionApi server environment.")
            
            
            logging.info(f"Starting MCP server: {server_name} with command: {command} {args}")
            try:
                # Create config dictionary for Server constructor
                server_config = {
                    "command": command,
                    "args": args,
                    "env": env
                }
                server = Server(server_name, server_config)
                servers.append(server)
                logging.info(f"MCP server '{server_name}' created successfully.")
            except Exception as e:
                logging.error(f"Failed to create MCP server '{server_name}': {e}")
        
        # Initialize servers
        initialized_servers = 0
        for server in servers:
            try:
                await server.initialize()
                initialized_servers += 1
                logging.info(f"✓ Server '{server.name}' initialized")
            except Exception as e:
                logging.error(f"✗ Failed to initialize server '{server.name}': {e}")
        
        if initialized_servers == 0:
            logging.warning("⚠️ No MCP servers were initialized. Continuing without tools...")
        
        # Create chat session
        mcp_session = ChatSession(servers, llm_client)
        
        # Load tools
        logging.info("🔧 Loading tools...")
        all_tools = []
        for server in servers:
            if server.session:
                try:
                    tools = await server.list_tools()
                    all_tools.extend(tools)
                    logging.info(f"✓ Loaded {len(tools)} tools from '{server.name}'")
                except Exception as e:
                    logging.warning(f"✗ Could not load tools from '{server.name}': {e}")
        
        logging.info(f"📦 Total tools available: {len(all_tools)}")
        
        # Set Groq as primary provider
        if LLMProvider.GROQ in llm_client.available_providers:
            llm_client.current_provider = LLMProvider.GROQ
            logging.info("🎯 Set Groq as primary LLM provider")
        elif LLMProvider.GEMINI in llm_client.available_providers:
            llm_client.current_provider = LLMProvider.GEMINI
            logging.info("🎯 Set Gemini as primary LLM provider")
        
        initialized = True
        logging.info("✅ MCP API Server initialized successfully!")
        
        # Initialize scheduler service
        async def mcp_executor(action: str, user_id: str) -> bool:
            """Execute MCP action for scheduled tasks"""
            try:
                # Clean the action by removing time-related parts
                cleaned_action = action
                
                # Remove common time expressions from the action
                time_patterns = [
                    r'\s+in\s+\d+\s+(minutes?|hours?|days?|seconds?)',
                    r'\s+at\s+\d{1,2}:\d{2}\s*(am|pm)?',
                    r'\s+at\s+\d{1,2}\s*(am|pm)',
                    r'\s+daily\s+',
                    r'\s+weekly\s+',
                    r'\s+monthly\s+',
                    r'\s+every\s+day\s+',
                    r'\s+every\s+week\s+',
                    r'\s+every\s+month\s+'
                ]
                
                for pattern in time_patterns:
                    cleaned_action = re.sub(pattern, '', cleaned_action, flags=re.IGNORECASE)
                
                # Clean up extra spaces
                cleaned_action = ' '.join(cleaned_action.split())
                
                logging.info(f"🎵 Executing cleaned action: '{cleaned_action}'")
                result = await mcp_session.chat(cleaned_action)
                return result is not None
            except Exception as e:
                logging.error(f"Error executing scheduled action '{action}': {e}")
                return False
        
        initialize_scheduler(mcp_executor)
        logging.info("🕐 Scheduler service initialized")
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize MCP API Server: {e}")
        initialized = False

async def shutdown_event():
    """Clean up MCP servers on shutdown."""
    global mcp_session, initialized
    
    # Shutdown scheduler service
    shutdown_scheduler()
    logging.info("🕐 Scheduler service stopped")
    
    if mcp_session and mcp_session.servers:
        logging.info("🧹 Cleaning up MCP servers...")
        cleanup_tasks = []
        for server in mcp_session.servers:
            cleanup_tasks.append(server.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logging.info("✅ MCP servers cleaned up")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if initialized else "not_initialized",
        "initialized": initialized,
        "servers_count": len(mcp_session.servers) if mcp_session else 0
    }

@app.get("/tools")
async def list_tools():
    """List available MCP tools."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
    try:
        all_tools = []
        for server in mcp_session.servers:
            if server.session:
                try:
                    tools = await server.list_tools()
                    for tool in tools:
                        all_tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "server": server.name
                        })
                except Exception as e:
                    logging.warning(f"Could not list tools from {server.name}: {e}")
        
        return {"tools": all_tools, "count": len(all_tools)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using MCP tools and LLM."""
    if not initialized or not mcp_session:
        raise HTTPException(status_code=503, detail="MCP servers not initialized")
    
    try:
        logging.info(f"🔧 Processing tools query: {request.query}")
        
        # Check if this is a scheduling request
        scheduler = get_scheduler()
        if scheduler:
            # Check for scheduling keywords
            scheduling_keywords = ['in ', 'at ', 'daily', 'weekly', 'monthly', 'every', 'schedule', 'remind']
            if any(keyword in request.query.lower() for keyword in scheduling_keywords):
                # Try to schedule the task
                task_id = scheduler.schedule_task(request.user_id, request.query, request.query)
                if task_id:
                    return QueryResponse(
                        response=f"✅ Task scheduled successfully! I'll execute '{request.query}' at the specified time. Task ID: {task_id}",
                        success=True
                    )
        
        # Regular MCP processing
        response = await mcp_session.chat(request.query)
        
        return QueryResponse(
            response=response,
            success=True
        )
        
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return QueryResponse(
            response=f"❌ Error processing tools request: {str(e)}",
            success=False,
            error=str(e)
        )

@app.get("/status")
async def get_status():
    """Get detailed status of MCP servers and tools."""
    if not initialized or not mcp_session:
        return {
            "initialized": False,
            "servers": [],
            "tools": [],
            "llm_provider": None
        }
    
    try:
        servers_status = []
        all_tools = []
        
        for server in mcp_session.servers:
            server_info = {
                "name": server.name,
                "initialized": server.session is not None,
                "tools": []
            }
            
            if server.session:
                try:
                    tools = await server.list_tools()
                    for tool in tools:
                        tool_info = {
                            "name": tool.name,
                            "description": tool.description
                        }
                        server_info["tools"].append(tool_info)
                        all_tools.append(tool_info)
                except Exception as e:
                    server_info["error"] = str(e)
            
            servers_status.append(server_info)
        
        return {
            "initialized": True,
            "servers": servers_status,
            "tools": all_tools,
            "llm_provider": mcp_session.llm_client.current_provider.value if mcp_session.llm_client.current_provider else None,
            "available_providers": [p.value for p in mcp_session.llm_client.available_providers]
        }
        
    except Exception as e:
        return {
            "initialized": True,
            "error": str(e),
            "servers": [],
            "tools": [],
            "llm_provider": None
        }

@app.get("/scheduled-tasks")
async def get_scheduled_tasks(user_id: str = None):
    """Get scheduled tasks, optionally filtered by user."""
    scheduler = get_scheduler()
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    tasks = scheduler.get_scheduled_tasks(user_id)
    return {"tasks": tasks}

@app.delete("/scheduled-tasks/{task_id}")
async def cancel_scheduled_task(task_id: str):
    """Cancel a scheduled task."""
    scheduler = get_scheduler()
    if not scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    success = scheduler.cancel_task(task_id)
    if success:
        return {"message": f"Task {task_id} cancelled successfully"}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    print("🚀 Starting MCP API Server...")
    print("📡 Server will be available at: http://localhost:8001")
    print("🔧 MCP servers will be initialized on startup")
    print("🌐 Frontend can connect to this server for tools functionality")
    print("\n" + "="*60)
    
    uvicorn.run(
        "mcp_api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
