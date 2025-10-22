#!/usr/bin/env python3
"""
Simple test server to verify Railway deployment works
"""

import os
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="MCP Test Server")

@app.get("/")
async def root():
    return {"message": "MCP Chatbot Test Server is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mcp-chatbot"}

@app.get("/test")
async def test():
    return {
        "message": "Test endpoint working",
        "environment": {
            "PORT": os.getenv("PORT", "8001"),
            "GROQ_API_KEY": "present" if os.getenv("GROQ_API_KEY") else "missing",
            "GEMINI_API_KEY": "present" if os.getenv("GEMINI_API_KEY") else "missing"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"Starting test server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
