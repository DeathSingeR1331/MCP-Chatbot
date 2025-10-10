#!/usr/bin/env python3
"""
Startup script for MCP API Server
This script starts the MCP API server that the Synapse frontend can connect to.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting MCP API Server for Synapse Frontend Integration")
    print("=" * 60)
    print("📡 Server will be available at: http://localhost:8001")
    print("🔧 MCP servers will be initialized on startup")
    print("🌐 Frontend can connect to this server for tools functionality")
    print("⚠️  Make sure your .env file has GROQ_API_KEY and GEMINI_API_KEY")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('mcp_api_server.py'):
        print("❌ Error: mcp_api_server.py not found in current directory")
        print("Please run this script from the mcp-chatbot directory")
        sys.exit(1)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("Please create a .env file with your API keys:")
        print("GROQ_API_KEY=your_groq_api_key_here")
        print("GEMINI_API_KEY=your_gemini_api_key_here")
        print("NOTION_TOKEN=your_notion_token_here (optional)")
        print()
    
    try:
        # Start the MCP API server
        print("🔄 Starting MCP API Server...")
        subprocess.run([sys.executable, "mcp_api_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 MCP API Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting MCP API Server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
