#!/usr/bin/env python3
"""
Simple Gmail sending test
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChatSession, Server, MultiLLMClient, Configuration
import json

async def test_gmail_send():
    """Test Gmail sending with a simple query."""
    print("🧪 Simple Gmail Send Test")
    print("=" * 40)
    
    try:
        # Load server configuration
        with open('servers_config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize all servers
        servers = []
        for name, server_config in config["mcpServers"].items():
            server = Server(name, server_config)
            servers.append(server)
        
        # Initialize LLM client
        llm_client = MultiLLMClient(Configuration())
        
        # Create chat session
        chat_session = ChatSession(servers, llm_client)
        
        # Start the session
        await chat_session.start()
        print("✅ Chat session started")
        
        # Test Gmail sending
        test_query = "send email to test@example.com about hello world"
        print(f"\n📧 Testing: {test_query}")
        print("-" * 40)
        
        result = await chat_session.chat(test_query)
        print(f"Result: {result}")
        
        print("\n✅ Gmail send test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gmail_send())
