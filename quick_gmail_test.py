#!/usr/bin/env python3
"""
Quick Gmail functionality test
"""

import asyncio
import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChatSession, Server, MultiLLMClient, Configuration

async def test_gmail_functionality():
    """Test Gmail functionality through the chat session."""
    print("🧪 Quick Gmail Functionality Test")
    print("=" * 50)
    
    try:
        # Load server configuration
        with open('servers_config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize only Gmail server
        servers = []
        for name, server_config in config["mcpServers"].items():
            if name == "gmail":
                server = Server(name, server_config)
                servers.append(server)
        
        # Initialize LLM client
        llm_client = MultiLLMClient(Configuration())
        
        # Create chat session
        chat_session = ChatSession(servers, llm_client)
        
        # Start the session
        await chat_session.start()
        print("✅ Chat session started")
        
        # Test 1: Read emails
        print("\n📧 Testing Gmail read functionality...")
        result = await chat_session._handle_gmail_read_via_mcp("read my emails")
        print(f"Result: {result}")
        
        # Test 2: Search emails
        print("\n🔍 Testing Gmail search functionality...")
        result = await chat_session._handle_gmail_search_via_mcp("test")
        print(f"Result: {result}")
        
        # Test 3: List labels
        print("\n🏷️ Testing Gmail labels functionality...")
        result = await chat_session._handle_gmail_labels_via_mcp()
        print(f"Result: {result}")
        
        print("\n✅ All Gmail tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gmail_functionality())
