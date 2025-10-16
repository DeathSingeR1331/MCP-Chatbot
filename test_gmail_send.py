#!/usr/bin/env python3
"""
Test Gmail sending functionality
"""

import asyncio
import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChatSession, Server, MultiLLMClient, Configuration

async def test_gmail_sending():
    """Test Gmail sending functionality."""
    print("🧪 Gmail Sending Test")
    print("=" * 50)
    
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
        
        # Test different Gmail sending patterns
        test_queries = [
            "send email to test@example.com about hello world",
            "email john@example.com saying how are you",
            "compose email to jane@example.com about meeting tomorrow",
            "write email to boss@company.com subject: Project Update body: The project is on track"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📧 Test {i}: {query}")
            print("-" * 40)
            
            # Test email data extraction
            email_data = chat_session._extract_gmail_data(query.lower())
            if email_data:
                print(f"✅ Email data extracted:")
                print(f"   To: {email_data['to']}")
                print(f"   Subject: {email_data['subject']}")
                print(f"   Body: {email_data['body']}")
            else:
                print("❌ Failed to extract email data")
        
        print("\n✅ Gmail sending tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gmail_sending())
