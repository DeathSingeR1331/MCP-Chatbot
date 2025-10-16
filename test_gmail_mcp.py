#!/usr/bin/env python3
"""
Gmail MCP Integration Test Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script tests the Gmail AutoAuth MCP Server integration to ensure all
Gmail functionality is working correctly.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChatSession, Server, MultiLLMClient, LLMProvider, Configuration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GmailMCPTester:
    """Test Gmail MCP integration."""
    
    def __init__(self):
        self.chat_session = None
        self.gmail_server = None
    
    async def setup(self):
        """Set up the test environment."""
        print("🔧 Setting up Gmail MCP test environment...")
        
        # Load server configuration
        with open("servers_config.json", "r") as f:
            config = json.load(f)
        
        # Initialize servers
        servers = []
        for name, server_config in config["mcpServers"].items():
            if name == "gmail":  # Only test Gmail server
                server = Server(name, server_config)
                servers.append(server)
        
        # Initialize LLM client (minimal setup for testing)
        config = Configuration()
        llm_client = MultiLLMClient(config)
        
        # Create chat session
        self.chat_session = ChatSession(servers, llm_client)
        
        # Initialize Gmail server
        if servers:
            await servers[0].initialize()
            self.gmail_server = servers[0]
            print(f"✅ Gmail MCP server '{self.gmail_server.name}' initialized")
        else:
            print("❌ No Gmail server found in configuration")
            return False
        
        return True
    
    async def test_gmail_server_connection(self) -> bool:
        """Test if Gmail MCP server is connected."""
        print("\n🔍 Testing Gmail MCP server connection...")
        
        if not self.gmail_server or not self.gmail_server.session:
            print("❌ Gmail MCP server not connected")
            return False
        
        try:
            # Test server connection
            tools = await self.gmail_server.list_tools()
            gmail_tools = [tool.name for tool in tools if 'gmail' in tool.name.lower() or tool.name in [
                'send_email', 'read_email', 'search_emails', 'draft_email', 
                'list_email_labels', 'create_label', 'delete_email', 'modify_email',
                'list_filters', 'create_filter'
            ]]
            
            print(f"✅ Gmail MCP server connected")
            print(f"✅ Found {len(gmail_tools)} Gmail tools: {gmail_tools}")
            return True
            
        except Exception as e:
            print(f"❌ Gmail MCP server connection failed: {e}")
            return False
    
    async def test_gmail_search(self) -> bool:
        """Test Gmail search functionality."""
        print("\n🔍 Testing Gmail search...")
        
        try:
            result = await self.gmail_server.execute_tool("search_emails", {
                "query": "in:inbox",
                "maxResults": 5
            })
            
            if result:
                print("✅ Gmail search successful")
                if result.get("messages"):
                    print(f"✅ Found {len(result['messages'])} emails")
                else:
                    print("ℹ️ No emails found (this is normal for empty inbox)")
                return True
            else:
                print("❌ Gmail search failed")
                return False
                
        except Exception as e:
            print(f"❌ Gmail search error: {e}")
            return False
    
    async def test_gmail_labels(self) -> bool:
        """Test Gmail labels functionality."""
        print("\n🏷️ Testing Gmail labels...")
        
        try:
            result = await self.gmail_server.execute_tool("list_email_labels", {})
            
            if result:
                print("✅ Gmail labels listing successful")
                if result.get("labels"):
                    print(f"✅ Found {len(result['labels'])} labels")
                else:
                    print("ℹ️ No labels found")
                return True
            else:
                print("❌ Gmail labels listing failed")
                return False
                
        except Exception as e:
            print(f"❌ Gmail labels error: {e}")
            return False
    
    async def test_gmail_filters(self) -> bool:
        """Test Gmail filters functionality."""
        print("\n🔍 Testing Gmail filters...")
        
        try:
            result = await self.gmail_server.execute_tool("list_filters", {})
            
            if result:
                print("✅ Gmail filters listing successful")
                if result.get("filters"):
                    print(f"✅ Found {len(result['filters'])} filters")
                else:
                    print("ℹ️ No filters found")
                return True
            else:
                print("❌ Gmail filters listing failed")
                return False
                
        except Exception as e:
            print(f"❌ Gmail filters error: {e}")
            return False
    
    async def test_gmail_draft_creation(self) -> bool:
        """Test Gmail draft creation (without sending)."""
        print("\n📝 Testing Gmail draft creation...")
        
        try:
            result = await self.gmail_server.execute_tool("draft_email", {
                "to": ["test@example.com"],
                "subject": "Test Draft - Gmail MCP Integration",
                "body": "This is a test draft created by the Gmail MCP integration test script."
            })
            
            if result and result.get("success"):
                print("✅ Gmail draft creation successful")
                print(f"✅ Draft ID: {result.get('draftId', 'N/A')}")
                return True
            else:
                print(f"❌ Gmail draft creation failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Gmail draft creation error: {e}")
            return False
    
    async def test_chat_session_gmail_handlers(self) -> bool:
        """Test Gmail handlers through chat session."""
        print("\n💬 Testing Gmail handlers through chat session...")
        
        try:
            # Test Gmail read handler
            result = await self.chat_session._handle_gmail_read_via_mcp("read my gmail emails")
            if "❌" not in result:
                print("✅ Gmail read handler working")
            else:
                print(f"❌ Gmail read handler failed: {result}")
                return False
            
            # Test Gmail search handler
            result = await self.chat_session._handle_gmail_search_via_mcp("test")
            if "❌" not in result:
                print("✅ Gmail search handler working")
            else:
                print(f"❌ Gmail search handler failed: {result}")
                return False
            
            # Test Gmail labels handler
            result = await self.chat_session._handle_gmail_labels_via_mcp()
            if "❌" not in result:
                print("✅ Gmail labels handler working")
            else:
                print(f"❌ Gmail labels handler failed: {result}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Chat session Gmail handlers error: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test environment."""
        if self.gmail_server:
            await self.gmail_server.cleanup()
        print("\n🧹 Test environment cleaned up")

async def main():
    """Run all Gmail MCP tests."""
    print("🧪 Gmail MCP Integration Test Suite")
    print("=" * 50)
    
    tester = GmailMCPTester()
    
    try:
        # Setup
        if not await tester.setup():
            print("\n❌ Setup failed. Exiting.")
            return False
        
        # Run tests
        tests = [
            ("Gmail Server Connection", tester.test_gmail_server_connection),
            ("Gmail Search", tester.test_gmail_search),
            ("Gmail Labels", tester.test_gmail_labels),
            ("Gmail Filters", tester.test_gmail_filters),
            ("Gmail Draft Creation", tester.test_gmail_draft_creation),
            ("Chat Session Gmail Handlers", tester.test_chat_session_gmail_handlers),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All Gmail MCP integration tests passed!")
            print("✅ Gmail AutoAuth MCP Server is working correctly!")
            return True
        else:
            print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        return False
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
