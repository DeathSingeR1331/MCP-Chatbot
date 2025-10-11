#!/usr/bin/env python3
"""
Simple test for Notion MCP tools through the API server
"""

import requests
import json
import time

def test_notion_tool_via_api(tool_name, arguments, description):
    """Test a Notion tool via the MCP API server"""
    try:
        print(f"\nTesting: {tool_name}")
        print(f"Description: {description}")
        print(f"Arguments: {json.dumps(arguments, indent=2)}")
        
        # Create the tool call JSON
        tool_call = json.dumps({
            "tool": tool_name,
            "arguments": arguments
        })
        
        # Send request to MCP API server
        url = "http://localhost:8001/execute_tool"
        headers = {"Content-Type": "application/json"}
        data = {"tool_call": tool_call}
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        execution_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS - {tool_name} executed in {execution_time:.2f}s")
            print(f"Result: {json.dumps(result, indent=2)[:300]}...")
            return True
        else:
            print(f"FAILED - {tool_name} (Status: {response.status_code})")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR testing {tool_name}: {e}")
        return False

def test_notion_tools():
    """Test all Notion MCP tools"""
    print("Starting Notion MCP Tools Test via API Server")
    print("="*60)
    
    # Test basic Notion tools
    tests = [
        {
            "tool": "API-get-self",
            "args": {},
            "desc": "Retrieve your token's bot user"
        },
        {
            "tool": "API-get-users", 
            "args": {"page_size": 5},
            "desc": "List all users (limited to 5)"
        },
        {
            "tool": "API-post-search",
            "args": {"query": "test", "page_size": 3},
            "desc": "Search by title for 'test'"
        },
        {
            "tool": "API-post-search",
            "args": {
                "filter": {"property": "object", "value": "page"},
                "page_size": 3
            },
            "desc": "Search for pages"
        },
        {
            "tool": "API-post-search",
            "args": {
                "filter": {"property": "object", "value": "database"},
                "page_size": 3
            },
            "desc": "Search for databases"
        }
    ]
    
    results = []
    for test in tests:
        success = test_notion_tool_via_api(test["tool"], test["args"], test["desc"])
        results.append({
            "tool": test["tool"],
            "success": success,
            "description": test["desc"]
        })
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    
    print(f"Total Tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        print(f"{status} | {result['tool']:<20} | {result['description']}")
    
    if successful == total:
        print("\nAll Notion MCP tools are working correctly!")
    elif successful > 0:
        print(f"\nSome Notion MCP tools are working. {failed} tools failed.")
    else:
        print("\nNo Notion MCP tools are working. Check server status and token.")

if __name__ == "__main__":
    test_notion_tools()
