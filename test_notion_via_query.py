#!/usr/bin/env python3
"""
Test Notion MCP tools via the query endpoint
"""

import requests
import json
import time

def test_notion_via_query(query, description):
    """Test Notion functionality via natural language query"""
    try:
        print(f"\nTesting: {description}")
        print(f"Query: {query}")
        
        url = "http://localhost:8001/query"
        headers = {"Content-Type": "application/json"}
        data = {
            "query": query,
            "user_id": "test_user"
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        execution_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            success = result.get("success", False)
            response_text = result.get("response", "")
            
            if success:
                print(f"SUCCESS - Query executed in {execution_time:.2f}s")
                print(f"Response: {response_text[:300]}...")
                return True
            else:
                print(f"FAILED - Query failed")
                print(f"Response: {response_text}")
                return False
        else:
            print(f"FAILED - HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_notion_tools():
    """Test Notion MCP tools via natural language queries"""
    print("Testing Notion MCP Tools via Natural Language Queries")
    print("="*70)
    
    # Test various Notion operations
    tests = [
        {
            "query": "Get my Notion user information",
            "description": "Test API-get-self (get bot user info)"
        },
        {
            "query": "List all users in my Notion workspace",
            "description": "Test API-get-users (list all users)"
        },
        {
            "query": "Search for pages in my Notion workspace",
            "description": "Test API-post-search (search pages)"
        },
        {
            "query": "Find databases in my Notion workspace",
            "description": "Test API-post-search (search databases)"
        },
        {
            "query": "Search for content about 'test' in Notion",
            "description": "Test API-post-search (search by query)"
        },
        {
            "query": "Show me all my Notion pages",
            "description": "Test API-post-search (list all pages)"
        },
        {
            "query": "What databases do I have in Notion?",
            "description": "Test API-post-search (list databases)"
        }
    ]
    
    results = []
    for test in tests:
        success = test_notion_via_query(test["query"], test["description"])
        results.append({
            "description": test["description"],
            "query": test["query"],
            "success": success
        })
        
        # Add a small delay between requests
        time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
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
        print(f"{status} | {result['description']}")
        if not result["success"]:
            print(f"     Query: {result['query']}")
    
    if successful == total:
        print("\nAll Notion MCP tools are working correctly!")
        print("The Notion integration is fully functional.")
    elif successful > 0:
        print(f"\nSome Notion MCP tools are working. {failed} tests failed.")
        print("Check the failed tests above for details.")
    else:
        print("\nNo Notion MCP tools are working.")
        print("Possible issues:")
        print("1. Notion token is invalid or expired")
        print("2. Notion integration doesn't have proper permissions")
        print("3. MCP server is not properly initialized")
        print("4. Network connectivity issues")

if __name__ == "__main__":
    test_notion_tools()
