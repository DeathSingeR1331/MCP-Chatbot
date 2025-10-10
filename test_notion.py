"""
Notion API test functions for the MCP chatbot
"""
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_VERSION = "2022-06-28"

def get_notion_headers() -> Dict[str, str]:
    if not NOTION_API_KEY:
        raise ValueError("NOTION_API_KEY not found")
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }

def test_comment_on_page(page_id: str, comment_text: str) -> bool:
    try:
        import requests
        url = "https://api.notion.com/v1/comments"
        headers = get_notion_headers()
        data = {
            "parent": {"page_id": page_id},
            "rich_text": [{"text": {"content": comment_text}}]
        }
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error adding comment: {e}")
        return False

def test_search_pages() -> List[Dict[str, Any]]:
    try:
        import requests
        url = "https://api.notion.com/v1/search"
        headers = get_notion_headers()
        data = {"filter": {"value": "page", "property": "object"}, "page_size": 10}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        print(f"❌ Error searching pages: {e}")
        return []

def test_list_databases() -> List[Dict[str, Any]]:
    try:
        import requests
        url = "https://api.notion.com/v1/search"
        headers = get_notion_headers()
        data = {"filter": {"value": "database", "property": "object"}, "page_size": 10}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        print(f"❌ Error listing databases: {e}")
        return []

def test_retrieve_comments(block_id: str) -> List[Dict[str, Any]]:
    try:
        import requests
        url = f"https://api.notion.com/v1/comments?block_id={block_id}"
        headers = get_notion_headers()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        print(f"❌ Error retrieving comments: {e}")
        return []

def test_query_database(database_id: str) -> List[Dict[str, Any]]:
    try:
        import requests
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        headers = get_notion_headers()
        data = {"page_size": 10}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        return []

def test_retrieve_page(page_id: str) -> Optional[Dict[str, Any]]:
    try:
        import requests
        url = f"https://api.notion.com/v1/pages/{page_id}"
        headers = get_notion_headers()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ Error retrieving page: {e}")
        return None

def test_get_page_blocks(block_id: str) -> List[Dict[str, Any]]:
    try:
        import requests
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        headers = get_notion_headers()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        print(f"❌ Error getting page blocks: {e}")
        return []