import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Your Notion token
NOTION_TOKEN = "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"

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def test_search_pages():
    """Test 1: Search for pages"""
    print("🔍 Searching for pages...")
    search_url = "https://api.notion.com/v1/search"
    search_data = {
        "query": "Habit tracker",
        "filter": {
            "value": "page",
            "property": "object"
        }
    }

    response = requests.post(search_url, headers=headers, json=search_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        pages = data.get('results', [])
        print(f"Found {len(pages)} pages:")
        for page in pages:
            title = page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('text', {}).get('content', 'No title')
            page_id = page.get('id', 'No ID')
            print(f"  📄 {title} (ID: {page_id})")
        return pages
    else:
        print(f"Error: {response.json()}")
        return []

def test_list_databases():
    """Test 2: List databases"""
    print("\n📊 Listing databases...")
    db_url = "https://api.notion.com/v1/search"
    db_data = {
        "filter": {
            "value": "database",
            "property": "object"
        }
    }

    response = requests.post(db_url, headers=headers, json=db_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        databases = data.get('results', [])
        print(f"Found {len(databases)} databases:")
        for db in databases:
            title = db.get('title', [{}])[0].get('text', {}).get('content', 'No title')
            db_id = db.get('id', 'No ID')
            print(f"  🗃️ {title} (ID: {db_id})")
        return databases
    else:
        print(f"Error: {response.json()}")
        return []

def test_comment_on_page(page_id, comment_text):
    """Test 3: Add comment to a specific page"""
    print(f"\n💬 Testing comment on page {page_id}...")
    comment_url = f"https://api.notion.com/v1/comments"
    comment_data = {
        "parent": {
            "page_id": page_id
        },
        "rich_text": [
            {
                "text": {
                    "content": comment_text
                }
            }
        ]
    }

    response = requests.post(comment_url, headers=headers, json=comment_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Comment added successfully!")
        print(f"Comment ID: {data.get('id', 'No ID')}")
        print(f"Comment text: {comment_text}")
        return True
    else:
        print(f"❌ Comment failed: {response.json()}")
        return False

def test_retrieve_comments(page_id):
    """Test 4: Retrieve comments from a page"""
    print(f"\n📝 Retrieving comments from page {page_id}...")
    comments_url = f"https://api.notion.com/v1/comments?block_id={page_id}"
    
    response = requests.get(comments_url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        comments = data.get('results', [])
        print(f"Found {len(comments)} comments:")
        for comment in comments:
            text = comment.get('rich_text', [{}])[0].get('text', {}).get('content', 'No text')
            created_time = comment.get('created_time', 'No time')
            print(f"  💭 {text} (Created: {created_time})")
        return comments
    else:
        print(f"Error: {response.json()}")
        return []

def test_query_database(db_id):
    """Test: Query database entries"""
    print(f"\n🔍 Querying database {db_id}...")
    query_url = f"https://api.notion.com/v1/databases/{db_id}/query"
    query_data = {
        "page_size": 5
    }
    
    response = requests.post(query_url, headers=headers, json=query_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f"✅ Found {len(results)} entries in database")
        for i, entry in enumerate(results[:3]):  # Show first 3
            # Try different title property names
            title_props = ['Title', 'title', 'Name', 'name', 'Project', 'Company']
            title = "No title found"
            for prop in title_props:
                if prop in entry.get('properties', {}):
                    title_obj = entry['properties'][prop]
                    if 'title' in title_obj:
                        title = title_obj['title'][0].get('text', {}).get('content', f'Entry {i+1}')
                        break
            print(f"  📝 {title}")
        return results
    else:
        print(f"❌ Database query failed: {response.json()}")
        return []

def test_retrieve_page(page_id):
    """Test: Retrieve page details"""
    print(f"\n📖 Retrieving page {page_id}...")
    page_url = f"https://api.notion.com/v1/pages/{page_id}"
    
    response = requests.get(page_url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        title = data.get('properties', {}).get('title', {}).get('title', [{}])[0].get('text', {}).get('content', 'No title')
        print(f"✅ Page retrieved: {title}")
        return data
    else:
        print(f"❌ Page retrieval failed: {response.json()}")
        return None

def test_get_page_blocks(page_id):
    """Test: Get page content blocks"""
    print(f"\n📝 Getting blocks from page {page_id}...")
    blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    
    response = requests.get(blocks_url, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        blocks = data.get('results', [])
        print(f"✅ Found {len(blocks)} blocks:")
        for i, block in enumerate(blocks[:3]):  # Show first 3
            block_type = block.get('type', 'unknown')
            print(f"  📦 Block {i+1}: {block_type}")
        return blocks
    else:
        print(f"❌ Block retrieval failed: {response.json()}")
        return []

def main():
    print("🚀 NOTION API COMPREHENSIVE TEST")
    print("=" * 50)
    
    # Test 1: Search pages
    pages = test_search_pages()
    
    # Test 2: List databases  
    databases = test_list_databases()
    
    # Test 3: Comment on specific page (using the page ID you provided)
    test_page_id = "1038ad69772680f085f4d2feadd19170"
    test_comment = "Weekly review done - Test from Python script"
    
    print(f"\n🎯 TESTING COMMENT FUNCTIONALITY")
    print(f"Target Page ID: {test_page_id}")
    print(f"Comment Text: {test_comment}")
    
    # Try to add comment
    comment_success = test_comment_on_page(test_page_id, test_comment)
    
    # Test 4: Retrieve comments to verify
    if comment_success:
        test_retrieve_comments(test_page_id)
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"✅ Pages found: {len(pages)}")
    print(f"✅ Databases found: {len(databases)}")
    print(f"✅ Comment test: {'PASSED' if comment_success else 'FAILED'}")
    
    print("\n🎯 EXACT PROMPTS FOR CHATBOT TESTING:")
    print("1. Add a comment 'Weekly review done' to Notion page 1038ad69772680f085f4d2feadd19170")
    print("2. Comment 'Test from chatbot' on page 1038ad69772680f085f4d2feadd19170")
    print("3. Add comment 'Python script test successful' to Notion page 1038ad69772680f085f4d2feadd19170")

if __name__ == "__main__":
    main()