#!/usr/bin/env python3
"""
Gmail API Test Script
~~~~~~~~~~~~~~~~~~~~~

This script tests the Gmail API functionality to ensure it's working correctly.
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported."""
    try:
        from gmail_service import start_oauth, finish_oauth, list_unread, send_message
        print("✅ Gmail service imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_credentials():
    """Test if credentials.json exists and is valid."""
    if not os.path.exists("credentials.json"):
        print("❌ credentials.json not found")
        print("   Please download OAuth 2.0 credentials from Google Cloud Console")
        print("   and save as 'credentials.json' in this directory")
        return False
    
    try:
        import json
        with open("credentials.json", "r") as f:
            creds = json.load(f)
        
        if "web" not in creds or "client_id" not in creds["web"]:
            print("❌ Invalid credentials.json format")
            print("   Make sure you downloaded OAuth 2.0 credentials for a Web application")
            return False
        
        print("✅ credentials.json is valid")
        return True
    except Exception as e:
        print(f"❌ Error reading credentials.json: {e}")
        return False

def test_oauth_tokens():
    """Test if OAuth tokens exist."""
    tokens_dir = "tokens"
    if not os.path.exists(tokens_dir):
        print("❌ No tokens directory found")
        print("   You need to complete the OAuth flow first")
        return False
    
    token_files = [f for f in os.listdir(tokens_dir) if f.endswith('.json')]
    if not token_files:
        print("❌ No OAuth tokens found")
        print("   You need to complete the OAuth flow first")
        return False
    
    print(f"✅ Found {len(token_files)} OAuth token(s)")
    return True

def test_gmail_api():
    """Test Gmail API functionality."""
    try:
        from gmail_service import send_message, list_unread
        
        # Test send message (this will fail if not authorized, but we can check the response)
        result = send_message('test_user', 'test@example.com', 'Test Subject', 'Test Body')
        
        if result.get('authorized'):
            print("✅ Gmail API send test successful")
            return True
        else:
            print("❌ Gmail API not authorized")
            print(f"   Response: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Gmail API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Gmail API Test Suite")
    print("=" * 30)
    
    tests = [
        ("Import Test", test_imports),
        ("Credentials Test", test_credentials),
        ("OAuth Tokens Test", test_oauth_tokens),
        ("Gmail API Test", test_gmail_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 30)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Gmail API is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please run 'python quick_gmail_setup.py' to fix the issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
