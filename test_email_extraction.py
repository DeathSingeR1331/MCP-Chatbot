#!/usr/bin/env python3
"""
Test email data extraction patterns
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChatSession, Server, MultiLLMClient, Configuration

def test_email_extraction():
    """Test email data extraction with various patterns."""
    print("🧪 Email Data Extraction Test")
    print("=" * 50)
    
    # Create a minimal chat session just for testing the extraction method
    chat_session = ChatSession([], None)
    
    # Test cases
    test_cases = [
        "send hello to gvkss29@gmail.com",
        "send email to john@example.com about meeting",
        "send hello world to jane@company.com",
        "email client@company.com saying project complete",
        "compose email to boss@company.com about quarterly report",
        "send message to team@company.com",
        "send hello to test@example.com",
        "send good morning to friend@email.com"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📧 Test {i}: '{test_case}'")
        print("-" * 40)
        
        email_data = chat_session._extract_gmail_data(test_case.lower())
        if email_data:
            print(f"✅ Successfully extracted:")
            print(f"   To: {email_data['to']}")
            print(f"   Subject: {email_data['subject']}")
            print(f"   Body: {email_data['body']}")
        else:
            print("❌ Failed to extract email data")
    
    print(f"\n✅ Email extraction tests completed!")

if __name__ == "__main__":
    test_email_extraction()
