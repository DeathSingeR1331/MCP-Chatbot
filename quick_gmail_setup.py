#!/usr/bin/env python3
"""
Quick Gmail Setup Script
~~~~~~~~~~~~~~~~~~~~~~~~

This script provides a quick way to set up Gmail API for the MCP chatbot.
It will guide you through the process step by step.
"""

import os
import json
import webbrowser
import time
from urllib.parse import urlparse, parse_qs
import http.server
import socketserver
import threading

def check_credentials():
    """Check if credentials.json exists."""
    if os.path.exists("credentials.json"):
        print("✅ credentials.json found")
        return True
    else:
        print("❌ credentials.json not found")
        return False

def create_credentials_guide():
    """Show user how to create credentials.json"""
    print("\n" + "="*60)
    print("📋 GMAIL API SETUP GUIDE")
    print("="*60)
    print("\n1. Go to Google Cloud Console:")
    print("   https://console.cloud.google.com/")
    print("\n2. Create a new project or select existing one")
    print("\n3. Enable Gmail API:")
    print("   - Go to 'APIs & Services' > 'Library'")
    print("   - Search for 'Gmail API' and enable it")
    print("\n4. Create OAuth 2.0 credentials:")
    print("   - Go to 'APIs & Services' > 'Credentials'")
    print("   - Click 'Create Credentials' > 'OAuth 2.0 Client IDs'")
    print("   - Choose 'Web application'")
    print("   - Add redirect URI: http://localhost:8080/callback")
    print("   - Download the JSON file")
    print("\n5. Rename the downloaded file to 'credentials.json'")
    print("   and place it in this directory")
    print("\n6. Run this script again")
    print("\n" + "="*60)

def test_gmail_api():
    """Test if Gmail API is working."""
    try:
        from gmail_service import send_message
        result = send_message('test_user', 'test@example.com', 'Test', 'Test body')
        if result.get('authorized'):
            print("✅ Gmail API is working correctly!")
            return True
        else:
            print("❌ Gmail API not authorized")
            return False
    except Exception as e:
        print(f"❌ Gmail API test failed: {e}")
        return False

def setup_oauth():
    """Set up OAuth flow."""
    try:
        from gmail_service import start_oauth, finish_oauth
        
        print("\n🚀 Starting OAuth setup...")
        
        # Start callback server
        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/callback'):
                    parsed_url = urlparse(self.path)
                    query_params = parse_qs(parsed_url.query)
                    code = query_params.get('code', [None])[0]
                    state = query_params.get('state', [None])[0]
                    
                    if code and state:
                        try:
                            user_id = finish_oauth(state, code, "http://localhost:8080/callback")
                            if user_id:
                                self.send_response(200)
                                self.send_header('Content-type', 'text/html')
                                self.end_headers()
                                html_content = """
                                <html><body>
                                <h1>✅ Gmail Authorization Successful!</h1>
                                <p>You can now close this window.</p>
                                </body></html>
                                """
                                self.wfile.write(html_content.encode('utf-8'))
                                print(f"✅ Authorization successful for user: {user_id}")
                                return
                        except Exception as e:
                            print(f"❌ OAuth failed: {e}")
                    
                    self.send_response(400)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                pass
        
        # Start server
        server_thread = threading.Thread(
            target=lambda: socketserver.TCPServer(("", 8080), CallbackHandler).serve_forever(),
            daemon=True
        )
        server_thread.start()
        time.sleep(1)
        
        # Start OAuth
        user_id = "default_user"
        redirect_uri = "http://localhost:8080/callback"
        auth_urls = start_oauth(user_id, redirect_uri)
        
        print(f"\n🌐 Opening browser for authorization...")
        print(f"URL: {auth_urls.auth_url}")
        
        webbrowser.open(auth_urls.auth_url)
        
        print("\n⏳ Complete the authorization in your browser...")
        print("Press Ctrl+C when done or if you want to cancel.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n✅ Setup completed!")
            
    except Exception as e:
        print(f"❌ OAuth setup failed: {e}")

def main():
    """Main setup function."""
    print("🔧 Quick Gmail API Setup")
    print("=" * 30)
    
    # Step 1: Check credentials
    if not check_credentials():
        create_credentials_guide()
        return
    
    # Step 2: Test API
    if test_gmail_api():
        print("\n🎉 Gmail API is already working!")
        return
    
    # Step 3: Set up OAuth
    print("\n🔐 Gmail API needs authorization...")
    setup_oauth()
    
    # Step 4: Test again
    print("\n🧪 Testing Gmail API after setup...")
    if test_gmail_api():
        print("\n🎉 Gmail API setup complete! You can now send emails.")
    else:
        print("\n❌ Setup incomplete. Please check your credentials and try again.")

if __name__ == "__main__":
    main()
