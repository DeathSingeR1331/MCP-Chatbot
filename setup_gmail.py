#!/usr/bin/env python3
"""
Gmail API Setup Script
~~~~~~~~~~~~~~~~~~~~~~

This script helps you set up Gmail API credentials for the MCP chatbot.
It will guide you through the OAuth flow to authorize Gmail access.

Usage:
    python setup_gmail.py

Requirements:
    1. Google Cloud Console project with Gmail API enabled
    2. OAuth 2.0 credentials downloaded as credentials.json
    3. credentials.json file in the same directory as this script
"""

import os
import json
import webbrowser
from urllib.parse import urlparse, parse_qs
import http.server
import socketserver
import threading
import time

from gmail_service import start_oauth, finish_oauth

def check_credentials():
    """Check if credentials.json exists and is valid."""
    creds_path = "credentials.json"
    if not os.path.exists(creds_path):
        print("❌ credentials.json not found!")
        print("\nTo set up Gmail API:")
        print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
        print("2. Create a new project or select existing one")
        print("3. Enable Gmail API")
        print("4. Create OAuth 2.0 credentials (Web application)")
        print("5. Download the JSON file and rename it to 'credentials.json'")
        print("6. Place it in this directory")
        return False
    
    try:
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        
        if 'web' not in creds or 'client_id' not in creds['web']:
            print("❌ Invalid credentials.json format!")
            print("Make sure you downloaded OAuth 2.0 credentials for a Web application.")
            return False
        
        print("✅ credentials.json found and valid")
        return True
    except Exception as e:
        print(f"❌ Error reading credentials.json: {e}")
        return False

class CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Handle OAuth callback."""
    
    def do_GET(self):
        """Handle GET request from OAuth callback."""
        if self.path.startswith('/callback'):
            # Parse the callback URL
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            
            # Extract code and state
            code = query_params.get('code', [None])[0]
            state = query_params.get('state', [None])[0]
            
            if code and state:
                # Complete OAuth flow
                try:
                    user_id = finish_oauth(state, code, "http://localhost:8080/callback")
                    if user_id:
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"""
                        <html>
                        <body>
                        <h1>✅ Gmail Authorization Successful!</h1>
                        <p>You can now close this window and use Gmail features in the chatbot.</p>
                        </body>
                        </html>
                        """)
                        print(f"\n✅ Gmail authorization successful for user: {user_id}")
                        return
                except Exception as e:
                    print(f"❌ OAuth completion failed: {e}")
            
            # Error case
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <body>
            <h1>❌ Authorization Failed</h1>
            <p>Please try again.</p>
            </body>
            </html>
            """)
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

def start_callback_server():
    """Start a local server to handle OAuth callback."""
    try:
        with socketserver.TCPServer(("", 8080), CallbackHandler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(f"❌ Failed to start callback server: {e}")

def main():
    """Main setup function."""
    print("🔧 Gmail API Setup for MCP Chatbot")
    print("=" * 40)
    
    # Check credentials
    if not check_credentials():
        return
    
    print("\n🚀 Starting Gmail authorization...")
    
    # Start callback server in a separate thread
    server_thread = threading.Thread(target=start_callback_server, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(1)
    
    try:
        # Start OAuth flow
        user_id = "default_user"  # You can customize this
        redirect_uri = "http://localhost:8080/callback"
        
        auth_urls = start_oauth(user_id, redirect_uri)
        
        print(f"\n🌐 Opening browser for Gmail authorization...")
        print(f"Authorization URL: {auth_urls.auth_url}")
        
        # Open browser
        webbrowser.open(auth_urls.auth_url)
        
        print("\n⏳ Waiting for authorization...")
        print("Please complete the authorization in your browser.")
        print("The server will automatically handle the callback.")
        
        # Keep the script running to handle the callback
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n✅ Setup completed! You can now use Gmail features.")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()
