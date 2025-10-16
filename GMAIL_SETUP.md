# Gmail API Setup Guide

This guide will help you set up Gmail API integration for the MCP chatbot, replacing the browser automation with proper Gmail API calls.

## Prerequisites

1. **Google Account** with Gmail access
2. **Google Cloud Console** access
3. **Python environment** with required dependencies

## Step 1: Enable Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to **APIs & Services** > **Library**
4. Search for "Gmail API" and click on it
5. Click **Enable**

## Step 2: Create OAuth 2.0 Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth 2.0 Client IDs**
3. If prompted, configure the OAuth consent screen:
   - Choose **External** user type
   - Fill in required fields (App name, User support email, Developer contact)
   - Add your email to test users
4. For Application type, select **Web application**
5. Add authorized redirect URIs:
   - `http://localhost:8080/callback`
6. Click **Create**
7. Download the JSON file and rename it to `credentials.json`
8. Place `credentials.json` in the `mcp-chatbot` directory

## Step 3: Install Dependencies

The required dependencies are already in `requirements.txt`:

```bash
pip install google-auth>=2.23.0
pip install google-auth-oauthlib>=1.1.0
pip install google-api-python-client>=2.99.0
```

## Step 4: Run Setup Script

```bash
python setup_gmail.py
```

This script will:
1. Verify your `credentials.json` file
2. Start a local server to handle OAuth callback
3. Open your browser for Gmail authorization
4. Complete the OAuth flow automatically

## Step 5: Test Gmail Integration

Once setup is complete, you can test Gmail functionality:

### Send Email
```
send email to someone@example.com about Hello, this is a test message
```

### Read Emails
```
read my gmail emails
show unread emails
```

### Search Emails
```
search emails for meeting
```

## Troubleshooting

### Common Issues

1. **"credentials.json not found"**
   - Make sure you downloaded the OAuth 2.0 credentials
   - Rename the file to exactly `credentials.json`
   - Place it in the `mcp-chatbot` directory

2. **"Invalid credentials.json format"**
   - Ensure you created OAuth 2.0 credentials for a **Web application**
   - Not Desktop application or Service account

3. **"Gmail access not authorized"**
   - Run the setup script again: `python setup_gmail.py`
   - Make sure you completed the OAuth flow in your browser

4. **"OAuth consent screen" issues**
   - Add your email to the test users list
   - Make sure the OAuth consent screen is configured

### Manual OAuth Flow

If the automatic setup doesn't work, you can complete OAuth manually:

1. Run the setup script to get the authorization URL
2. Copy the URL and open it in your browser
3. Complete the authorization
4. Copy the authorization code from the callback URL
5. Use the code to complete the OAuth flow

## Security Notes

- The `credentials.json` file contains sensitive information
- Never commit this file to version control
- The `tokens/` directory stores user authorization tokens
- These tokens are user-specific and should be kept secure

## API Limits

- Gmail API has daily quotas and rate limits
- For development, the default quotas should be sufficient
- For production use, consider requesting quota increases

## Support

If you encounter issues:
1. Check the console logs for error messages
2. Verify your Google Cloud Console setup
3. Ensure all dependencies are installed correctly
4. Check that the Gmail API is enabled in your project
