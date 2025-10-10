# Synapse Frontend Integration

This document explains how to use the MCP API Server with the Synapse frontend for tools functionality.

## 🚀 Quick Start

### 1. Setup Environment

Make sure you have your API keys in a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
NOTION_TOKEN=your_notion_token_here  # Optional
```

### 2. Install Dependencies

```bash
# Activate your virtual environment
# Then install dependencies
pip install -r requirements.txt
```

### 3. Start MCP API Server

```bash
# Option 1: Use the startup script
python start_mcp_server.py

# Option 2: Run directly
python mcp_api_server.py
```

The server will start on `http://localhost:8001`

### 4. Start Synapse Frontend

In another terminal, start your Synapse frontend:
```bash
cd C:\Users\Saatvik\Synapse-Frontend
npm run dev
```

### 5. Use Tools Mode

1. Open the Synapse frontend in your browser
2. Select "🔧 Tools" mode from the dropdown
3. Type your query - it will be processed by the MCP API server
4. Tools will be executed using Groq LLM and MCP servers

## 🔧 How It Works

### Architecture

```
Frontend (Tools Mode) → MCP API Server (localhost:8001) → MCP Servers → Groq LLM
```

1. **Frontend**: When "Tools" mode is selected, queries are sent to `http://localhost:8001/query`
2. **MCP API Server**: Processes the query using MCP servers and Groq LLM
3. **MCP Servers**: Execute tools (playwright, puppeteer, notion, sqlite, airbnb)
4. **Response**: Results are sent back to frontend and displayed

### Available Endpoints

- `GET /health` - Health check
- `GET /tools` - List available tools
- `GET /status` - Detailed status of servers and tools
- `POST /query` - Process a query with tools

### MCP Servers

The following MCP servers are initialized:
- **playwright**: Browser automation
- **puppeteer**: Browser automation (alternative)
- **notionApi**: Notion integration
- **sqlite**: Database operations
- **airbnb**: Airbnb search

## 🎯 Key Features

- **Groq Primary**: Uses Groq as the primary LLM for tool decisions
- **Gemini Fallback**: Falls back to Gemini if Groq fails
- **No Ollama**: Ollama has been removed as requested
- **Local Execution**: MCP servers run locally in your venv
- **Frontend Integration**: Seamless integration with Synapse frontend
- **Personalization Preserved**: Personalization mode continues to use the backend

## 🔍 Troubleshooting

### Server Won't Start
- Check if port 8001 is available
- Verify your `.env` file has correct API keys
- Make sure all dependencies are installed

### Tools Not Working
- Check the `/status` endpoint to see which servers are initialized
- Look at the server logs for error messages
- Verify MCP server dependencies (Node.js, npm packages)

### Frontend Connection Issues
- Ensure MCP API server is running on `http://localhost:8001`
- Check browser console for CORS errors
- Verify the frontend is calling the correct endpoint

## 📝 Notes

- The MCP API server runs independently of the Synapse backend
- Personalization mode continues to use the original Synapse backend
- Tools mode bypasses personalization and uses direct MCP execution
- All MCP servers are initialized on startup for better performance
