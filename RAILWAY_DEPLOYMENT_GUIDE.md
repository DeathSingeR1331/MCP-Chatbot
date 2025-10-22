# 🚀 MCP Chatbot Railway Deployment Guide

This guide will help you deploy the MCP Chatbot on Railway and integrate it with your existing Synapse backend.

## 📋 Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **API Keys**: 
   - Groq API Key (from [console.groq.com](https://console.groq.com))
   - Gemini API Key (from [makersuite.google.com](https://makersuite.google.com))
   - Notion Token (optional, from [notion.so](https://notion.so))

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vercel        │    │   Railway       │    │   Railway       │
│   Frontend      │───▶│   Backend       │    │   MCP Chatbot   │
│   (React)       │    │   (FastAPI)     │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OAuth         │    │   PostgreSQL    │    │   MCP Servers   │
│   (Google)      │    │   Database      │    │   (Tools)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 How It Works

### **Chat Modes in Synapse Frontend:**

1. **🧠 Personalization Mode**: 
   - Uses Railway Backend
   - Stores conversation history
   - Learns from user interactions

2. **🔧 Tools Mode**: 
   - Uses Railway MCP Chatbot
   - Executes tools (Gmail, Web browsing, etc.)
   - No conversation memory

3. **🚀 Both Mode**: 
   - Uses Railway Backend with tool capabilities
   - Combines personalization + tools

### **MCP Chatbot Features:**
- **Gmail Integration**: Send/receive emails
- **Web Browsing**: Search and browse websites
- **Notion Integration**: Manage Notion pages/databases
- **Airbnb Search**: Find accommodations
- **Database Operations**: SQLite queries

## 🚀 Step-by-Step Deployment

### **Step 1: Prepare the Code**

1. **Navigate to MCP Chatbot directory:**
   ```bash
   cd "C:\Users\Nishank Goswami\Downloads\mcp-chatbot"
   ```

2. **Commit the fixes:**
   ```bash
   git add -A
   git commit -m "Fix imports and add Railway deployment files"
   git push origin main
   ```

### **Step 2: Deploy to Railway**

1. **Go to Railway Dashboard:**
   - Visit [railway.app](https://railway.app)
   - Click "New Project"

2. **Connect GitHub Repository:**
   - Select "Deploy from GitHub repo"
   - Choose your `mcp-chatbot` repository
   - Click "Deploy Now"

3. **Configure Environment Variables:**
   In Railway dashboard, go to Variables tab and add:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   NOTION_TOKEN=your_notion_token_here (optional)
   PORT=8001
   HOST=0.0.0.0
   LOG_LEVEL=INFO
   DISPLAY=:1
   ```

4. **Set Custom Domain (Optional):**
   - Go to Settings → Domains
   - Add custom domain: `synapse-mcp.railway.app`

### **Step 3: Update Frontend Configuration**

The frontend is already configured to use the MCP API at:
```
VITE_MCP_API_URL=https://synapse-mcp.railway.app
```

### **Step 4: Test the Integration**

1. **Check MCP Chatbot Health:**
   ```bash
   curl https://synapse-mcp.railway.app/health
   ```

2. **Test Tools Endpoint:**
   ```bash
   curl https://synapse-mcp.railway.app/tools
   ```

3. **Test Query Endpoint:**
   ```bash
   curl -X POST https://synapse-mcp.railway.app/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, can you help me?", "user_id": "test-user"}'
   ```

## 🔄 Integration Flow

### **When User Selects "🔧 Tools" Mode:**

1. **Frontend** sends request to MCP Chatbot:
   ```javascript
   POST https://synapse-mcp.railway.app/query
   {
     "query": "Send an email to john@example.com",
     "user_id": "user-uuid"
   }
   ```

2. **MCP Chatbot** processes the query:
   - Uses Groq LLM to understand intent
   - Identifies required tools (Gmail, Web, etc.)
   - Executes tools via MCP servers
   - Returns formatted response

3. **Frontend** displays the response in chat

### **When User Selects "🧠 Personalization" Mode:**

1. **Frontend** sends request to Backend:
   ```javascript
   POST https://synapse-backend-production-7887.up.railway.app/api/v1/conversations/{id}/messages
   ```

2. **Backend** processes with personalization:
   - Stores conversation history
   - Learns from interactions
   - Returns personalized response

## 🛠️ Available MCP Tools

### **Gmail Tools:**
- Send emails
- Read inbox
- Search emails
- Manage labels

### **Web Browsing Tools:**
- Search Google
- Visit websites
- Take screenshots
- Extract content

### **Notion Tools:**
- Create pages
- Search pages
- Query databases
- Add comments

### **Airbnb Tools:**
- Search accommodations
- Get property details
- Check availability

## 🔍 Troubleshooting

### **Common Issues:**

1. **MCP Servers Not Starting:**
   - Check Railway logs for Node.js errors
   - Verify npm packages are installed
   - Check environment variables

2. **CORS Errors:**
   - Verify frontend URL is in CORS allowlist
   - Check if MCP API is accessible

3. **API Key Issues:**
   - Verify API keys are set in Railway
   - Check key permissions and quotas

4. **Tool Execution Failures:**
   - Check MCP server logs
   - Verify tool dependencies
   - Test individual tools

### **Debug Commands:**

```bash
# Check MCP Chatbot status
curl https://synapse-mcp.railway.app/status

# List available tools
curl https://synapse-mcp.railway.app/tools

# Test specific tool
curl -X POST https://synapse-mcp.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What tools do you have?", "user_id": "test"}'
```

## 📊 Monitoring

### **Railway Dashboard:**
- View deployment logs
- Monitor resource usage
- Check environment variables
- View custom domains

### **Health Checks:**
- `/health` - Basic health check
- `/status` - Detailed server status
- `/tools` - Available tools list

## 🔐 Security Considerations

1. **API Keys**: Store securely in Railway environment variables
2. **CORS**: Only allow trusted origins
3. **Rate Limiting**: Consider implementing rate limits
4. **Authentication**: Add user authentication if needed

## 📈 Scaling

### **Railway Scaling:**
- Automatic scaling based on traffic
- Resource limits on free tier
- Upgrade to paid plan for more resources

### **Performance Optimization:**
- MCP servers initialize once on startup
- Connection pooling for databases
- Caching for frequently used tools

## 🎯 Next Steps

1. **Deploy to Railway** following the steps above
2. **Test all chat modes** in the frontend
3. **Monitor performance** and logs
4. **Add more MCP tools** as needed
5. **Implement user authentication** for production use

## 📞 Support

If you encounter issues:
1. Check Railway deployment logs
2. Verify environment variables
3. Test individual components
4. Check this guide for troubleshooting steps

---

**🎉 Congratulations!** Your MCP Chatbot should now be running on Railway and integrated with your Synapse frontend!
