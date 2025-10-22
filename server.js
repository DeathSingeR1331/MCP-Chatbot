const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// API endpoint to communicate with Python backend
app.post('/api/chat', async (req, res) => {
    try {
        const { message, userName, llmProvider } = req.body;
        
        // For now, return a mock response
        // In a real implementation, this would communicate with your Python backend
        const mockResponse = await generateMockResponse(message, userName);
        
        res.json({
            success: true,
            response: mockResponse,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('Error processing chat request:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            message: 'Failed to process your request'
        });
    }
});

// API endpoint to get user settings
app.get('/api/settings', (req, res) => {
    res.json({
        userName: req.query.userName || '',
        theme: req.query.theme || 'light',
        llmProvider: req.query.llmProvider || 'groq'
    });
});

// API endpoint to save user settings
app.post('/api/settings', (req, res) => {
    const { userName, theme, llmProvider } = req.body;
    
    // In a real implementation, you would save this to a database
    console.log('Saving settings:', { userName, theme, llmProvider });
    
    res.json({
        success: true,
        message: 'Settings saved successfully'
    });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0'
    });
});

// Mock response generator (replace with actual Python backend integration)
async function generateMockResponse(message, userName) {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('gmail') || lowerMessage.includes('email')) {
        return "I can help you with Gmail! I can send emails, read your inbox, search for specific emails, and manage your email. What would you like to do with Gmail?";
    } else if (lowerMessage.includes('weather')) {
        return "I can check the weather for you! Please provide a city name or location, and I'll get the current weather conditions and forecast for you.";
    } else if (lowerMessage.includes('news')) {
        return "I can fetch the latest news for you! I can get news on various topics like technology, sports, politics, or general headlines. What type of news are you interested in?";
    } else if (lowerMessage.includes('browse') || lowerMessage.includes('web')) {
        return "I can help you browse the web! I can search for information, visit websites, take screenshots, and help you find what you're looking for online. What would you like to search for?";
    } else if (lowerMessage.includes('name') && lowerMessage.includes('my')) {
        if (userName) {
            return `Your name is ${userName}! I remember that from our previous conversations. 😊`;
        } else {
            return "I don't have a name set for you yet. You can tell me your name by saying something like 'My name is John' or 'Call me Sarah'.";
        }
    } else if (lowerMessage.includes('my name is') || lowerMessage.includes('call me')) {
        const nameMatch = message.match(/(?:my name is|call me|i am)\s+([a-zA-Z0-9\s\-\.']+)/i);
        if (nameMatch) {
            const name = nameMatch[1].trim();
            return `✅ Got it! I'll remember your name as ${name}. Nice to meet you, ${name}! 😊`;
        }
    } else if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        const greeting = userName ? `Hello ${userName}!` : 'Hello!';
        return `${greeting} How can I help you today?`;
    } else if (lowerMessage.includes('help')) {
        return `I'm your Synapse AI assistant! I can help you with:

📧 **Gmail Operations**: Send emails, read your inbox, search emails
🌐 **Web Browsing**: Search the web, visit websites, take screenshots  
📰 **News & Weather**: Get latest news and weather information
💬 **General Chat**: Answer questions and have conversations
🔧 **Tools & Integrations**: Various productivity tools

Just ask me what you'd like to do!`;
    } else {
        return `I understand you're asking about "${message}". I'm your Synapse AI assistant and I'm here to help! I can assist with Gmail, web browsing, news, weather, and general questions. What specific task would you like me to help you with?`;
    }
}

// Function to integrate with Python backend (for future implementation)
function callPythonBackend(message, userName, llmProvider) {
    return new Promise((resolve, reject) => {
        const python = spawn('python', ['main.py', '--message', message, '--user-name', userName, '--provider', llmProvider]);
        
        let output = '';
        let error = '';
        
        python.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        python.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                resolve(output.trim());
            } else {
                reject(new Error(`Python script exited with code ${code}: ${error}`));
            }
        });
    });
}

// Start the server
app.listen(PORT, () => {
    console.log(`🚀 Synapse AI Chatbot Frontend running on http://localhost:${PORT}`);
    console.log(`📱 Open your browser and navigate to the URL above to start chatting!`);
    console.log(`🔧 API endpoints available at http://localhost:${PORT}/api/`);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n👋 Shutting down Synapse AI Chatbot Frontend...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n👋 Shutting down Synapse AI Chatbot Frontend...');
    process.exit(0);
});

