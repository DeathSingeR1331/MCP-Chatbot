// Synapse AI Chatbot Frontend
class SynapseChatbot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCounter = document.getElementById('charCounter');
        this.settingsModal = document.getElementById('settingsModal');
        this.toastContainer = document.getElementById('toastContainer');
        
        this.isTyping = false;
        this.messageHistory = [];
        this.userName = localStorage.getItem('userName') || '';
        this.theme = localStorage.getItem('theme') || 'light';
        this.llmProvider = localStorage.getItem('llmProvider') || 'groq';
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.applyTheme();
        this.loadSettings();
        this.setupAutoResize();
        this.setupQuickActions();
    }
    
    setupEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Input validation
        this.messageInput.addEventListener('input', () => {
            this.updateCharCounter();
            this.updateSendButton();
        });
        
        // Settings modal
        document.getElementById('settingsBtn').addEventListener('click', () => this.openSettings());
        document.getElementById('closeSettingsBtn').addEventListener('click', () => this.closeSettings());
        document.getElementById('cancelSettingsBtn').addEventListener('click', () => this.closeSettings());
        document.getElementById('saveSettingsBtn').addEventListener('click', () => this.saveSettings());
        
        // Clear chat
        document.getElementById('clearChatBtn').addEventListener('click', () => this.clearChat());
        
        // Voice input (placeholder)
        document.getElementById('voiceBtn').addEventListener('click', () => this.toggleVoiceInput());
        
        // Attach file (placeholder)
        document.getElementById('attachBtn').addEventListener('click', () => this.attachFile());
        
        // Close modal on outside click
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettings();
            }
        });
        
        // Theme change
        document.getElementById('theme').addEventListener('change', (e) => {
            this.theme = e.target.value;
            this.applyTheme();
        });
    }
    
    setupAutoResize() {
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
    }
    
    setupQuickActions() {
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;
                this.handleQuickAction(action);
            });
        });
    }
    
    handleQuickAction(action) {
        const actions = {
            gmail: "Help me with Gmail - send an email, read emails, or search my inbox",
            weather: "What's the weather like today?",
            news: "Show me the latest news",
            browse: "Help me browse the web and find information"
        };
        
        if (actions[action]) {
            this.messageInput.value = actions[action];
            this.updateCharCounter();
            this.updateSendButton();
            this.sendMessage();
        }
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateCharCounter();
        this.updateSendButton();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // Send message to backend
            const response = await this.sendToBackend(message);
            this.hideTypingIndicator();
            
            // Add bot response to chat
            this.addMessage(response, 'bot');
            
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            this.showToast('Error sending message', 'error');
            console.error('Error:', error);
        }
    }
    
    async sendToBackend(message) {
        // For now, simulate a response
        // In a real implementation, this would call your Python backend
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
        
        // Simulate different responses based on message content
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
            if (this.userName) {
                return `Your name is ${this.userName}! I remember that from our previous conversations. 😊`;
            } else {
                return "I don't have a name set for you yet. You can tell me your name by saying something like 'My name is John' or 'Call me Sarah'.";
            }
        } else if (lowerMessage.includes('my name is') || lowerMessage.includes('call me')) {
            // Extract name from message
            const nameMatch = message.match(/(?:my name is|call me|i am)\s+([a-zA-Z0-9\s\-\.']+)/i);
            if (nameMatch) {
                const name = nameMatch[1].trim();
                this.userName = name;
                localStorage.setItem('userName', name);
                return `✅ Got it! I'll remember your name as ${name}. Nice to meet you, ${name}! 😊`;
            }
        } else if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
            const greeting = this.userName ? `Hello ${this.userName}!` : 'Hello!';
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
            // Default response
            return `I understand you're asking about "${message}". I'm your Synapse AI assistant and I'm here to help! I can assist with Gmail, web browsing, news, weather, and general questions. What specific task would you like me to help you with?`;
        }
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = this.formatMessage(content);
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageContent.appendChild(messageText);
        messageContent.appendChild(messageTime);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store in history
        this.messageHistory.push({ content, sender, timestamp: new Date() });
    }
    
    formatMessage(content) {
        // Convert markdown-like formatting to HTML
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/^\- (.*$)/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        this.typingIndicator.classList.add('show');
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        this.typingIndicator.classList.remove('show');
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    updateCharCounter() {
        const count = this.messageInput.value.length;
        this.charCounter.textContent = `${count}/2000`;
        
        if (count > 1800) {
            this.charCounter.style.color = 'var(--error-color)';
        } else if (count > 1500) {
            this.charCounter.style.color = 'var(--warning-color)';
        } else {
            this.charCounter.style.color = 'var(--text-muted)';
        }
    }
    
    updateSendButton() {
        this.sendBtn.disabled = !this.messageInput.value.trim() || this.isTyping;
    }
    
    openSettings() {
        this.settingsModal.classList.add('show');
        document.getElementById('userName').value = this.userName;
        document.getElementById('theme').value = this.theme;
        document.getElementById('llmProvider').value = this.llmProvider;
    }
    
    closeSettings() {
        this.settingsModal.classList.remove('show');
    }
    
    saveSettings() {
        this.userName = document.getElementById('userName').value.trim();
        this.theme = document.getElementById('theme').value;
        this.llmProvider = document.getElementById('llmProvider').value;
        
        localStorage.setItem('userName', this.userName);
        localStorage.setItem('theme', this.theme);
        localStorage.setItem('llmProvider', this.llmProvider);
        
        this.applyTheme();
        this.closeSettings();
        this.showToast('Settings saved successfully!', 'success');
    }
    
    loadSettings() {
        if (this.userName) {
            document.getElementById('userName').value = this.userName;
        }
    }
    
    applyTheme() {
        if (this.theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            document.documentElement.setAttribute('data-theme', this.theme);
        }
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            // Keep only the welcome message
            const welcomeMessage = this.chatMessages.querySelector('.bot-message');
            this.chatMessages.innerHTML = '';
            if (welcomeMessage) {
                this.chatMessages.appendChild(welcomeMessage);
            }
            this.messageHistory = [];
            this.showToast('Chat cleared successfully!', 'success');
        }
    }
    
    toggleVoiceInput() {
        this.showToast('Voice input coming soon!', 'warning');
    }
    
    attachFile() {
        this.showToast('File attachment coming soon!', 'warning');
    }
    
    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = type === 'success' ? 'check-circle' : 
                    type === 'error' ? 'exclamation-circle' : 
                    type === 'warning' ? 'exclamation-triangle' : 'info-circle';
        
        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;
        
        this.toastContainer.appendChild(toast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new SynapseChatbot();
});

// Handle theme changes
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    const chatbot = window.synapseChatbot;
    if (chatbot && chatbot.theme === 'auto') {
        chatbot.applyTheme();
    }
});

// Service Worker registration for PWA capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

