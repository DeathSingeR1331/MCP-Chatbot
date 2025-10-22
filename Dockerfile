# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    wget \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install Playwright and browsers (as root, before switching users)
RUN npm install -g playwright && playwright install chromium

# Create a non-root user and set permissions
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port (Railway will override this)
EXPOSE 8001

# Simple health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health', timeout=5)" || exit 1

# Start the MCP API server
CMD ["python", "mcp_api_server.py"]
