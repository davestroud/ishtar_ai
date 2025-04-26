#!/bin/bash
# Script to run the app with Tavily API integration

# Activate virtual environment
source bin/activate

# Check if Tavily API key is provided as environment variable or argument
TAVILY_KEY=${TAVILY_API_KEY:-$1}

if [ -z "$TAVILY_KEY" ]; then
    echo "No Tavily API key found."
    echo "Usage: $0 <your_tavily_api_key>"
    echo "  or set the TAVILY_API_KEY environment variable before running"
    echo "You can get an API key from https://tavily.com"
    
    # Ask if user wants to continue without API key
    read -p "Do you want to continue without web search capabilities? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    # Export the Tavily API key
    export TAVILY_API_KEY="$TAVILY_KEY"
    echo "✅ Tavily API key set. Web search will be enabled."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Ensure Ollama container is running
echo "🐳 Ensuring Ollama container is running..."
if docker ps | grep -q ollama; then
    echo "✅ Ollama container is already running"
else
    echo "Starting Ollama container..."
    docker restart ollama 2>/dev/null || {
        echo "Creating new Ollama container..."
        docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
        echo "⏳ Waiting for Ollama to initialize..."
        sleep 5
    }
fi

# Check if Ollama is responding
echo "🔍 Testing Ollama API connection..."
if curl -s "http://localhost:11434/api/tags" >/dev/null; then
    echo "✅ Ollama API is responding"
else
    echo "⚠️ Warning: Ollama API is not responding. The container might still be initializing."
    echo "⏳ Waiting a bit longer..."
    sleep 5
fi

echo "🚀 Starting Streamlit app..."
# Run the Streamlit app
streamlit run src/streamlit_app.py 