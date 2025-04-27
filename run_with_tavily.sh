#!/bin/bash
# Script to run the app with Tavily API integration

# Activate virtual environment
source llm_env/bin/activate

# Load the environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a

    # Verify that the API key was loaded
    if [ -n "$TAVILY_API_KEY" ]; then
        echo "✅ Tavily API key found in .env file"
    else
        echo "⚠️ Tavily API key not found in .env file"
    fi
else
    echo "⚠️ No .env file found"
fi

# Check if Tavily API key is provided as environment variable or argument
TAVILY_KEY=${TAVILY_API_KEY:-$1}

if [ -z "$TAVILY_KEY" ]; then
    echo "No Tavily API key found."
    echo
    echo "Would you like to set up your Tavily API key now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        ./setup_tavily.sh
        # Re-source the environment file after setup
        if [ -f .env ]; then
            set -a
            source .env
            set +a
        fi
    else
        echo "You can run the app without web search, but some features will be limited."
        # Ask if user wants to continue without API key
        echo "Do you want to continue without web search capabilities? (y/n)"
        read -r continue_response
        if [[ ! $continue_response =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    # Export the Tavily API key
    export TAVILY_API_KEY="$TAVILY_KEY"
    echo "✅ Tavily API key is set"
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
echo "TAVILY_API_KEY=${TAVILY_API_KEY:+is set}"
streamlit run streamlit_app.py 