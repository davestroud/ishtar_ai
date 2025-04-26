#!/bin/bash
# Setup Tavily API key and run the app

# Activate virtual environment
source bin/activate

# Check if we need to install python-dotenv
if ! pip show python-dotenv > /dev/null 2>&1; then
    echo "Installing python-dotenv..."
    pip install python-dotenv
fi

# Check if Tavily API key is provided as an argument
if [ -n "$1" ]; then
    # Run the setup script with the provided key
    python set_tavily_key.py "$1"
else
    # Ask for the API key interactively
    python set_tavily_key.py
fi

# Check if the setup was successful
if [ $? -ne 0 ]; then
    echo "Setup failed. Exiting."
    exit 1
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

# Run the app
echo "🚀 Starting the app..."
streamlit run src/streamlit_app.py 