#!/bin/bash
# Simple script to update the Tavily API key in .env file

# Check if API key is provided as argument
if [ -z "$1" ]; then
    echo "Please provide your Tavily API key as an argument."
    echo "Usage: ./update_tavily_key.sh YOUR_TAVILY_API_KEY"
    exit 1
fi

API_KEY=$1

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating new .env file..."
    echo "# API Keys" > .env
    echo "TAVILY_API_KEY=$API_KEY" >> .env
    echo "" >> .env
    echo "# Ollama Configuration" >> .env
    echo "OLLAMA_HOST=localhost" >> .env
    echo "OLLAMA_PORT=11434" >> .env
    echo "DEFAULT_MODEL=llama3" >> .env
else
    # Update existing .env file
    echo "Updating Tavily API key in .env file..."
    # Check if TAVILY_API_KEY exists in the file
    if grep -q "TAVILY_API_KEY=" .env; then
        # Replace the existing key
        sed -i '' "s/TAVILY_API_KEY=.*/TAVILY_API_KEY=$API_KEY/" .env
    else
        # Add the key if it doesn't exist
        echo "TAVILY_API_KEY=$API_KEY" >> .env
    fi
fi

echo "✅ Tavily API key has been updated in .env file."
echo "You can now run the app with: ./run_with_tavily.sh" 