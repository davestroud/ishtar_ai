#!/bin/bash
# Script to update Tavily API key in .env file

if [ -z "$1" ]; then
    echo "Error: No API key provided"
    echo "Usage: ./update_tavily_key.sh YOUR_TAVILY_API_KEY"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    touch .env
    echo "Created new .env file"
fi

# Check if TAVILY_API_KEY already exists in the file
if grep -q "^TAVILY_API_KEY=" .env; then
    # Replace existing TAVILY_API_KEY 
    sed -i.bak "s/^TAVILY_API_KEY=.*/TAVILY_API_KEY=$1/" .env
    echo "Updated TAVILY_API_KEY in .env file"
else
    # Add TAVILY_API_KEY to file
    echo "TAVILY_API_KEY=$1" >> .env
    echo "Added TAVILY_API_KEY to .env file"
fi

echo "Tavily API key has been set. You can now run the application with web search enabled." 