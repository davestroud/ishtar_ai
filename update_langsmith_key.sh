#!/bin/bash
# Script to update LangSmith API key in .env file

if [ -z "$1" ]; then
    echo "Error: No API key provided"
    echo "Usage: ./update_langsmith_key.sh YOUR_LANGSMITH_API_KEY"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    touch .env
    echo "Created new .env file"
fi

# Check if LANGCHAIN_API_KEY already exists in the file
if grep -q "^LANGCHAIN_API_KEY=" .env; then
    # Replace existing LANGCHAIN_API_KEY
    sed -i.bak "s/^LANGCHAIN_API_KEY=.*/LANGCHAIN_API_KEY=$1/" .env
    echo "Updated LANGCHAIN_API_KEY in .env file"
else
    # Add LANGCHAIN_API_KEY to file
    echo "LANGCHAIN_API_KEY=$1" >> .env
    echo "Added LANGCHAIN_API_KEY to .env file"
fi

# Make sure the LANGCHAIN_PROJECT is set
if ! grep -q "^LANGCHAIN_PROJECT=" .env; then
    echo "LANGCHAIN_PROJECT=ishtar-ai" >> .env
    echo "Added default LANGCHAIN_PROJECT to .env file"
fi

# Enable LangSmith tracing
if ! grep -q "^LANGSMITH_TRACING=" .env; then
    echo "LANGSMITH_TRACING=true" >> .env
    echo "Enabled LangSmith tracing in .env file"
fi

echo "LangSmith key has been set. You can now run the application with LangSmith tracing enabled."
