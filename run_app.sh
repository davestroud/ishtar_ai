#!/bin/bash
# Simple script to run the Ishtar AI application

# Ensure using Python 3.13.2 from pyenv
export PYENV_VERSION=3.13.2

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Check for Tavily API key
if [ -z "$TAVILY_API_KEY" ]; then
    echo "Warning: TAVILY_API_KEY environment variable is not set"
    echo "Web search capabilities will be disabled"
else
    echo "Tavily API key found (length: ${#TAVILY_API_KEY})"
fi

# Check for LangSmith API key
if [ -z "$LANGCHAIN_API_KEY" ]; then
    echo "No LangChain API key found in environment variables"
    echo "LangSmith tracing will be disabled"
else
    echo "LangSmith API key found (length: ${#LANGCHAIN_API_KEY})"
fi

# Check if Python 3.13 is available
python_version=$(python -V 2>&1)
if [[ $python_version != *"3.13"* ]]; then
    echo "Error: Python 3.13 is required but not found: $python_version"
    echo "Make sure to have Python 3.13 installed and activated with pyenv"
    exit 1
fi

# Run the Streamlit app with Poetry's Python
echo "Starting Ishtar AI Assistant with Poetry and Python $python_version..."
poetry run python -m streamlit run ishtar_app/app.py 