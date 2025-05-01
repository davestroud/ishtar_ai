#!/bin/bash
# Simple script to run the Ishtar AI application

# Ensure using Python 3.13.2 from pyenv
export PYENV_VERSION=3.13.2

# Create offload folder for model disk offloading if it doesn't exist
if [ ! -d "offload_folder" ]; then
    echo "Creating offload folder for model disk offloading"
    mkdir -p offload_folder
fi

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
    echo "----------------------------------------"
    echo "Environment variables loaded:"
    
    # Check for Hugging Face API key
    if [ -n "$HUGGINGFACE_API_KEY" ]; then
        echo "✅ Hugging Face API key found (length: ${#HUGGINGFACE_API_KEY})"
    else
        echo "⚠️ No Hugging Face API key found"
    fi
    
    # Check for Tavily API key
    if [ -n "$TAVILY_API_KEY" ]; then
        echo "✅ Tavily API key found (length: ${#TAVILY_API_KEY})"
    else
        echo "⚠️ No Tavily API key found"
    fi
    
    # Check for LangSmith/LangChain API key
    if [ -n "$LANGCHAIN_API_KEY" ] || [ -n "$LANGSMITH_API_KEY" ]; then
        echo "✅ LangSmith/LangChain API key found"
        # Explicitly enable tracing when we have a key
        export LANGSMITH_TRACING=true
        if [ -n "$LANGCHAIN_PROJECT" ]; then
            echo "   - LangChain project: $LANGCHAIN_PROJECT"
        else
            # Set a default project name if not already set
            export LANGCHAIN_PROJECT="ishtar_ai"
            echo "   - Set default LangChain project: $LANGCHAIN_PROJECT"
        fi
        
        # Make sure we have both env vars for older integrations
        if [ -n "$LANGCHAIN_API_KEY" ] && [ -z "$LANGSMITH_API_KEY" ]; then
            export LANGSMITH_API_KEY=$LANGCHAIN_API_KEY
            echo "   - Copied LANGCHAIN_API_KEY to LANGSMITH_API_KEY for compatibility"
        fi
        if [ -n "$LANGSMITH_API_KEY" ] && [ -z "$LANGCHAIN_API_KEY" ]; then
            export LANGCHAIN_API_KEY=$LANGSMITH_API_KEY
            echo "   - Copied LANGSMITH_API_KEY to LANGCHAIN_API_KEY for compatibility"
        fi
    else
        echo "⚠️ No LangSmith/LangChain API key found"
        # Explicitly disable tracing when we don't have a key
        export LANGSMITH_TRACING=false
    fi
    
    # Check for Pinecone
    if [ -n "$PINECONE_API_KEY" ] && [ -n "$PINECONE_HOST" ]; then
        echo "✅ Pinecone API key and host found"
        if [ -n "$PINECONE_INDEX" ]; then
            echo "   - Pinecone index: $PINECONE_INDEX"
        fi
    else
        echo "⚠️ Pinecone configuration incomplete or missing"
    fi
    
    # Check for OpenWeather API key
    if [ -n "$OPENWEATHER_API_KEY" ]; then
        echo "✅ OpenWeather API key found"
    else
        echo "⚠️ No OpenWeather API key found"
    fi
    
    # Check for OpenAI API key
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "✅ OpenAI API key found (length: ${#OPENAI_API_KEY})"
    else
        echo "⚠️ No OpenAI API key found"
    fi
    
    # Check for NewsAPI key
    if [ -n "$NEWSAPI_KEY" ]; then
        echo "✅ NewsAPI key found"
    else
        echo "⚠️ No NewsAPI key found"
    fi
    
    echo "----------------------------------------"
else
    echo "No .env file found. Using environment variables already set."
fi

# Check if Python 3.13 is available
python_version=$(python -V 2>&1)
if [[ $python_version != *"3.13"* ]]; then
    echo "Error: Python 3.13 is required but not found: $python_version"
    echo "Make sure to have Python 3.13 installed and activated with pyenv"
    exit 1
fi

# Configure memory settings for model disk offloading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run the Streamlit app
echo "Starting Ishtar AI with Streamlit..."
echo "App will be available at http://localhost:8501"
echo "Disk offloading enabled for large models in './offload_folder/'"

# Run with streamlit
streamlit run ishtar_app/app.py 