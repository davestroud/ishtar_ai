#!/bin/bash
# Setup script for Ishtar AI

print_help() {
    echo "Ishtar AI Setup Script"
    echo ""
    echo "Usage: ./setup.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help                 Show this help message"
    echo "  --tavily-key KEY       Set Tavily API key for web search"
    echo "  --langsmith-key KEY    Set LangSmith API key for tracing"
    echo "  --openai-key KEY       Set OpenAI API key"
    echo "  --langchain-project NAME  Set LangChain project name"
    echo "  --install              Install dependencies"
    echo "  --run                  Run the application after setup"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh --install --tavily-key your_tavily_key"
    echo "  ./setup.sh --langchain-project ishtar_langchain"
    echo "  ./setup.sh --run"
}

update_env_var() {
    # Update or add an environment variable in .env file
    local var_name=$1
    local var_value=$2
    
    if [ ! -f .env ]; then
        # Create .env file if it doesn't exist
        echo "# Created by setup.sh" > .env
    fi
    
    # Check if variable exists
    if grep -q "^${var_name}=" .env; then
        # Replace existing variable
        sed -i '' "s|^${var_name}=.*|${var_name}=${var_value}|" .env
    else
        # Add new variable
        echo "${var_name}=${var_value}" >> .env
    fi
    
    echo "✅ Updated ${var_name} in .env file"
}

setup_env() {
    # Create basic .env file with default configuration
    if [ ! -f .env ]; then
        echo "# API Keys" > .env
        echo "# Add your API keys here:" >> .env
        echo "# TAVILY_API_KEY=your_tavily_api_key" >> .env
        echo "# LANGCHAIN_API_KEY=your_langsmith_api_key" >> .env
        echo "# OPENAI_API_KEY=your_openai_api_key" >> .env
        echo "" >> .env
        echo "# LangSmith Configuration" >> .env
        echo "LANGCHAIN_PROJECT=ishtar_langchain" >> .env
        echo "LANGCHAIN_ENDPOINT=https://api.smith.langchain.com" >> .env
        echo "LANGSMITH_TRACING=true" >> .env
        echo "" >> .env
        echo "# Ollama Configuration" >> .env
        echo "OLLAMA_HOST=localhost" >> .env
        echo "OLLAMA_PORT=11434" >> .env
        echo "DEFAULT_MODEL=llama3" >> .env
        
        echo "✅ Created default .env file"
    else
        echo "ℹ️ Using existing .env file"
    fi
}

install_dependencies() {
    echo "🔄 Installing Python dependencies..."
    
    # Check if virtual environment exists, create if not
    if [ ! -d "llm_env" ]; then
        echo "Creating virtual environment..."
        python3 -m venv llm_env
    fi
    
    # Activate virtual environment and install dependencies
    source llm_env/bin/activate
    pip install -r requirements.txt
    
    echo "✅ Dependencies installed"
}

check_docker() {
    echo "🐳 Checking Docker..."
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker is not running. Please start Docker first."
        return 1
    fi
    
    echo "✅ Docker is available"
    return 0
}

ensure_ollama() {
    echo "🤖 Checking Ollama..."
    if ! docker ps | grep -q ollama; then
        echo "Ollama container not running. Attempting to start..."
        docker restart ollama &> /dev/null || {
            echo "Creating new Ollama container..."
            docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
            echo "⏳ Waiting for Ollama to initialize..."
            sleep 5
        }
    else
        echo "✅ Ollama container is already running"
    fi
    
    # Check if Ollama is responding
    echo "🔍 Testing Ollama API connection..."
    if curl -s "http://localhost:11434/api/tags" &> /dev/null; then
        echo "✅ Ollama API is responding"
        return 0
    else
        echo "⚠️ Warning: Ollama API is not responding. The container might still be initializing."
        echo "⏳ Waiting a bit longer..."
        sleep 5
        return 1
    fi
}

run_app() {
    echo "🚀 Starting Ishtar AI..."
    source llm_env/bin/activate
    streamlit run streamlit_app.py
}

# Main script execution
if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

# Process command line arguments
DO_INSTALL=false
DO_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            print_help
            exit 0
            ;;
        --tavily-key)
            TAVILY_KEY="$2"
            shift 2
            ;;
        --langsmith-key)
            LANGSMITH_KEY="$2"
            shift 2
            ;;
        --openai-key)
            OPENAI_KEY="$2"
            shift 2
            ;;
        --langchain-project)
            LANGCHAIN_PROJECT="$2"
            shift 2
            ;;
        --install)
            DO_INSTALL=true
            shift
            ;;
        --run)
            DO_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Setup environment
setup_env

# Update API keys if provided
if [ ! -z "$TAVILY_KEY" ]; then
    update_env_var "TAVILY_API_KEY" "$TAVILY_KEY"
fi

if [ ! -z "$LANGSMITH_KEY" ]; then
    update_env_var "LANGCHAIN_API_KEY" "$LANGSMITH_KEY"
fi

if [ ! -z "$OPENAI_KEY" ]; then
    update_env_var "OPENAI_API_KEY" "$OPENAI_KEY"
fi

if [ ! -z "$LANGCHAIN_PROJECT" ]; then
    update_env_var "LANGCHAIN_PROJECT" "$LANGCHAIN_PROJECT"
fi

# Install dependencies if requested
if $DO_INSTALL; then
    install_dependencies
fi

# Run the application if requested
if $DO_RUN; then
    check_docker && ensure_ollama && run_app
fi

echo "✨ Setup complete!"
echo "To run the application: ./setup.sh --run" 