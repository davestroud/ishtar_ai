#!/bin/bash
# Setup script for Ishtar AI using Poetry

print_help() {
    echo "Ishtar AI Poetry Setup Script"
    echo ""
    echo "Usage: ./poetry_setup.sh [options]"
    echo ""
    echo "Options:"
    echo "  --help                 Show this help message"
    echo "  --tavily-key KEY       Set Tavily API key for web search"
    echo "  --langsmith-key KEY    Set LangSmith API key for tracing"
    echo "  --openai-key KEY       Set OpenAI API key"
    echo "  --langchain-project NAME  Set LangChain project name"
    echo "  --install              Install dependencies with Poetry"
    echo "  --docker               Build and run with Docker Compose"
    echo "  --run                  Run the application after setup"
    echo ""
    echo "Examples:"
    echo "  ./poetry_setup.sh --install --tavily-key your_tavily_key"
    echo "  ./poetry_setup.sh --docker"
    echo "  ./poetry_setup.sh --run"
    echo ""
    echo "API Key Management:"
    echo "  You can also use the scripts in the 'scripts/' directory to manage API keys individually:"
    echo "  ./scripts/update_tavily_key.sh YOUR_TAVILY_API_KEY"
    echo "  ./scripts/update_langsmith_key.sh YOUR_LANGSMITH_API_KEY"
    echo "  ./scripts/update_pinecone.sh --api-key YOUR_PINECONE_API_KEY --host YOUR_PINECONE_HOST"
    echo "  ./scripts/update_weather_key.sh YOUR_OPENWEATHER_API_KEY"
}

update_env_var() {
    # Update or add an environment variable in .env file
    local var_name=$1
    local var_value=$2
    
    if [ ! -f .env ]; then
        # Create .env file if it doesn't exist
        echo "# Created by poetry_setup.sh" > .env
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

install_poetry() {
    echo "🔄 Checking for Poetry..."
    
    if ! command -v poetry &> /dev/null; then
        echo "Poetry not found. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Add Poetry to PATH
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            source ~/.zshrc
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            source ~/.bashrc
        fi
    else
        echo "✅ Poetry is already installed"
    fi
    
    # Find Python executable
    PYTHON_PATH=$(which python3)
    if [ -z "$PYTHON_PATH" ]; then
        echo "⚠️ Could not find python3 executable. Using system default."
    else
        echo "🔄 Configuring Poetry to use Python at: $PYTHON_PATH"
        poetry config virtualenvs.prefer-active-python true
    fi
}

initialize_poetry_project() {
    echo "🔄 Initializing Poetry project..."
    
    if [ ! -f "pyproject.toml" ]; then
        echo "⚠️ pyproject.toml not found. It should have been created during repository setup."
        echo "Creating a basic pyproject.toml file..."
        cat > pyproject.toml << EOL
[tool.poetry]
name = "ishtar-ai"
version = "0.1.0"
description = "Ishtar AI Initiative for media and journalism entities"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.31.0"
python-dotenv = "^1.0.0"
streamlit = "^1.29.0"
numpy = "^1.26.0"
langchain = "^0.1.12"
langchain-core = ">=0.1.31,<0.2.0"
langsmith = "^0.0.87"
openai = "^1.76.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOL
    fi
    
    # Generate lock file if it doesn't exist
    if [ ! -f "poetry.lock" ]; then
        echo "🔄 Generating poetry.lock file..."
        if ! poetry lock; then
            echo "⚠️ Warning: Could not generate poetry.lock file automatically."
            echo "This may be due to Python configuration issues."
            echo ""
            echo "You can try the Docker option instead: ./poetry_setup.sh --docker"
            echo "Or continue using the traditional setup: ./setup.sh --install"
        else
            echo "✅ Poetry lock file generated successfully"
        fi
    else
        echo "✅ Poetry lock file already exists"
    fi
}

install_dependencies() {
    echo "🔄 Installing dependencies with Poetry..."
    
    if [ ! -f "pyproject.toml" ]; then
        echo "❌ pyproject.toml not found. Please run the script from the project root."
        exit 1
    fi
    
    # Try to install
    if ! poetry install; then
        echo "⚠️ Warning: Poetry dependency installation encountered issues."
        echo "This may be due to Python configuration issues."
        echo ""
        echo "You can try the Docker option instead: ./poetry_setup.sh --docker"
        echo "Or continue using the traditional setup: ./setup.sh --install"
    else
        echo "✅ Dependencies installed with Poetry"
    fi
}

check_docker() {
    echo "🐳 Checking Docker..."
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
        return 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose not found. Please install Docker Compose first."
        echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker is not running. Please start Docker first."
        return 1
    fi
    
    echo "✅ Docker and Docker Compose are available"
    return 0
}

run_with_docker() {
    echo "🐳 Building and running with Docker Compose..."
    docker-compose up --build -d
    echo "✅ Services are running in the background"
    echo "Access the Ishtar AI web app at: http://localhost:8501"
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
}

run_app() {
    echo "🚀 Starting Ishtar AI with Poetry..."
    poetry run streamlit run streamlit_app.py
}

# Main script execution
if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

# Process command line arguments
DO_INSTALL=false
DO_RUN=false
DO_DOCKER=false

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
        --docker)
            DO_DOCKER=true
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

# Check/install Poetry
install_poetry

# Initialize Poetry project
initialize_poetry_project

# Install dependencies if requested
if $DO_INSTALL; then
    install_dependencies
fi

# Build and run with Docker if requested
if $DO_DOCKER; then
    check_docker && run_with_docker
fi

# Run the application if requested
if $DO_RUN; then
    run_app
fi

echo "✨ Setup complete!"
if ! $DO_RUN && ! $DO_DOCKER; then
    echo "To run the application: ./poetry_setup.sh --run"
    echo "To run with Docker: ./poetry_setup.sh --docker"
fi 