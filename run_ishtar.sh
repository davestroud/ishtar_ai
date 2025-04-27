#!/bin/bash
# Helper script to run Ishtar AI with proper environment variables

# Ensure LANGCHAIN_API_KEY is set
if [ -z "$LANGCHAIN_API_KEY" ]; then
    export LANGCHAIN_API_KEY="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"
    echo "Setting default LANGCHAIN_API_KEY"
fi

# Ensure LANGCHAIN_PROJECT is set
if [ -z "$LANGCHAIN_PROJECT" ]; then
    export LANGCHAIN_PROJECT="ishtar-ai"
    echo "Setting default LANGCHAIN_PROJECT"
fi

# Enable LangSmith tracing
export LANGSMITH_TRACING=true

# Function to show help
show_help() {
    echo "Ishtar AI Runner"
    echo "Usage: ./run_ishtar.sh [options]"
    echo ""
    echo "Options:"
    echo "  --streamlit    Run the Streamlit app (default)"
    echo "  --asgi         Run the ASGI app"
    echo "  --docker       Run with Docker Compose"
    echo "  --docker-asgi  Run ASGI with Docker Compose"
    echo "  --help         Show this help message"
    echo ""
}

# Parse command line arguments
RUN_TYPE="streamlit"

while [[ $# -gt 0 ]]; do
    case $1 in
        --streamlit)
            RUN_TYPE="streamlit"
            shift
            ;;
        --asgi)
            RUN_TYPE="asgi"
            shift
            ;;
        --docker)
            RUN_TYPE="docker"
            shift
            ;;
        --docker-asgi)
            RUN_TYPE="docker-asgi"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run the appropriate component
case $RUN_TYPE in
    streamlit)
        echo "Starting Streamlit app..."
        poetry run streamlit run streamlit_app.py
        ;;
    asgi)
        echo "Starting ASGI app..."
        poetry run uvicorn asgi_app:app --host 0.0.0.0 --port 8000 --reload
        ;;
    docker)
        echo "Starting Streamlit app with Docker..."
        docker-compose up
        ;;
    docker-asgi)
        echo "Starting ASGI app with Docker..."
        docker-compose -f docker-compose-asgi.yml up
        ;;
esac 