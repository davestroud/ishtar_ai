#!/bin/bash
# Script to restart the Ollama Docker container and run the Streamlit app

# Activate virtual environment
source bin/activate

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Restart the Ollama Docker container
echo "Restarting Ollama container..."
docker restart ollama || {
    echo "Failed to restart Ollama container. Making sure it's running..."
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
}

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 5

echo "Starting Streamlit app..."
# Run the Streamlit app
streamlit run src/streamlit_app.py 