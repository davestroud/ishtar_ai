# Ollama Docker Client

A simple Python client for interacting with Ollama LLM running in Docker.

## Prerequisites

- Docker installed on your system
- Python 3.7+ (if running locally)

## Setup Options

### Option 1: Using Docker Compose (Recommended)

1. Start both Ollama and the client application:

```bash
docker-compose up -d
```

2. Access the Streamlit web app at: http://localhost:8501

### Option 2: Running Ollama in Docker and Client Locally

1. Start Ollama in Docker:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit web app:

```bash
streamlit run streamlit_app.py
```

### Option 3: Running in Docker individually

1. Start Ollama in Docker:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

2. Build and run the client app in Docker:

```bash
docker build -t ollama-client .
docker run -d -p 8501:8501 --name ollama-client --link ollama:ollama ollama-client
```

## Usage

### CLI Client

Run the basic CLI client:

```bash
python ollama_client.py
```

This will:
- List available models
- Generate a response to a sample prompt
- Run a sample chat conversation

### Streamlit Web App

Features:
- Chat interface for conversations
- Model selection
- Pull new models directly from the UI
- Adjust generation parameters

### Testing Embeddings

Run the embeddings example:

```bash
python sample_embeddings.py
```

### Using the API Directly

Try the curl examples:

```bash
chmod +x curl_examples.sh
./curl_examples.sh
```

## Managing Ollama Models

Pull a model from Ollama (either through the API or using Docker):

```bash
# Using Docker
docker exec -it ollama ollama pull llama3
```

List available models:

```bash
# Using Docker
docker exec -it ollama ollama list
```

## Web Search Integration

The app now includes web search capabilities using the Tavily API. This allows the model to provide up-to-date information from the internet.

### Setting up Tavily API

1. Sign up for a Tavily API key at [tavily.com](https://tavily.com)
2. Set the API key as an environment variable:
   ```bash
   export TAVILY_API_KEY=your_api_key_here
   ```
3. Alternatively, you can enter the API key directly in the app's web interface

### Using Web Search

1. Enable the "Web Search" checkbox in the sidebar
2. Ask questions that may benefit from current information
3. The app will search the web for relevant information before responding

## API Reference

The `OllamaClient` class provides these main methods:

- `list_models()`: Get available models
- `generate(prompt, model)`: Generate text with a single prompt
- `chat(messages, model)`: Chat using a conversation format
- `pull_model(model_name)`: Pull a model from Ollama
- `stream_generate(prompt, model, callback)`: Stream responses with callback

## Troubleshooting

- If you get connection errors, make sure the Ollama Docker container is running
- Check Docker logs: `docker logs ollama`
- Make sure port 11434 is accessible and not blocked by a firewall 