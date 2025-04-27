# Ishtar AI

The Ishtar AI Initiative is dedicated to harnessing the potential of Artificial Intelligence and Large Language Models (LLMs) to provide actionable insights and data analysis to media and journalism entities. Our goal is to support news organizations by delivering enhanced reporting and analytical capabilities for covering conflict zones, humanitarian crises, and regional developments.

## Prerequisites

- Docker installed on your system
- Python 3.7+ (if running locally)

## Setup Options

### Option 1: Using Docker Compose (Recommended)

1. Start both Ollama and the client application:

```bash
docker-compose up -d
```

2. Access the Ishtar AI web app at: http://localhost:8501

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
docker build -t ishtar-ai .
docker run -d -p 8501:8501 --name ishtar-ai --link ollama:ollama ishtar-ai
```

## Features

- Web-based interface for querying LLMs
- Web search integration for up-to-date information retrieval
- Model selection and parameter configuration
- Pull new models directly from the UI
- Advanced settings for generation parameters

## Web Search Integration

To enable web search capabilities, you'll need to:

1. Get a Tavily API key from [tavily.com](https://tavily.com)
2. Update your .env file with your API key:

```bash
./update_tavily_key.sh YOUR_TAVILY_API_KEY
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