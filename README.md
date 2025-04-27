# Ishtar AI

The Ishtar AI Initiative is dedicated to harnessing the potential of Artificial Intelligence and Large Language Models (LLMs) to provide actionable insights and data analysis to media and journalism entities. Our goal is to support news organizations by delivering enhanced reporting and analytical capabilities for covering conflict zones, humanitarian crises, and regional developments.

## Features

- Web-based interface for querying various LLM providers (Ollama, OpenAI)
- Web search integration for up-to-date information retrieval
- Model selection and parameter configuration
- Pull new models directly from the UI
- Advanced settings for generation parameters
- LangSmith integration for tracing and monitoring

## Prerequisites

- Docker installed on your system
- Python 3.7+ (if running locally)

## Quick Start

The easiest way to get started is to use our setup script:

```bash
# Install dependencies and start the app
./setup.sh --install --run

# Set API keys
./setup.sh --tavily-key YOUR_TAVILY_API_KEY --langsmith-key YOUR_LANGSMITH_API_KEY --openai-key YOUR_OPENAI_KEY
```

For help with the setup script:
```bash
./setup.sh --help
```

## Setup Options

### Option 1: Using Docker Compose

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
python -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt
```

3. Run the Streamlit web app:

```bash
streamlit run streamlit_app.py
```

## API Integrations

### Web Search (Tavily)

To enable web search capabilities, you'll need to:

1. Get a Tavily API key from [tavily.com](https://tavily.com)
2. Add it to your .env file or use our setup script:

```bash
./setup.sh --tavily-key YOUR_TAVILY_API_KEY
```

### LangSmith Tracing

Ishtar AI includes integration with [LangSmith](https://smith.langchain.com), LangChain's tracing and monitoring platform. This allows you to:

- Track and monitor all interactions with the LLM
- Debug issues with model responses
- Analyze model performance and user interactions
- Gather analytics on your application's usage

To set up LangSmith:

1. Create an account at [smith.langchain.com](https://smith.langchain.com)
2. Get your API key from your LangSmith account
3. Add it to your .env file or use our setup script:

```bash
./setup.sh --langsmith-key YOUR_LANGSMITH_API_KEY
```

### OpenAI Integration

To use OpenAI models instead of Ollama:

1. Get an API key from [platform.openai.com](https://platform.openai.com)
2. Add it to your .env file or use our setup script:

```bash
./setup.sh --openai-key YOUR_OPENAI_API_KEY
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

## Troubleshooting

- If you get connection errors, make sure the Ollama Docker container is running
- Check Docker logs: `docker logs ollama`
- Make sure port 11434 is accessible and not blocked by a firewall 