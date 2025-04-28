# Ishtar AI Project

Ishtar AI is a comprehensive AI platform for analyzing and providing insights on conflict zones, humanitarian crises, and regional developments.

## Features

- **Chat Interface**: Interact with various AI models through an intuitive Streamlit UI
- **Web Search**: Get real-time information from the web using Tavily integration
- **Vector Database**: Store and retrieve relevant information using Pinecone
- **News Integration**: Fetch and index news articles from various sources
- **ASGI API**: Access all functionality through a modern API with WebSocket support
- **Real-time Chat**: Use WebSockets for bidirectional communication with AI models

## Getting Started

### Prerequisites

- Python 3.11+
- Poetry for dependency management
- Ollama for local AI model hosting
- Required API keys (OpenAI, Tavily, Pinecone, LangSmith, NewsAPI)

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/ishtar_ai.git
   cd ishtar_ai
   ```

2. Install dependencies
   ```bash
   poetry install
   ```

3. Set up environment variables by creating a `.env` file or using the provided scripts:

   See [API_SETUP.md](API_SETUP.md) for detailed instructions on setting up API keys.

   ```
   # Required API keys
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_HOST=your_pinecone_host
   PINECONE_INDEX=ishtar
   
   # LangSmith configuration
   LANGCHAIN_API_KEY=your_langsmith_key
   LANGCHAIN_PROJECT=ishtar-ai
   LANGSMITH_TRACING=true
   
   # News API configuration
   NEWSAPI_KEY=your_newsapi_key
   
   # Ollama configuration (if using Docker)
   OLLAMA_HOST=host.docker.internal
   ```

   Alternatively, use the utility scripts to set API keys:
   ```bash
   # Set API keys individually
   ./scripts/update_tavily_key.sh YOUR_TAVILY_API_KEY
   ./scripts/update_langsmith_key.sh YOUR_LANGSMITH_API_KEY
   ./scripts/update_pinecone.sh --api-key YOUR_API_KEY --host YOUR_HOST
   ```

   > Note: The helper script `run_ishtar.sh` will automatically set a default value for `LANGCHAIN_PROJECT` but you must provide your own `LANGCHAIN_API_KEY`.

## Running the Application

### Using the Helper Script (Recommended)

The easiest way to run Ishtar AI is using the provided helper script, which automatically sets the required environment variables:

```bash
# Make the script executable
chmod +x run_ishtar.sh

# Run Streamlit app (default)
./run_ishtar.sh --streamlit

# Run ASGI app
./run_ishtar.sh --asgi

# Run with Docker
./run_ishtar.sh --docker

# Run ASGI with Docker
./run_ishtar.sh --docker-asgi

# Show help
./run_ishtar.sh --help
```

### Streamlit Interface

```bash
poetry run streamlit run streamlit_app.py
```

Access the Streamlit UI at http://localhost:8501

### ASGI API Interface

```bash
# Run with Python
poetry run python asgi_app.py

# Or run with Uvicorn directly
poetry run uvicorn asgi_app:app --host 0.0.0.0 --port 8000 --reload
```

Access:
- ASGI API: http://localhost:8000
- WebSocket Demo: http://localhost:8000/static/index.html
- API Documentation: http://localhost:8000/docs

### Using Docker Compose

```bash
# For Streamlit only
docker-compose up

# For ASGI API (includes Streamlit)
docker-compose -f docker-compose-asgi.yml up
```

## Project Structure

```
ishtar_ai/
├── streamlit_app.py   # Streamlit UI
├── asgi_app.py        # ASGI API
├── ollama_client.py   # Ollama client integration
├── src/
│   ├── langsmith_integration.py  # LangSmith tracing
│   ├── newsapi_integration.py    # News API integration
│   ├── pinecone_integration.py   # Pinecone vector DB
│   └── tavily_search.py          # Tavily search API
├── scripts/           # Utility scripts
│   ├── update_*.sh    # API key update scripts
│   └── env/           # Environment management scripts
├── utils/
│   └── news_fetcher.py           # Command-line news fetcher
└── static/
    └── index.html                # WebSocket demo interface
```

### Utility Scripts

The project includes several utility scripts organized in directories:

- **Root Directory Scripts**:
  - `setup.sh` - Main setup script for the application
  - `poetry_setup.sh` - Alternative setup script using Poetry
  - `run_ishtar.sh` - Script to run the application in different modes

- **API Key Management** (`scripts/`):
  - `update_langsmith_key.sh` - Update LangSmith API key
  - `update_pinecone.sh` - Update Pinecone credentials
  - `update_tavily_key.sh` - Update Tavily API key
  - `update_weather_key.sh` - Update OpenWeather API key

- **Environment Scripts** (`scripts/env/`):
  - `restart_app.sh` - Restart Ollama container and run Streamlit
  - `run_with_tavily.sh` - Run with Tavily web search integration

These scripts have symbolic links in the project root for easy access.

## ASGI API

The ASGI application provides both REST endpoints and WebSocket support:

### REST Endpoints

- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `POST /api/search` - Search the web with Tavily
- `POST /api/vector-search` - Search vectors in Pinecone

### WebSocket Endpoint

- `WebSocket /ws/chat` - Real-time chat interface

For more details, see [ASGI_README.md](ASGI_README.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 