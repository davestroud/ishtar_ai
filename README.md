# Ishtar AI Assistant

A locally-running AI assistant powered by Hugging Face models with web search capabilities.

## Features

- 🤖 Chat with various Hugging Face models locally
- 🔍 Web search integration using Tavily Search API
- 📊 Tracing and debugging with LangSmith
- 🎛️ Configurable settings for model parameters
- 💻 Simple, clean interface built with Streamlit

## Requirements

- Python 3.9+
- Hugging Face Transformers
- Tavily API key (optional, for web search)
- LangSmith API key (optional, for tracing and debugging)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ishtar_ai.git
cd ishtar_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=ishtar_ai
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

## Usage

1. Run the application:
```bash
./run_app.sh
```
Or directly with Streamlit:
```bash
streamlit run ishtar_app/app.py
```

2. Open your browser at `http://localhost:8501`

3. Select a model from the sidebar and start chatting!

## Application Structure

```
ishtar_ai/
├── src/                 # Core functionality
│   ├── config.py            # Configuration handling
│   ├── langsmith_integration.py # LangSmith integration
│   ├── newsapi_integration.py   # News API integration
│   ├── pinecone_integration.py  # Pinecone vector DB integration
│   ├── tavily_search.py     # Tavily search integration
│   └── weather_api.py       # Weather API integration
├── ishtar_app/          # Streamlit application
│   ├── app.py               # Main application
│   ├── components/          # UI components
│   │   ├── sidebar.py       # Sidebar with settings
│   │   ├── chat.py          # Chat interface
│   │   └── header.py        # App header
│   └── utils/               # Utility functions
├── requirements.txt     # Python dependencies
├── run_app.sh           # Script to run the application
└── .env                 # Environment variables
```

## Models

The application comes configured with several Hugging Face models:
- google/gemma-2b
- mistralai/Mistral-7B-Instruct-v0.2
- meta-llama/Llama-2-7b-chat-hf
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- microsoft/phi-2
- HuggingFaceH4/zephyr-7b-beta

Note: Some models may require authentication with a Hugging Face token. Add `HUGGING_FACE_TOKEN=your_token` to your `.env` file if needed.

## Development

This project follows a modular architecture:

- Core functionality is implemented in the `src` directory
- The Streamlit application is in the `refactor` directory
- UI components are separated for easier maintenance

To contribute:
1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## License

MIT 