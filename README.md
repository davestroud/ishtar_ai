# Ishtar AI

An AI assistant that runs locally using Hugging Face models.

## Features

- Chat with various LLM models from Hugging Face (Llama 3, Llama 4, Mistral, etc.)
- Web search capability using Tavily (optional)
- Debugging and tracing with LangSmith (optional)

## Setup

1. Ensure you have Python 3.10+ installed
2. Install requirements: `pip install -r requirements.txt`
3. Set up your environment variables in `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_api_key
   ```

## Running the Application

Simply execute:

```bash
./run_app.sh
```

Or run directly with Python:

```bash
python -m ishtar_app.app
```

## Accessing Models

To use Meta Llama models and other gated models, you need a Hugging Face API key:

1. Create an account at [Hugging Face](https://huggingface.co/)
2. Generate an API token at https://huggingface.co/settings/tokens
3. Add the token to your `.env` file as `HUGGINGFACE_API_KEY`

## Additional API Keys (Optional)

For enhanced functionality, you can add these API keys:

- `TAVILY_API_KEY`: For web search capabilities
- `LANGCHAIN_API_KEY`: For tracing and debugging with LangSmith

## Application Structure

```
ishtar_ai/
├── retrieval/            # Core retrieval functionality
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