# API Keys Setup Guide for Ishtar AI

This document explains how to set up the various API keys needed for Ishtar AI to function properly.

## Required API Keys

Ishtar AI can use several external APIs to enhance its capabilities:

1. **OpenWeatherMap API** - For real-time weather data
2. **Tavily API** - For web search capabilities
3. **LangSmith API** - For tracing and debugging LLM interactions
4. **Pinecone API** - For vector database storage

## Setting Up API Keys

We've provided simple scripts to help you set up each API key.

### 1. OpenWeatherMap API (for Weather Data)

1. Sign up for a free API key at [OpenWeatherMap](https://openweathermap.org/)
2. Run the following command:
   ```bash
   ./update_weather_key.sh YOUR_OPENWEATHER_API_KEY
   ```

### 2. Tavily API (for Web Search)

1. Sign up for an API key at [Tavily](https://tavily.com/)
2. Run the following command:
   ```bash
   ./update_tavily_key.sh YOUR_TAVILY_API_KEY
   ```

### 3. LangSmith API (for LLM Tracing)

1. Sign up for an account at [LangSmith](https://smith.langchain.com/)
2. Generate an API key in your LangSmith dashboard
3. Run the following command:
   ```bash
   ./update_langsmith_key.sh YOUR_LANGSMITH_API_KEY
   ```

### 4. Pinecone API (for Vector Database)

1. Sign up for an account at [Pinecone](https://www.pinecone.io/)
2. Create an index and note your API key, index name, and host
3. Add the following variables to your `.env` file:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_index_name
   PINECONE_HOST=your_host_url
   ```

## Checking API Status

When you run the Streamlit app, you can see the status of each API in the sidebar. A green checkmark (✅) indicates that the API is connected and working. A red X (❌) indicates that the API is not connected.

## Troubleshooting

If you're having issues with any of the APIs:

1. Verify that your API key is correct
2. Check if your account has sufficient quota/credits
3. Ensure your API key has the necessary permissions
4. Look for error messages in the terminal logs

Remember to keep your API keys secure and never commit them to public repositories.

## Using the App without APIs

The Ishtar AI app will function even without these APIs, but with reduced capabilities:

- Without OpenWeatherMap API: Cannot provide real-time weather data
- Without Tavily API: Cannot perform web searches
- Without LangSmith API: Cannot trace and debug model interactions
- Without Pinecone API: Cannot use vector search capabilities 