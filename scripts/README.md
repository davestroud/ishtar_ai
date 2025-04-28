# Ishtar AI Scripts

This directory contains utility scripts for managing API keys and configurations for the Ishtar AI project.

## Update Scripts

- `update_langsmith_key.sh` - Updates the LangSmith API key in the .env file and configures LangSmith tracing
- `update_pinecone.sh` - Updates Pinecone API key and host URL in the .env file
- `update_tavily_key.sh` - Updates the Tavily API key in the .env file for web search functionality
- `update_weather_key.sh` - Updates the OpenWeather API key in the .env file for weather data functionality

## Usage

You can run these scripts from the root directory using the symbolic links or directly from this directory:

```bash
# From project root:
./update_tavily_key.sh YOUR_TAVILY_API_KEY

# Or from the scripts directory:
cd scripts
./update_tavily_key.sh YOUR_TAVILY_API_KEY
```

Note that symbolic links are maintained in the project root for backward compatibility. 