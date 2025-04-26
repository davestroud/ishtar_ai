import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Find and load .env file from the project root
# Try to locate .env file in multiple locations
env_paths = [
    Path(".") / ".env",  # Current directory
    Path("..") / ".env",  # Parent directory
    Path(__file__).parent / ".env",  # Same directory as this file
    Path(__file__).parent.parent / ".env",  # Parent of this file's directory
]

for env_path in env_paths:
    if env_path.exists():
        print(f"Loading environment variables from {env_path}", file=sys.stderr)
        load_dotenv(env_path)
        break
else:
    # If no .env file found, still try to load from any .env file
    load_dotenv()

# Ollama API Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Default model to use
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")

# Generation parameters
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))

# Tavily API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    print(f"Tavily API key found (length: {len(TAVILY_API_KEY)})", file=sys.stderr)
else:
    print("No Tavily API key found in environment variables", file=sys.stderr)


# Configure the client
def get_client_config():
    return {
        "base_url": OLLAMA_BASE_URL,
        "default_model": DEFAULT_MODEL,
        "default_params": {
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        },
    }
