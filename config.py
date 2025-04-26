import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
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
