import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, root_validator

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


class IshtarSettings(BaseSettings):
    # Ollama API Configuration
    ollama_host: str = Field(default="localhost", env="OLLAMA_HOST")
    ollama_port: str = Field(default="11434", env="OLLAMA_PORT")

    # Default model to use
    default_model: str = Field(default="llama3", env="DEFAULT_MODEL")

    # Generation parameters
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=1000, env="DEFAULT_MAX_TOKENS")

    # Tavily API key
    tavily_api_key: str = Field(default=None, env="TAVILY_API_KEY")

    # NewsAPI key
    newsapi_key: str = Field(
        default="b3e63a8ea4b84d858d7784cc6a46a2e3", env="NEWSAPI_KEY"
    )

    # OpenAI API key
    openai_api_key: str = Field(default=None, env="OPENAI_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @root_validator(skip_on_failure=True)
    def validate_api_keys(cls, values):
        if values.get("tavily_api_key"):
            print(
                f"Tavily API key found (length: {len(values['tavily_api_key'])})",
                file=sys.stderr,
            )
        else:
            print("No Tavily API key found in environment variables", file=sys.stderr)
        return values


# Initialize settings
settings = IshtarSettings()


# Configure the client (for backward compatibility)
def get_client_config():
    return {
        "base_url": settings.ollama_base_url,
        "default_model": settings.default_model,
        "default_params": {
            "temperature": settings.default_temperature,
            "max_tokens": settings.default_max_tokens,
        },
    }
