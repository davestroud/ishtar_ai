import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field, root_validator
from typing import Optional, Union

# Try to import BaseSettings from pydantic_settings, fall back to regular BaseModel if not available
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # If pydantic_settings is not available, use BaseModel from pydantic directly
    from pydantic import BaseModel as BaseSettingsBase

    class BaseSettings(BaseSettingsBase):
        """
        Fallback BaseSettings implementation when pydantic_settings is not available.
        This provides basic functionality similar to BaseSettings.
        """

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

        def __init__(self, **data):
            # Load environment variables for fields
            env_data = {}
            for field_name, field_info in self.__class__.__fields__.items():
                # Get env var name from field extra info or use uppercase field name
                env_var = None

                # Try different ways to access the env attribute based on Pydantic version
                if hasattr(field_info, "field_info") and hasattr(
                    field_info.field_info, "extra"
                ):
                    # Pydantic v2 style
                    env_var = field_info.field_info.extra.get("env", field_name.upper())
                elif hasattr(field_info, "extra"):
                    # Pydantic v1 style
                    env_var = field_info.extra.get("env", field_name.upper())
                else:
                    # Fallback to uppercase field name
                    env_var = field_name.upper()

                if env_var in os.environ:
                    env_data[field_name] = os.environ[env_var]

            # Override with any explicitly provided values
            env_data.update(data)
            super().__init__(**env_data)


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
    # Hugging Face configuration
    huggingface_token: Optional[str] = Field(default=None, env="HUGGING_FACE_TOKEN")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")

    # Default model to use
    default_model: str = Field(default="google/gemma-2b", env="DEFAULT_MODEL")

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

    # Pinecone configuration
    pinecone_api_key: str = Field(default=None, env="PINECONE_API_KEY")
    pinecone_host: str = Field(default=None, env="PINECONE_HOST")
    pinecone_index: str = Field(default="ishtar", env="PINECONE_INDEX")

    # LangSmith configuration
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="ishtar-ai", env="LANGCHAIN_PROJECT")
    langsmith_tracing: bool = Field(default=True, env="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT"
    )
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow extra fields to avoid validation errors for new environment variables
        extra = "allow"

    @root_validator(skip_on_failure=True)
    def validate_api_keys(cls, values):
        # Check Tavily API key
        if values.get("tavily_api_key"):
            print(
                f"Tavily API key found (length: {len(values['tavily_api_key'])})",
                file=sys.stderr,
            )
        else:
            print("No Tavily API key found in environment variables", file=sys.stderr)

        # Check LangChain API key
        if values.get("langchain_api_key"):
            print(
                f"LangChain API key found (length: {len(values['langchain_api_key'])})",
                file=sys.stderr,
            )
        else:
            print(
                "No LangChain API key found in environment variables", file=sys.stderr
            )
            print("LangSmith tracing will be disabled", file=sys.stderr)
            values["langsmith_tracing"] = False

        # Check Hugging Face token/API key (try API_KEY first, then TOKEN for backward compatibility)
        if values.get("huggingface_api_key"):
            print(
                f"Hugging Face API key found (length: {len(values['huggingface_api_key'])})",
                file=sys.stderr,
            )
        elif values.get("huggingface_token"):
            print(
                f"Hugging Face token found (length: {len(values['huggingface_token'])})",
                file=sys.stderr,
            )
        else:
            print(
                "No Hugging Face API key or token found in environment variables",
                file=sys.stderr,
            )
            print("Some models may not be accessible", file=sys.stderr)

        return values


# Initialize settings
settings = IshtarSettings()


# Configure the client (for backward compatibility)
def get_client_config():
    return {
        "default_model": settings.default_model,
        "default_params": {
            "temperature": settings.default_temperature,
            "max_tokens": settings.default_max_tokens,
        },
    }
