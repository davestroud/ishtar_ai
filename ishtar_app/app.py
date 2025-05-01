#!/usr/bin/env python3
import streamlit as st
import logging
import sys
import os
from dotenv import load_dotenv
import traceback

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix "no running event loop" error
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    # No running event loop, create a new one
    asyncio.set_event_loop(asyncio.new_event_loop())

# Patch for the torch.__path__._path error in Streamlit
# This occurs because Streamlit tries to extract paths from torch modules
# which have custom attribute handling
import types
import torch


# Create a helper function to safely get paths without error
def safe_extract_paths(module):
    if module.__name__.startswith("torch"):
        # For torch modules, return empty list to avoid the error
        return []

    # For regular modules with __path__ attribute
    if hasattr(module, "__path__"):
        # Handle torch._classes which causes the error
        if isinstance(module.__path__, types.ModuleType) or not hasattr(
            module.__path__, "_path"
        ):
            return []
        return list(module.__path__._path) if hasattr(module.__path__, "_path") else []
    return []


# Apply patch to Streamlit's local_sources_watcher if possible
try:
    import streamlit.watcher.local_sources_watcher as lsw

    # Store the original function
    original_extract_paths = lsw.extract_paths

    # Define our patched function
    def patched_extract_paths(module):
        try:
            return original_extract_paths(module)
        except (AttributeError, RuntimeError):
            return safe_extract_paths(module)

    # Apply the patch
    lsw.extract_paths = patched_extract_paths
except (ImportError, AttributeError):
    pass

# Import components using absolute imports
from ishtar_app.components.sidebar import render_sidebar
from ishtar_app.components.chat import render_chat_ui
from ishtar_app.components.header import render_header

# Import API integrations
from src.config import IshtarSettings, settings
from src.langsmith_integration import get_langsmith_client
from src.tavily_search import TavilySearch

# Import optional modules if available
try:
    from src.weather_api import WeatherAPI
    from src.pinecone_integration import PineconeClient
except ImportError:
    WeatherAPI = None
    PineconeClient = None

# Import Hugging Face transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set Hugging Face token if available (try API_KEY first, then TOKEN for backward compatibility)
hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGING_FACE_TOKEN")
if hf_token:
    import huggingface_hub

    huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    logger.info("Logged in to Hugging Face Hub")
else:
    logger.warning("No Hugging Face token found. Some models may not be accessible.")


def initialize_clients():
    """Initialize API clients based on available credentials"""
    clients = {}

    # Initialize Hugging Face model
    try:
        # We'll initialize this on-demand in the sidebar component
        clients["hf_model"] = None
        logger.info("Hugging Face model will be loaded on demand")
    except Exception as e:
        logger.error(f"Error initializing Hugging Face: {str(e)}")

    # Initialize Tavily client if API key is available
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            clients["tavily"] = TavilySearch(api_key=tavily_api_key)
            logger.info("Tavily Search client initialized")
        except Exception as e:
            logger.error(f"Error initializing Tavily client: {str(e)}")

    # Initialize Weather API if module is available and API key exists
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")
    if WeatherAPI and weather_api_key:
        try:
            clients["weather"] = WeatherAPI(api_key=weather_api_key)
            logger.info("Weather API client initialized")
        except Exception as e:
            logger.error(f"Error initializing Weather API: {str(e)}")

    # Initialize Pinecone client if module is available and API key exists
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_HOST")
    if PineconeClient and pinecone_api_key and pinecone_host:
        try:
            clients["pinecone"] = PineconeClient(
                api_key=pinecone_api_key, host=pinecone_host
            )
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {str(e)}")

    # Initialize LangSmith client if API keys are available
    try:
        # Check for LangSmith API key directly
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

        if langchain_api_key or langsmith_api_key:
            try:
                from langsmith import Client

                # Use whichever key is available
                api_key = langchain_api_key or langsmith_api_key
                langsmith_client = Client(api_key=api_key)

                clients["langsmith"] = langsmith_client
                logger.info(
                    f"LangSmith client initialized with API key (length: {len(api_key)})"
                )

                # Also set tracing to enabled
                os.environ["LANGSMITH_TRACING"] = "true"
            except ImportError:
                logger.error(
                    "LangSmith package not installed. Try running 'pip install langsmith'"
                )
            except Exception as e:
                logger.error(f"Error creating LangSmith client directly: {str(e)}")
        else:
            # Fall back to the integration module
            langsmith_client = get_langsmith_client()
            if langsmith_client:
                clients["langsmith"] = langsmith_client
                logger.info("LangSmith client initialized via integration module")
    except Exception as e:
        logger.error(f"Error initializing LangSmith client: {str(e)}")

    return clients


def main():
    """Main Streamlit application"""
    # Set page config
    st.set_page_config(
        page_title="Ishtar AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize API clients
    clients = initialize_clients()

    try:
        # Render header
        render_header()

        # Render sidebar and get user settings
        settings = render_sidebar(clients)

        # Display chat UI with the settings
        render_chat_ui(clients, settings)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if st.checkbox("Show detailed error"):
            st.code(traceback.format_exc())
        logger.error(f"Application error: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
