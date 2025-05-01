#!/usr/bin/env python3
import streamlit as st
import logging
import sys
import os
from dotenv import load_dotenv
import traceback

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components using absolute imports
from ishtar_app.components.sidebar import render_sidebar
from ishtar_app.components.chat import render_chat_ui
from ishtar_app.components.header import render_header

# Import API integrations
from src.config import IshtarSettings, settings
from src.langsmith_integration import get_langsmith_client
from src.tavily_search import TavilySearch

# Import Hugging Face transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set Hugging Face token if available
hf_token = os.getenv("HUGGING_FACE_TOKEN")
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

    # Initialize LangSmith client if API keys are available
    try:
        langsmith_client = get_langsmith_client()
        if langsmith_client:
            clients["langsmith"] = langsmith_client
            logger.info("LangSmith client initialized")
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
