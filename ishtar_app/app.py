#!/usr/bin/env python3
"""
Ishtar AI Application
Main application file for running the Streamlit interface
"""

import os
import sys
import logging

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and apply monkeypatches to fix PyTorch/Streamlit compatibility early
from ishtar_app.monkeypatch import apply_monkeypatches, apply_torch_patches

# Apply monkeypatches before any other imports to ensure PyTorch works with Streamlit
apply_monkeypatches()
apply_torch_patches()

# Standard imports
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from dotenv import load_dotenv
import traceback

# Import components and settings
from ishtar_app.components.sidebar import render_sidebar
from ishtar_app.components.chat import render_chat_ui
from ishtar_app.components.header import render_header

# Import retrieval modules
from retrieval.config import IshtarSettings, settings
from retrieval.langsmith_integration import get_langsmith_client
from retrieval.tavily_search import TavilySearch
from retrieval.newsapi_integration import NewsAPIClient
from retrieval.weather_api import WeatherAPI
from retrieval.pinecone_integration import PineconeClient

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

    # Initialize LangSmith client
    try:
        # First ensure required environment variables are set
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv(
            "LANGSMITH_API_KEY"
        )

        if langchain_api_key:
            # Import the function directly
            from retrieval.langsmith_integration import get_langsmith_client

            # Initialize client
            langsmith_client = get_langsmith_client()

            if langsmith_client:
                clients["langsmith"] = langsmith_client
                logger.info("LangSmith client initialized successfully")

                # Success message with project link
                project_name = os.environ.get("LANGCHAIN_PROJECT", "ishtar_ai")
                logger.info(
                    f"LangSmith dashboard: https://smith.langchain.com/projects/{project_name}"
                )
            else:
                logger.warning("LangSmith client could not be initialized")
                os.environ["LANGSMITH_TRACING"] = "false"
        else:
            logger.info("No LangSmith API key found. Tracing will be disabled.")
            os.environ["LANGSMITH_TRACING"] = "false"
    except Exception as e:
        logger.error(f"Error initializing LangSmith client: {str(e)}")
        os.environ["LANGSMITH_TRACING"] = "false"

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
