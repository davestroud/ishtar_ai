#!/usr/bin/env python3
import streamlit as st
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_available_models():
    """Get a list of available models from Hugging Face"""
    return [
        "meta-llama/Meta-Llama-3-8B-Instruct",  # Llama 3 8B
        "meta-llama/Llama-3-70B-Instruct",  # Llama 3 70B
        "meta-llama/Meta-Llama-4-8B-Instruct",  # Llama 4 8B
        "meta-llama/Meta-Llama-4-48B-Instruct",  # Llama 4 48B
        "google/gemma-2b",
        "google/gemma-7b",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "HuggingFaceH4/zephyr-7b-beta",
    ]


def render_sidebar(clients: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render the sidebar with configuration options and return the settings

    Args:
        clients: Dictionary of API clients

    Returns:
        Dict containing user settings
    """
    settings = {}

    with st.sidebar:
        st.title("🤖 Ishtar AI")
        st.subheader("Configuration")

        # Model selection
        models_list = get_available_models()

        selected_model = st.selectbox(
            "Select Model",
            options=models_list,
            index=0,
            help="Choose the Hugging Face model to use for generating responses",
        )
        settings["model"] = selected_model

        # Check for Hugging Face token
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token and "meta-llama" in selected_model:
            st.error("⚠️ Hugging Face token required for Meta models")
            st.info("Please add your HUGGING_FACE_TOKEN to the .env file")

            # Form to add Hugging Face token
            with st.form("hf_token_form"):
                new_token = st.text_input("Enter Hugging Face Token", type="password")
                submit_button = st.form_submit_button("Save Token")

                if submit_button and new_token:
                    if save_huggingface_token(new_token):
                        st.success("Token saved successfully! Restarting app...")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to save token")

            st.markdown(
                """
            To get your token:
            1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
            2. Create a new token
            3. Enter it above
            """
            )
        elif not hf_token:
            st.warning(
                "No Hugging Face token found. Some models may not be accessible."
            )

            # Form to add Hugging Face token
            with st.expander("Add Hugging Face Token"):
                with st.form("hf_token_form"):
                    new_token = st.text_input(
                        "Enter Hugging Face Token", type="password"
                    )
                    submit_button = st.form_submit_button("Save Token")

                    if submit_button and new_token:
                        if save_huggingface_token(new_token):
                            st.success("Token saved successfully! Restarting app...")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to save token")

        # Load selected model if needed
        if "hf_model" in clients and clients["hf_model"] is None:
            with st.status("Loading model, please wait...", expanded=True):
                try:
                    # Only load the tokenizer first to check access and save memory
                    tokenizer = AutoTokenizer.from_pretrained(selected_model)
                    # We'll load the actual model on demand when needed
                    clients["hf_tokenizer"] = tokenizer
                    settings["model_loaded"] = True
                    st.success(f"Model {selected_model} is ready to use")
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
                    logging.error(f"Failed to load model {selected_model}: {str(e)}")
                    settings["model_loaded"] = False

        # Web search settings
        st.divider()
        web_search_enabled = st.toggle(
            "Enable Web Search",
            value=False,
            help="When enabled, the assistant can search the web for information",
        )
        settings["web_search_enabled"] = web_search_enabled

        # Only show web search settings if web search is enabled
        if web_search_enabled:
            if "tavily" not in clients:
                st.warning(
                    "Tavily API key not found. Add it to your .env file to enable web search."
                )

            search_depth = st.select_slider(
                "Search Depth",
                options=["basic", "advanced"],
                value="basic",
                help="Basic is faster, Advanced is more thorough",
            )
            settings["search_depth"] = search_depth

            max_results = st.slider(
                "Max Search Results",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum number of search results to include",
            )
            settings["max_results"] = max_results

        # Advanced LLM settings
        st.divider()
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values make output more random, lower values more deterministic",
            )
            settings["temperature"] = temperature

            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                help="Maximum length of the model response",
            )
            settings["max_tokens"] = max_tokens

            debug_mode = st.checkbox(
                "Debug Mode", value=False, help="Show additional debugging information"
            )
            settings["debug_mode"] = debug_mode

            # LangSmith tracing
            if "langsmith" in clients:
                enable_tracing = st.checkbox(
                    "Enable LangSmith Tracing",
                    value=False,
                    help="Record interactions in LangSmith for debugging",
                )
                settings["enable_tracing"] = enable_tracing
            else:
                settings["enable_tracing"] = False
                if debug_mode:
                    st.info(
                        "LangSmith integration not available. Add LANGCHAIN_API_KEY to your .env file to enable tracing."
                    )

        # About section
        st.divider()
        with st.expander("About"):
            st.markdown(
                """
            **Ishtar AI Assistant**
            
            A locally-running AI assistant powered by:
            - Hugging Face for local LLM inference
            - Tavily for web search capabilities
            - LangSmith for tracing and debugging
            
            Built with Streamlit and Python.
            """
            )

    return settings


def save_huggingface_token(token: str) -> bool:
    """Save Hugging Face token to .env file"""
    try:
        # Check if .env exists
        env_file = ".env"
        env_data = {}

        # Read existing .env file if it exists
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                for line in f:
                    if line.strip() and not line.strip().startswith("#"):
                        key, value = line.strip().split("=", 1)
                        env_data[key] = value

        # Update or add huggingface token
        env_data["HUGGING_FACE_TOKEN"] = token

        # Write back to .env file
        with open(env_file, "w") as f:
            for key, value in env_data.items():
                f.write(f"{key}={value}\n")

        # Reload environment
        load_dotenv(override=True)
        return True
    except Exception as e:
        logging.error(f"Failed to save Hugging Face token: {str(e)}")
        return False
