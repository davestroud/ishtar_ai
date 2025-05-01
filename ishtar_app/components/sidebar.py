#!/usr/bin/env python3
import streamlit as st
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_available_models():
    """Get a list of available models from Hugging Face"""
    return [
        # Smaller models that work well on most hardware
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Very small, fast
        "facebook/opt-350m",  # Small, fast
        "bigscience/bloom-560m",  # Small, fast
        "EleutherAI/pythia-410m",  # Small, fast
        "microsoft/phi-2",  # Medium size
        "google/gemma-2b",  # Medium size
        # Larger models that require disk offloading
        "google/gemma-7b",  # Larger, needs offloading
        "HuggingFaceH4/zephyr-7b-beta",  # Larger, needs offloading
        # Premium/gated models - need special access
        "meta-llama/Meta-Llama-3-8B-Instruct",  # Premium model
        "meta-llama/Llama-3-70B-Instruct",  # Premium model
        "meta-llama/Meta-Llama-4-8B-Instruct",  # Premium model
        "meta-llama/Meta-Llama-4-48B-Instruct",  # Premium model
        "mistralai/Mistral-7B-Instruct-v0.2",  # Premium model
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

    # Try to automatically load a small model at startup
    if "hf_model" not in clients or clients["hf_model"] is None:
        try:
            # Choose a small model that loads quickly
            default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logging.info(f"Auto-loading small model {default_model} at startup")

            # Get token
            hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv(
                "HUGGING_FACE_TOKEN"
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(default_model, token=hf_token)
            clients["hf_tokenizer"] = tokenizer

            # Load the model directly
            import torch
            from transformers import pipeline

            model = AutoModelForCausalLM.from_pretrained(
                default_model,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token,
            )

            # Create pipeline
            text_generation = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
            )

            # Store in clients dictionary
            clients["hf_model"] = text_generation
            logging.info(f"Small model {default_model} auto-loaded successfully")
        except Exception as e:
            logging.error(f"Error auto-loading default model: {str(e)}")

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

        # Check for Hugging Face token (try both API_KEY and TOKEN for backward compatibility)
        hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token and "meta-llama" in selected_model:
            st.error("⚠️ Hugging Face token required for Meta models")
            st.info("Please add your HUGGINGFACE_API_KEY to the .env file")

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
                    tokenizer = AutoTokenizer.from_pretrained(
                        selected_model, token=hf_token
                    )
                    # Store the tokenizer for later use
                    clients["hf_tokenizer"] = tokenizer

                    # Don't load the actual model here - we'll only load on demand when a query is made
                    # This prevents the model from being loaded multiple times and saves memory
                    st.success(f"Tokenizer for {selected_model} loaded successfully")
                    settings["model_loaded"] = False
                    settings["model"] = selected_model

                    # Add notes about disk offloading for larger models
                    if any(
                        name in selected_model.lower()
                        for name in ["llama", "mistral-7b", "zephyr-7b"]
                    ):
                        st.info(
                            "This is a large model. It will be loaded with disk offloading when you send your first message."
                        )
                    elif any(
                        name in selected_model.lower() for name in ["phi-2", "gemma-2b"]
                    ):
                        st.info(
                            "This is a medium-sized model. It will be loaded when you send your first message."
                        )
                    else:
                        st.info(
                            "This is a small model. It will load quickly when you send your first message."
                        )

                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
                    logging.error(f"Failed to load model {selected_model}: {str(e)}")
                    settings["model_loaded"] = False

                    # Add model to settings anyway to prevent NoneType errors
                    st.warning("Using fallback settings. Some features may be limited.")
                    settings["model"] = (
                        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fallback to smallest model
                    )

                    # Try to load the fallback model
                    try:
                        st.info("Loading tokenizer for fallback model...")
                        fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

                        tokenizer = AutoTokenizer.from_pretrained(
                            fallback_model, token=hf_token
                        )

                        clients["hf_tokenizer"] = tokenizer
                        settings["model"] = fallback_model
                        st.success(f"Fallback tokenizer loaded successfully")
                    except Exception as fallback_error:
                        st.error(
                            f"Failed to load fallback tokenizer: {str(fallback_error)}"
                        )
                        logging.error(
                            f"Failed to load fallback tokenizer: {str(fallback_error)}"
                        )

        # Web search settings
        st.divider()
        web_search_enabled = st.toggle(
            "Enable Web Search",
            value=True,
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

        # Update or add huggingface token (use API_KEY as the preferred key)
        env_data["HUGGINGFACE_API_KEY"] = token

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
