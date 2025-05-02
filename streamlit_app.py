#!/usr/bin/env python3
import streamlit as st
import os
import logging
from retrieval.tavily_search import TavilySearch
from retrieval.langsmith_integration import get_langsmith_client
from retrieval.pinecone_integration import get_pinecone_client
from dotenv import load_dotenv
import json
import inspect
import importlib
import sys
import traceback
from pathlib import Path
import time
import torch
import transformers
import accelerate
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from retrieval.weather_api import WeatherAPI
import openai
import huggingface_hub

# Page configuration
st.set_page_config(page_title="Ishtar AI", page_icon="🔍", layout="wide")

# Configure API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set Hugging Face token if available
hf_token = os.environ.get("HUGGING_FACE_TOKEN")
if hf_token:
    huggingface_hub.login(token=hf_token, add_to_git_credential=False)

# Import langchain_community for embeddings to avoid deprecation warnings
try:
    from langchain_community.embeddings import OpenAIEmbeddings
except ImportError:
    from langchain.embeddings import OpenAIEmbeddings


def is_langsmith_key_set():
    """Check if LangSmith API key is set in .env file or environment"""
    # First check environment
    if os.environ.get("LANGCHAIN_API_KEY"):
        return True

    # Then check .env file if it exists
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                if line.strip().startswith("LANGCHAIN_API_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    if key and key != "your_langsmith_key" and not key.startswith("#"):
                        return True
    return False


# Initialize clients
@st.cache_resource
def get_available_hf_models():
    """Get a list of available models from Hugging Face"""
    return [
        "google/gemma-2b",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "HuggingFaceH4/zephyr-7b-beta",
    ]


@st.cache_resource
def get_tavily_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if api_key:
        try:
            return TavilySearch(api_key)
        except Exception as e:
            st.sidebar.error(f"Error initializing Tavily: {str(e)}")
    return None


@st.cache_resource
def get_ls_client():
    if is_langsmith_key_set():
        try:
            return get_langsmith_client()
        except Exception as e:
            st.warning(f"Error initializing LangSmith client: {str(e)}")
    return None


@st.cache_resource
def get_pc_client():
    try:
        return get_pinecone_client()
    except Exception as e:
        st.sidebar.warning(f"Pinecone client initialization failed: {str(e)}")
        return None


@st.cache_resource
def get_weather_client():
    if WeatherAPI is not None:
        try:
            return WeatherAPI()
        except Exception as e:
            st.sidebar.warning(f"Weather client initialization failed: {str(e)}")
    return None


# Initialize clients, gracefully handling failures
try:
    tavily_client = get_tavily_client()
except Exception as e:
    st.warning(f"Failed to initialize Tavily client: {str(e)}")
    tavily_client = None

try:
    langsmith_client = get_ls_client()
except Exception as e:
    st.warning(f"Failed to initialize LangSmith client: {str(e)}")
    langsmith_client = None

try:
    pinecone_client = get_pc_client()
except Exception as e:
    st.warning(f"Failed to initialize Pinecone client: {str(e)}")
    pinecone_client = None

try:
    weather_client = get_weather_client()
except Exception as e:
    st.warning(f"Failed to initialize Weather client: {str(e)}")
    weather_client = None

# Page title and description
st.title("🔍 Ishtar AI")

# Mission statement
st.markdown(
    """
    The Ishtar AI Initiative is dedicated to harnessing the potential of Artificial Intelligence 
    and Large Language Models (LLMs) to provide actionable insights and data analysis to media 
    and journalism entities. Our goal is to support news organizations by delivering enhanced 
    reporting and analytical capabilities for covering conflict zones, humanitarian crises, 
    and regional developments.
"""
)

st.markdown("---")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("Settings")

    # Select model provider
    model_provider = st.radio(
        "Select Model Provider",
        ["Hugging Face", "OpenAI"],
        index=0,
        help="Choose which LLM provider to use",
    )

    # Model selection based on provider
    if model_provider == "Hugging Face":
        model_names = get_available_hf_models()
        selected_model = st.selectbox("Select Hugging Face Model", model_names)

        if hf_token is None:
            st.warning(
                "No Hugging Face token found. Some models may not be accessible."
            )
            st.info(
                "Add HUGGING_FACE_TOKEN to your .env file if needed for accessing restricted models."
            )
    else:
        # OpenAI models
        openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Select OpenAI Model", openai_models)

        if not openai.api_key:
            st.error(
                "OpenAI API key not found. Please add OPENAI_API_KEY to your .env file."
            )

    # Debug mode
    debug_mode = st.checkbox("Enable debug mode", value=False)

    # API Status Section
    st.sidebar.subheader("API Status")

    # Tavily API status
    tavily_available = tavily_client is not None
    tavily_status = "✅ Connected" if tavily_available else "❌ Not Connected"
    st.sidebar.text(f"Tavily Search API: {tavily_status}")
    if not tavily_available:
        st.sidebar.info("Add TAVILY_API_KEY to your .env file")

    # OpenWeather API status
    weather_available = (
        weather_client is not None and weather_client.api_key is not None
    )
    weather_status = "✅ Connected" if weather_available else "❌ Not Connected"
    st.sidebar.text(f"OpenWeather API: {weather_status}")
    if not weather_available:
        st.sidebar.info("Add OPENWEATHER_API_KEY to your .env file")

    # LangSmith API status
    langsmith_available = langsmith_client is not None
    langsmith_status = "✅ Connected" if langsmith_available else "❌ Not Connected"
    st.sidebar.text(f"LangSmith API: {langsmith_status}")
    if not langsmith_available:
        st.sidebar.info("Add LANGCHAIN_API_KEY to your .env file")

    # Pinecone status
    pinecone_available = (
        pinecone_client is not None and pinecone_client.index is not None
    )
    pinecone_status = "✅ Connected" if pinecone_available else "❌ Not Connected"
    st.sidebar.text(f"Pinecone Vector DB: {pinecone_status}")

    # LangSmith integration
    has_langsmith_key = is_langsmith_key_set()
    langsmith_enabled = st.checkbox(
        "Enable LangSmith tracing",
        value=has_langsmith_key,
        help="Record and trace your AI interactions in LangSmith for debugging and analysis",
    )

    if langsmith_enabled:
        if not has_langsmith_key:
            st.warning(
                "LangSmith API key not found. Set LANGCHAIN_API_KEY in your .env file."
            )
            st.info("LangSmith is optional, you can continue without it.")
            langsmith_enabled = False
        elif langsmith_client is None:
            st.warning("LangSmith client initialization failed. Check your API key.")
            langsmith_enabled = False
        else:
            project_name = os.environ.get("LANGCHAIN_PROJECT", "default")
            st.success(f"LangSmith tracing enabled for project: {project_name}")
            st.markdown(
                f"[View traces](https://smith.langchain.com/projects/{project_name})"
            )

    # Pinecone integration
    pinecone_available = (
        pinecone_client is not None and pinecone_client.index is not None
    )
    if pinecone_available:
        st.success(f"Pinecone connected to index: {pinecone_client.index_name}")
    else:
        st.warning("Pinecone connection not available. Vector search will be disabled.")
        st.info("To enable vector search, add your Pinecone API key to the .env file.")

    # Web search option
    tavily_available = tavily_client is not None
    enable_web_search = st.checkbox(
        "Enable web search", value=tavily_available, disabled=not tavily_available
    )
    if not tavily_available:
        st.warning("Tavily API key not found or invalid. Web search is disabled.")
        st.info("Add TAVILY_API_KEY to your .env file")
    elif enable_web_search:
        st.success(
            "Web search is enabled. Your queries will use Tavily to retrieve up-to-date information."
        )

    # Advanced settings
    st.subheader("Advanced Settings")
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1
    )
    max_tokens = st.number_input(
        "Max Tokens", min_value=10, max_value=4096, value=1000, step=10
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Function to format prompt based on model
def format_prompt_for_model(prompt, model_name):
    """Format prompt based on the model's expected format"""
    if "mistral" in model_name.lower():
        # Mistral format
        return f"<s>[INST] {prompt} [/INST]"
    elif "llama" in model_name.lower():
        # Llama format
        return f"<s>[INST] {prompt} [/INST]"
    elif "gemma" in model_name.lower():
        # Gemma format
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"
    else:
        # Default format
        return f"USER: {prompt}\nASSISTANT:"


# Chat input
if prompt := st.chat_input("Ask a question or request an analysis..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the model
    with st.chat_message("assistant"):
        if model_provider == "OpenAI" and not openai.api_key:
            st.error("⚠️ OpenAI API key is not set. Please add it to your .env file.")
            st.info("Set the OPENAI_API_KEY environment variable")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "⚠️ OpenAI API key is not set. Please add it to your .env file.",
                }
            )
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Check if web search is enabled and Tavily client is available
                    web_search_results = None
                    search_metadata = {}
                    search_successful = False

                    if enable_web_search and tavily_client is not None:
                        try:
                            with st.status("Searching the web for information..."):
                                # For real-time data like weather, we need to ensure search is enabled
                                is_realtime_query = any(
                                    keyword in prompt.lower()
                                    for keyword in [
                                        "weather",
                                        "temperature",
                                        "forecast",
                                        "current",
                                        "now",
                                        "today",
                                    ]
                                )

                                # Use advanced search for real-time queries
                                search_depth = (
                                    "advanced" if is_realtime_query else "basic"
                                )
                                max_results = 5 if is_realtime_query else 3

                                if is_realtime_query:
                                    st.info(
                                        "Detected real-time information query. Using advanced search."
                                    )

                                search_result = tavily_client.search(
                                    query=prompt,
                                    search_depth=search_depth,
                                    max_results=max_results,
                                )

                                if (
                                    isinstance(search_result, dict)
                                    and search_result.get("error")
                                    and "401" in str(search_result.get("error"))
                                ):
                                    st.error(
                                        "Tavily API key is invalid or expired. Web search functionality is disabled."
                                    )
                                    st.info(
                                        "To get a valid API key, visit https://tavily.com/"
                                    )
                                    web_search_results = "Note: Web search is currently unavailable. Please provide a valid Tavily API key."
                                elif "error" in search_result:
                                    st.warning(
                                        f"Web search issue: {search_result.get('error')}"
                                    )
                                    web_search_results = f"Note: Web search was attempted but encountered an issue: {search_result.get('error')}"
                                    search_metadata["search_error"] = search_result.get(
                                        "error"
                                    )
                                elif search_result and "results" in search_result:
                                    web_search_results = "Web search results:\n\n"
                                    search_metadata["search_success"] = True
                                    search_metadata["result_count"] = len(
                                        search_result["results"]
                                    )

                                    for i, result in enumerate(
                                        search_result["results"], 1
                                    ):
                                        title = result.get("title", "No title")
                                        content = result.get(
                                            "content", "No content available"
                                        )
                                        url = result.get("url", "")
                                        web_search_results += f"{i}. **{title}**\n{content}\n[Source]({url})\n\n"
                                        # Store source URLs in metadata
                                        search_metadata[f"source_url_{i}"] = url
                                else:
                                    st.info("Web search did not return any results")
                                    web_search_results = "Note: Web search was attempted but did not return any results."
                                    search_metadata["search_no_results"] = True
                        except Exception as e:
                            err_msg = str(e)
                            if "401" in err_msg and "invalid API key" in err_msg:
                                st.error(
                                    "Invalid Tavily API key. Please update your API key in the .env file."
                                )
                            else:
                                st.error(f"Web search error: {err_msg}")
                            web_search_results = (
                                f"Note: Web search failed due to an error: {err_msg}"
                            )
                            search_metadata["search_error"] = err_msg

                    # Check if pinecone is available to provide vector search results
                    vector_search_results = None
                    if (
                        pinecone_available
                        and model_provider == "OpenAI"
                        and openai.api_key
                    ):
                        try:
                            with st.status("Searching vector database..."):
                                # Get embeddings from OpenAI
                                response = openai.embeddings.create(
                                    input=prompt, model="text-embedding-3-small"
                                )
                                embedding = response.data[0].embedding

                                # Query Pinecone
                                results = pinecone_client.query(
                                    vector=embedding, top_k=3, include_metadata=True
                                )

                                if results:
                                    vector_search_results = (
                                        "Vector database results:\n\n"
                                    )
                                    for i, result in enumerate(results, 1):
                                        title = result.get("metadata", {}).get(
                                            "title", "No title"
                                        )
                                        content = result.get("metadata", {}).get(
                                            "content", "No content available"
                                        )
                                        url = result.get("metadata", {}).get("url", "")
                                        score = result.get("score", 0)
                                        vector_search_results += f"{i}. **{title}** (Relevance: {score:.2f})\n{content}\n"
                                        if url:
                                            vector_search_results += (
                                                f"[Source]({url})\n\n"
                                            )
                                        else:
                                            vector_search_results += "\n"
                                else:
                                    vector_search_results = "Note: No relevant information found in the vector database."
                        except Exception as e:
                            st.error(f"Vector search error: {str(e)}")
                            vector_search_results = (
                                f"Note: Vector search failed due to an error: {str(e)}"
                            )

                    # Create the prompt for the LLM
                    messages = st.session_state.messages.copy()

                    # Add web search results to the prompt if available
                    if web_search_results:
                        messages.append(
                            {"role": "system", "content": web_search_results}
                        )

                    # Add vector search results to the prompt if available
                    if vector_search_results:
                        messages.append(
                            {"role": "system", "content": vector_search_results}
                        )

                    # Check if this is a weather query
                    is_weather_query = any(
                        keyword in prompt.lower()
                        for keyword in [
                            "weather",
                            "temperature",
                            "forecast",
                            "humidity",
                            "rain",
                            "sunny",
                            "snow",
                        ]
                    )

                    # Add weather information for weather queries
                    if is_weather_query:
                        # Extract location from prompt
                        location = None
                        location_words = ["in", "at", "for", "of"]
                        words = prompt.split()

                        for i, word in enumerate(words):
                            if word.lower() in location_words and i < len(words) - 1:
                                # Extract what appears to be the location after prepositions
                                location = words[i + 1]
                                # Look for multi-word locations (capitalized words)
                                j = i + 2
                                while j < len(words) and (
                                    words[j][0].isupper()
                                    or words[j].lower() in ["and", "of", "the"]
                                ):
                                    location += " " + words[j]
                                    j += 1
                                break

                        # Try to get weather directly if we have a location and a weather client
                        weather_available = False
                        if location:
                            if weather_client and weather_client.api_key:
                                with st.status(
                                    f"Fetching weather data for {location}..."
                                ):
                                    weather_data = weather_client.get_current_weather(
                                        location
                                    )

                                    if "error" not in weather_data:
                                        weather_available = True
                                        weather_msg = (
                                            weather_client.format_weather_message(
                                                weather_data
                                            )
                                        )
                                        # Add the weather data as a special system message
                                        messages.append(
                                            {
                                                "role": "system",
                                                "content": f"Here is the current weather information:\n\n{weather_msg}\n\nPlease use this information to answer the user's query.",
                                            }
                                        )

                        # Handle case where weather data is not available
                        if not weather_available:
                            # Check if web search provided any results
                            if (
                                not web_search_results
                                or "Note: Web search" in web_search_results
                            ):
                                weather_fallback_msg = f"""
                                The user is asking about weather in {location}, but:
                                1. No direct weather data is available - OpenWeather API key is not set
                                2. Web search is not available or returned no results
                                
                                Please respond with:
                                "I don't have access to current weather data for {location}. To get accurate weather information,
                                I recommend checking a weather service like Weather.com, AccuWeather, or your local weather app.
                                
                                To enable weather data in this app, an administrator can add an OpenWeatherMap API key 
                                (get one at https://openweathermap.org/)."
                                """
                                messages.append(
                                    {"role": "system", "content": weather_fallback_msg}
                                )

                    # Get response from the model based on selected provider
                    if model_provider == "Hugging Face":
                        with st.status(f"Loading {selected_model}..."):
                            # Initialize tokenizer
                            tokenizer = AutoTokenizer.from_pretrained(selected_model)

                            # Initialize model with proper settings
                            model = AutoModelForCausalLM.from_pretrained(
                                selected_model,
                                device_map="auto",
                                torch_dtype="auto",
                                trust_remote_code=True,
                            )

                            # Format the prompt for the model
                            formatted_prompt = format_prompt_for_model(
                                prompt, selected_model
                            )

                            # Create a text generation pipeline
                            text_generator = pipeline(
                                "text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                repetition_penalty=1.1,
                                do_sample=True,
                            )

                            # Generate response
                            result = text_generator(formatted_prompt)
                            assistant_response = (
                                result[0]["generated_text"]
                                .replace(formatted_prompt, "")
                                .strip()
                            )
                    else:
                        # Use OpenAI
                        openai_messages = []
                        for msg in messages:
                            # Convert format if needed
                            openai_messages.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )

                        openai_response = openai.chat.completions.create(
                            model=selected_model,
                            messages=openai_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        assistant_response = openai_response.choices[0].message.content

                    # If debug mode is enabled, show the raw response
                    if debug_mode:
                        st.markdown("### Debug: Raw Response")
                        if model_provider == "Hugging Face":
                            st.json(
                                {
                                    "model": selected_model,
                                    "generated_text": assistant_response,
                                }
                            )
                        else:
                            st.json(openai_response.model_dump())

                    st.markdown(assistant_response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )

                    # Record the interaction in LangSmith if enabled
                    if langsmith_enabled and langsmith_client:
                        metadata = {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "provider": model_provider,
                            **search_metadata,
                        }
                        try:
                            from langsmith import Client

                            langsmith_client = Client()
                            run = langsmith_client.run_create(
                                name=f"{model_provider} Chat",
                                run_type="chain",
                                inputs={"query": prompt},
                                outputs={"response": assistant_response},
                                runtime={
                                    "model_name": selected_model,
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                },
                            )
                            run_id = run.id

                            if debug_mode and run_id:
                                st.info(f"Traced in LangSmith with run ID: {run_id}")
                                st.markdown(
                                    f"[View trace](https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT', 'default')})"
                                )
                        except Exception as e:
                            st.warning(f"Error recording in LangSmith: {str(e)}")

                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    # Add error message to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ {error_message}"}
                    )

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
