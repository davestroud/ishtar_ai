#!/usr/bin/env python3
import streamlit as st
from ollama_client import OllamaClient
import json
import os
from src.tavily_search import TavilySearch

st.set_page_config(page_title="Ollama AI Chat", page_icon="🤖", layout="wide")


# Initialize Ollama client
@st.cache_resource
def get_client():
    return OllamaClient()


# Initialize Tavily client if API key is available
@st.cache_resource
def get_tavily_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if api_key:
        try:
            return TavilySearch(api_key)
        except Exception as e:
            st.sidebar.error(f"Error initializing Tavily: {str(e)}")
    return None


client = get_client()
tavily_client = get_tavily_client()

# Page title
st.title("🤖 Ollama AI Chat")
st.markdown("Interact with Ollama AI models running in Docker")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("Settings")

    # Try to get available models
    try:
        models_response = client.list_models()
        model_names = [model.get("name") for model in models_response.get("models", [])]

        if not model_names:
            st.warning("No models found. Please pull a model first.")
            model_names = ["llama3"]  # Default fallback
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        model_names = ["llama3"]  # Default fallback

    selected_model = st.selectbox("Select Model", model_names)

    # Debug mode
    debug_mode = st.checkbox("Enable debug mode", value=False)

    # Web search option
    tavily_available = tavily_client is not None
    enable_web_search = st.checkbox(
        "Enable web search", value=tavily_available, disabled=not tavily_available
    )
    if not tavily_available:
        st.warning(
            "Tavily API key not found or invalid. Set a valid TAVILY_API_KEY in your .env file."
        )
        st.info("Run: ./update_tavily_key.sh YOUR_TAVILY_API_KEY")
    elif enable_web_search:
        st.success(
            "Web search is enabled. Your queries will use Tavily to retrieve up-to-date information."
        )

    # Model pulling section
    st.subheader("Pull a new model")
    new_model = st.text_input("Model name (e.g., llama3, mistral, gemma:7b)")

    if st.button("Pull Model"):
        with st.spinner(f"Pulling model {new_model}..."):
            try:
                result = client.pull_model(new_model)
                st.success(f"Model pulled successfully: {new_model}")
            except Exception as e:
                st.error(f"Error pulling model: {str(e)}")

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

# Chat input
if prompt := st.chat_input("Say something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if web search is enabled and Tavily client is available
                web_search_results = None
                if enable_web_search and tavily_client is not None:
                    try:
                        with st.status("Searching the web for information..."):
                            search_result = tavily_client.search(
                                query=prompt, search_depth="advanced", max_results=3
                            )
                            if "error" in search_result:
                                st.warning(
                                    f"Web search issue: {search_result.get('error')}"
                                )
                                web_search_results = f"Note: Web search was attempted but encountered an issue: {search_result.get('error')}"
                            elif search_result and "results" in search_result:
                                web_search_results = "Web search results:\n\n"
                                for i, result in enumerate(search_result["results"], 1):
                                    title = result.get("title", "No title")
                                    content = result.get(
                                        "content", "No content available"
                                    )
                                    url = result.get("url", "")
                                    web_search_results += f"{i}. **{title}**\n{content}\n[Source]({url})\n\n"
                            else:
                                st.info("Web search did not return any results")
                                web_search_results = "Note: Web search was attempted but did not return any results."
                    except Exception as e:
                        err_msg = str(e)
                        if "401" in err_msg and "invalid API key" in err_msg:
                            st.error(
                                "Invalid Tavily API key. Please update your API key in the .env file."
                            )
                            st.info("You can run: ./update_tavily_key.sh YOUR_API_KEY")
                        else:
                            st.error(f"Web search error: {err_msg}")
                        web_search_results = (
                            f"Note: Web search failed due to an error: {err_msg}"
                        )

                # Create the prompt for the LLM
                messages = st.session_state.messages.copy()

                # Add web search results to the prompt if available
                if web_search_results:
                    messages.append({"role": "system", "content": web_search_results})

                # Get response from the model
                response = client.chat(
                    messages=messages,
                    model=selected_model,
                    options={"temperature": temperature, "max_tokens": max_tokens},
                )

                # If debug mode is enabled, show the raw response
                if debug_mode:
                    st.markdown("### Debug: Raw Response")
                    st.json(response)

                assistant_response = response.get("message", {}).get("content", "")

                # In debug mode, also show additional information about the response
                if debug_mode:
                    st.markdown("### Response Keys")
                    st.text(f"Keys in response: {', '.join(response.keys())}")
                    if "message" in response:
                        st.text(
                            f"Keys in message: {', '.join(response.get('message', {}).keys())}"
                        )

                st.markdown(assistant_response)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
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
    st.experimental_rerun()
