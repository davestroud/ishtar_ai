#!/usr/bin/env python3
import streamlit as st
from ollama_client import OllamaClient
import json
import os
from src.tavily_search import TavilySearch
from src.langsmith_integration import get_langsmith_client, trace_ollama_chat
import openai

# Page configuration
st.set_page_config(page_title="Ishtar AI", page_icon="🔍", layout="wide")

# Configure API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")


# Initialize clients
@st.cache_resource
def get_client():
    return OllamaClient()


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
    return get_langsmith_client()


client = get_client()
tavily_client = get_tavily_client()
langsmith_client = get_ls_client()

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
        ["Ollama", "OpenAI"],
        index=0,
        help="Choose which LLM provider to use",
    )

    # Model selection based on provider
    if model_provider == "Ollama":
        try:
            models_response = client.list_models()
            model_names = [
                model.get("name") for model in models_response.get("models", [])
            ]

            if not model_names:
                st.warning("No Ollama models found. Please pull a model first.")
                model_names = ["llama3"]  # Default fallback
        except Exception as e:
            st.error(f"Error fetching Ollama models: {str(e)}")
            model_names = ["llama3"]  # Default fallback

        selected_model = st.selectbox("Select Ollama Model", model_names)
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

    # LangSmith integration
    langsmith_enabled = st.checkbox(
        "Enable LangSmith tracing", value=(langsmith_client is not None)
    )
    if langsmith_enabled and langsmith_client is None:
        st.warning(
            "LangSmith API key not found. Set LANGCHAIN_API_KEY in your .env file."
        )
        st.info("Run: ./update_langsmith_key.sh YOUR_LANGSMITH_API_KEY")
    elif langsmith_enabled:
        project_name = os.environ.get("LANGCHAIN_PROJECT", "default")
        st.success(f"LangSmith tracing enabled for project: {project_name}")
        st.markdown(
            f"[View traces](https://smith.langchain.com/projects/{project_name})"
        )

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

    # Model pulling section (Ollama only)
    if model_provider == "Ollama":
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
if prompt := st.chat_input("Ask a question or request an analysis..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the model
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Check if web search is enabled and Tavily client is available
                web_search_results = None
                search_metadata = {}
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
                                search_metadata["search_error"] = search_result.get(
                                    "error"
                                )
                            elif search_result and "results" in search_result:
                                web_search_results = "Web search results:\n\n"
                                search_metadata["search_success"] = True
                                search_metadata["result_count"] = len(
                                    search_result["results"]
                                )

                                for i, result in enumerate(search_result["results"], 1):
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
                            st.info("You can run: ./update_tavily_key.sh YOUR_API_KEY")
                        else:
                            st.error(f"Web search error: {err_msg}")
                        web_search_results = (
                            f"Note: Web search failed due to an error: {err_msg}"
                        )
                        search_metadata["search_error"] = err_msg

                # Create the prompt for the LLM
                messages = st.session_state.messages.copy()

                # Add web search results to the prompt if available
                if web_search_results:
                    messages.append({"role": "system", "content": web_search_results})

                # Get response from the model based on selected provider
                if model_provider == "Ollama":
                    response = client.chat(
                        messages=messages,
                        model=selected_model,
                        options={"temperature": temperature, "max_tokens": max_tokens},
                    )
                    assistant_response = response.get("message", {}).get("content", "")
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
                    if model_provider == "Ollama":
                        st.json(response)
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
                    run_id = trace_ollama_chat(
                        query=prompt,
                        model_name=selected_model,
                        messages=messages,
                        response=assistant_response,
                        metadata=metadata,
                    )
                    if debug_mode and run_id:
                        st.info(f"Traced in LangSmith with run ID: {run_id}")
                        st.markdown(
                            f"[View trace](https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT', 'default')})"
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
