#!/usr/bin/env python3
import streamlit as st
from ollama_client import OllamaClient
import json
import time
import os
import sys
from tavily_search import TavilySearch

st.set_page_config(page_title="Ollama AI Chat", page_icon="🤖", layout="wide")

# Debug message for Tavily API key
if "TAVILY_API_KEY" in os.environ:
    print(
        f"Tavily API key is set (length: {len(os.environ['TAVILY_API_KEY'])})",
        file=sys.stderr,
    )
else:
    print("Tavily API key is not set in environment variables", file=sys.stderr)


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
            client = TavilySearch(api_key)
            # Test the client with a simple query
            test_result = client.search("test", max_results=1)
            print("Tavily client initialized successfully", file=sys.stderr)
            return client
        except Exception as e:
            print(f"Error initializing Tavily client: {e}", file=sys.stderr)
            return None
    print("No Tavily API key found", file=sys.stderr)
    return None


client = get_client()
tavily_client = get_tavily_client()

# Page title
st.title("🤖 Ollama AI Chat with Web Search")
st.markdown(
    "Interact with Ollama AI models running in Docker. Web search powered by Tavily API."
)

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

    # Model pulling section
    st.subheader("Pull a new model")
    new_model = st.text_input("Model name (e.g., llama3, mistral, gemma:7b)")

    if st.button("Pull Model"):
        with st.spinner(f"Pulling model {new_model}..."):
            try:
                result = client.pull_model(new_model, timeout=180)
                st.success(f"Model {new_model} pulled successfully!")
                # Refresh the page to show the new model
                st.rerun()
            except TimeoutError as e:
                st.error(f"Timeout while pulling model: {str(e)}")
                st.info(
                    "The model may still be downloading in the background. Check Docker logs for progress."
                )
            except Exception as e:
                st.error(f"Error pulling model: {str(e)}")

    # Web search settings
    st.subheader("Web Search")

    # Display Tavily status
    if tavily_client is not None:
        st.success("✅ Tavily API connected and working")
    else:
        st.error("❌ Tavily API not configured properly")

    use_web_search = st.checkbox(
        "Enable Tavily web search", value=tavily_client is not None
    )

    if use_web_search and tavily_client is None:
        st.warning(
            "Tavily API key not found or invalid. Set TAVILY_API_KEY environment variable."
        )
        tavily_api_key = st.text_input("Enter Tavily API key here:", type="password")
        if tavily_api_key:
            try:
                os.environ["TAVILY_API_KEY"] = tavily_api_key
                st.success("API key set! Click the button below to apply.")
                if st.button("Apply API Key"):
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error setting API key: {str(e)}")

    # Search settings
    if use_web_search and tavily_client is not None:
        search_depth = st.radio(
            "Search Quality",
            ["basic", "advanced"],
            index=1,
            help="Basic: faster but less thorough. Advanced: more thorough but slower.",
        )
        max_search_results = st.slider(
            "Max Search Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of web search results to retrieve",
        )

    # Advanced settings
    st.subheader("Advanced Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.2) make it more deterministic",
    )
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=10,
        max_value=8192,
        value=2000,
        step=100,
        help="Maximum number of tokens to generate",
    )

    # Timeout setting
    timeout = st.number_input(
        "Request Timeout (seconds)",
        min_value=10,
        max_value=300,
        value=30,
        step=5,
        help="Maximum time to wait for a response",
    )

    # Use streaming option
    use_streaming = st.checkbox(
        "Use streaming",
        value=True,
        help="Stream responses as they're generated (recommended)",
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle special metadata in messages
        if message.get("metadata") and message["role"] == "assistant":
            # If there are search results in metadata, display them in an expander
            if "search_results" in message["metadata"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["metadata"]["search_results"]):
                        st.markdown(
                            f"**Source {i+1}**: [{source['title']}]({source['url']})"
                        )
                        st.markdown(f"{source['content'][:1000]}...")
                        st.markdown("---")

        # Display the actual message content
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Say something..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get web search results if enabled
    web_context = ""
    search_results = []
    if use_web_search and tavily_client is not None:
        with st.status(
            "🔍 Searching the web for information...", expanded=True
        ) as status:
            try:
                st.write("Querying search engine...")
                search_params = {
                    "query": prompt,
                    "max_results": max_search_results,
                    "include_answer": True,
                    "search_depth": search_depth,
                }

                # Perform the search
                search_data = tavily_client.search(**search_params)

                # Extract search results and answer
                search_results = search_data.get("results", [])
                tavily_answer = search_data.get("answer", "")

                # Format web context for the LLM
                if search_results:
                    st.write(f"Found {len(search_results)} relevant sources")
                    sources_text = ""
                    for i, result in enumerate(search_results):
                        title = result.get("title", "Untitled")
                        content = result.get("content", "")
                        sources_text += f"\nSource {i+1} - {title}:\n{content}\n"

                    web_context = f"Here is information from the web about this topic:\n{sources_text}\n"
                    web_context += f"\nAI-generated summary: {tavily_answer}\n\n"

                    # Display the first few results as a preview - without using expanders
                    st.write("### Search Results Preview:")
                    for i, result in enumerate(search_results[:2]):
                        st.write(f"**Source {i+1}: {result.get('title')}**")
                        st.write(result.get("content", "")[:1000] + "...")
                        st.write(f"[Link]({result.get('url', '')})")
                        st.markdown("---")
                else:
                    st.write("No relevant search results found")
                    web_context = ""

                status.update(
                    label="✅ Web search complete", state="complete", expanded=False
                )
            except Exception as e:
                st.write(f"Search failed: {str(e)}")
                status.update(
                    label=f"❌ Web search failed: {str(e)}",
                    state="error",
                    expanded=True,
                )
                web_context = ""

    # Get response from the model
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.info("Thinking...")

        if use_streaming:
            # Use streaming for more responsive UI
            try:
                # Initialize the session state for display text
                st.session_state.display_text = ""

                # Define a callback function for streaming
                def on_token(token, _):
                    if token:
                        # Update the display text and refresh the display
                        st.session_state.display_text = (
                            st.session_state.display_text + token
                        )
                        message_placeholder.markdown(
                            st.session_state.display_text + "▌"
                        )

                # Prepare prompt with web search results if available
                if web_context:
                    augmented_prompt = f"""Please answer the following question based on the web search results provided. 
If the search results don't contain relevant information, just say you don't know.

WEB SEARCH RESULTS:
{web_context}

USER QUESTION: {prompt}

Your answer should be comprehensive and based on the search results:
"""
                else:
                    augmented_prompt = prompt

                # Use the streaming API with the enhanced prompt
                full_response = client.stream_generate(
                    prompt=augmented_prompt,
                    model=selected_model,
                    callback=on_token,
                    options={"temperature": temperature, "max_tokens": max_tokens},
                    timeout=timeout,
                )

                # Show final response without cursor
                message_placeholder.markdown(full_response)

                # Add to chat history with metadata about search results if available
                assistant_message = {"role": "assistant", "content": full_response}

                # Add metadata if we have search results
                if search_results:
                    assistant_message["metadata"] = {
                        "search_results": [
                            {
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "content": result.get("content", "")[
                                    :2000
                                ],  # Increased storage limit
                            }
                            for result in search_results
                        ]
                    }

                st.session_state.messages.append(assistant_message)

            except TimeoutError as e:
                message_placeholder.error(
                    f"Request timed out after {timeout} seconds. Try a shorter question or reduce the max tokens."
                )
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")
        else:
            # Use regular non-streaming API
            try:
                # Prepare prompt with web search results if available
                if web_context:
                    augmented_prompt = f"""Please answer the following question based on the web search results provided. 
If the search results don't contain relevant information, just say you don't know.

WEB SEARCH RESULTS:
{web_context}

USER QUESTION: {prompt}

Your answer should be comprehensive and based on the search results:
"""
                    response = client.generate(
                        prompt=augmented_prompt,
                        model=selected_model,
                        options={"temperature": temperature, "max_tokens": max_tokens},
                        timeout=timeout,
                    )
                    assistant_response = response.get("response", "")
                else:
                    # Regular chat without web context
                    response = client.chat(
                        messages=st.session_state.messages,
                        model=selected_model,
                        options={"temperature": temperature, "max_tokens": max_tokens},
                        timeout=timeout,
                    )
                    assistant_response = response.get("message", {}).get("content", "")

                if not assistant_response:
                    assistant_response = "I couldn't generate a proper response. Please try again with a different prompt."

                message_placeholder.markdown(assistant_response)

                # Add to chat history with metadata about search results if available
                assistant_message = {"role": "assistant", "content": assistant_response}

                # Add metadata if we have search results
                if search_results:
                    assistant_message["metadata"] = {
                        "search_results": [
                            {
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "content": result.get("content", "")[
                                    :2000
                                ],  # Increased storage limit
                            }
                            for result in search_results
                        ]
                    }

                st.session_state.messages.append(assistant_message)

            except TimeoutError as e:
                message_placeholder.error(
                    f"Request timed out after {timeout} seconds. Try a shorter question or reduce the max tokens."
                )
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")

# Add a clear chat button and download button
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("Test Web Search", use_container_width=True):
        if tavily_client is not None:
            try:
                with st.spinner("Testing Tavily API..."):
                    result = tavily_client.search("Test query", max_results=1)
                    st.success("✅ Tavily API is working correctly!")
                    st.json(result)
            except Exception as e:
                st.error(f"❌ Tavily API test failed: {str(e)}")
        else:
            st.error("❌ Tavily client not initialized. Check your API key.")

# Information section at the bottom
with st.expander("ℹ️ Help & Information"):
    st.markdown(
        """
    ### Tips for better results:
    - **Keep your questions clear and concise**
    - If responses are slow, try reducing the max tokens or using a smaller model
    - Use a lower temperature (0.1-0.5) for more factual, deterministic responses
    - Use a higher temperature (0.7-1.0) for more creative, varied responses
    - Enable web search for up-to-date information
    
    ### Troubleshooting Web Search:
    - Make sure your Tavily API key is valid and has sufficient credits
    - If no results appear, try a more specific query
    - Check the console logs for detailed error messages
    
    ### General Troubleshooting:
    - If you get JSON errors, try using streaming mode (checkbox in settings)
    - If requests time out, try reducing the complexity of your queries
    - Make sure the Docker container is running: `docker ps | grep ollama`
    """
    )
