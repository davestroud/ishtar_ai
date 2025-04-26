#!/usr/bin/env python3
import streamlit as st
from ollama_client import OllamaClient
import json

st.set_page_config(page_title="Ollama AI Chat", page_icon="🤖", layout="wide")


# Initialize Ollama client
@st.cache_resource
def get_client():
    return OllamaClient()


client = get_client()

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
                response = client.chat(
                    messages=st.session_state.messages,
                    model=selected_model,
                    options={"temperature": temperature, "max_tokens": max_tokens},
                )

                assistant_response = response.get("message", {}).get("content", "")
                st.markdown(assistant_response)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()
