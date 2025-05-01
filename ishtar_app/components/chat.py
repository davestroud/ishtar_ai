#!/usr/bin/env python3
import streamlit as st
import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
from threading import Thread


def init_chat_state():
    """Initialize the chat state in the Streamlit session"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    if "response_tokens" not in st.session_state:
        st.session_state.response_tokens = 0


def display_chat_history():
    """Display the chat history in the Streamlit UI"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_user_message(message: str):
    """Add a user message to the chat history"""
    if message:
        st.session_state.messages.append({"role": "user", "content": message})
        with st.chat_message("user"):
            st.markdown(message)


def add_assistant_message(message: str):
    """Add an assistant message to the chat history"""
    if message:
        st.session_state.messages.append({"role": "assistant", "content": message})
        with st.chat_message("assistant"):
            st.markdown(message)


def get_chat_history() -> List[Dict[str, str]]:
    """Get the chat history in the format expected by the LLM"""
    return st.session_state.messages


def format_search_results(search_results: List[Dict]) -> str:
    """Format search results to be included in the LLM prompt"""
    if not search_results:
        return ""

    formatted_results = "Here is some additional context from the web:\n\n"
    for i, result in enumerate(search_results, 1):
        formatted_results += f"SOURCE {i}: {result.get('url', 'No URL')}\n"
        formatted_results += f"TITLE: {result.get('title', 'No Title')}\n"
        formatted_results += f"CONTENT: {result.get('content', 'No Content')}\n\n"

    formatted_results += (
        "Please use this information to help answer the user's question.\n"
    )
    return formatted_results


def get_chat_response(
    clients: Dict[str, Any], user_input: str, settings: Dict[str, Any]
) -> str:
    """
    Get a response from the LLM based on user input and current settings

    Args:
        clients: Dictionary containing API clients
        user_input: The user's message
        settings: Dictionary containing user settings

    Returns:
        The model's response as a string
    """
    if "ollama" not in clients or not settings.get("model"):
        return "Error: LLM service is not available. Please check your configuration."

    # Prepare system message
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_message = (
        f"You are Ishtar AI, a helpful assistant. Current time: {current_time}."
    )

    # Add web search results if enabled
    web_search_results = ""
    if settings.get("enable_web_search") and "tavily" in clients:
        try:
            with st.spinner("Searching the web..."):
                logging.info(f"Performing web search for: {user_input}")
                search_results = clients["tavily"].search(
                    query=user_input,
                    search_depth=settings.get("search_depth", "basic"),
                    max_results=settings.get("max_results", 3),
                )

                if search_results and "results" in search_results:
                    web_search_results = "Web search results:\n\n"
                    for i, result in enumerate(search_results["results"], 1):
                        web_search_results += (
                            f"{i}. {result.get('title', 'Untitled')}\n"
                        )
                        web_search_results += f"   URL: {result.get('url', 'No URL')}\n"
                        web_search_results += (
                            f"   {result.get('content', 'No content')[:300]}...\n\n"
                        )

                    logging.info(
                        f"Found {len(search_results['results'])} search results"
                    )
                else:
                    web_search_results = (
                        "Web search performed but no relevant results found.\n\n"
                    )
                    logging.info("No search results found")
        except Exception as e:
            web_search_results = f"Error performing web search: {str(e)}\n\n"
            logging.error(f"Web search error: {str(e)}")

    if web_search_results:
        system_message += "\n\n" + web_search_results

    # Create messages array
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input},
    ]

    # Generate response from Ollama
    try:
        if settings.get("debug_mode"):
            st.info(
                f"Using model: {settings['model']} with temperature: {settings['temperature']}"
            )

        with st.spinner("Thinking..."):
            # Trace with LangSmith if enabled
            if settings.get("enable_tracing") and "langsmith" in clients:
                try:
                    from langsmith import trace

                    with trace(
                        name="ollama_chat",
                        run_type="llm",
                        project_name="ishtar_ai",
                    ) as span:
                        span.add_input(messages)
                        response = clients["ollama"].chat(
                            model=settings["model"],
                            messages=messages,
                            options={
                                "temperature": settings["temperature"],
                                "num_predict": settings["max_tokens"],
                            },
                        )
                        span.add_output(response)
                except Exception as e:
                    logging.error(f"Error with LangSmith tracing: {str(e)}")
                    response = clients["ollama"].chat(
                        model=settings["model"],
                        messages=messages,
                        options={
                            "temperature": settings["temperature"],
                            "num_predict": settings["max_tokens"],
                        },
                    )
            else:
                response = clients["ollama"].chat(
                    model=settings["model"],
                    messages=messages,
                    options={
                        "temperature": settings["temperature"],
                        "num_predict": settings["max_tokens"],
                    },
                )

        return response["message"]["content"]
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logging.error(error_msg)
        return error_msg


def render_chat_ui(clients: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """
    Render the chat interface and handle message interactions

    Args:
        clients: Dictionary of API clients
        settings: User settings from sidebar
    """
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat header
    st.title("💬 Chat with Ishtar AI")

    # Display welcome message if chat is empty
    if not st.session_state.messages:
        st.info(
            "👋 Welcome to Ishtar AI! Ask me anything or try one of these examples:"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📰 Summarize the latest news"):
                prompt = "Summarize the latest world news headlines"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()
        with col2:
            if st.button("💡 Generate a creative idea"):
                prompt = "Give me a creative project idea using AI and Python"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.experimental_rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new messages
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # If web search is enabled and Tavily client exists
            if settings.get("web_search_enabled") and "tavily" in clients:
                with st.status(
                    "🔍 Searching the web for information...", expanded=False
                ) as status:
                    try:
                        search_results = clients["tavily"].search(
                            query=prompt,
                            search_depth=settings.get("search_depth", "basic"),
                            max_results=settings.get("max_results", 3),
                        )

                        if search_results and "results" in search_results:
                            num_results = len(search_results["results"])
                            status.update(
                                label=f"🔍 Found {num_results} relevant results",
                                state="complete",
                            )

                            # Show search results if debug mode is on
                            if settings.get("debug_mode"):
                                with st.expander("Search Results"):
                                    st.json(search_results)

                            # Build context from search results
                            context = "\n\n".join(
                                [
                                    f"Source: {result['url']}\n{result['content']}"
                                    for result in search_results["results"]
                                ]
                            )

                            # Prepend context to prompt for the model
                            augmented_prompt = (
                                "Based on the following information, please answer the user's question.\n\n"
                                f"Information:\n{context}\n\n"
                                f"User's question: {prompt}\n\n"
                                "If the information provided doesn't answer the question, please say so."
                            )
                        else:
                            status.update(
                                label="No search results found", state="complete"
                            )
                            augmented_prompt = prompt
                    except Exception as e:
                        logging.error(f"Error in web search: {str(e)}")
                        status.update(label=f"⚠️ Search error: {str(e)}", state="error")
                        augmented_prompt = prompt
            else:
                augmented_prompt = prompt

            # Display typing animation and stream response
            try:
                if "ollama" in clients and settings.get("model"):
                    # Check if tracing is enabled
                    if settings.get("enable_tracing") and "langsmith" in clients:
                        try:
                            from src.langsmith_integration import trace_ollama_chat

                            # Stream response with tracing
                            for chunk in trace_ollama_chat(
                                client=clients["ollama"],
                                messages=[
                                    {"role": "user", "content": augmented_prompt}
                                ],
                                model=settings["model"],
                                temperature=settings.get("temperature", 0.7),
                                query=prompt,
                            ):
                                if chunk:
                                    content = chunk.get("message", {}).get(
                                        "content", ""
                                    )
                                    full_response += content
                                    message_placeholder.markdown(full_response + "▌")
                                    time.sleep(0.01)
                        except Exception as e:
                            logging.error(f"Error with LangSmith tracing: {str(e)}")
                            st.error(
                                "LangSmith tracing failed. Continuing without tracing."
                            )
                            # Fall back to regular streaming
                            full_response = stream_response(
                                clients["ollama"],
                                augmented_prompt,
                                settings,
                                message_placeholder,
                            )
                    else:
                        # Stream response without tracing
                        full_response = stream_response(
                            clients["ollama"],
                            augmented_prompt,
                            settings,
                            message_placeholder,
                        )
                else:
                    message_placeholder.error("No model or Ollama client available")
                    full_response = "I'm sorry, but I can't process your request right now. Please check if Ollama is running and a model is selected."
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logging.error(error_msg)
                message_placeholder.error(error_msg)
                full_response = f"I encountered an error: {str(e)}"

            # Update placeholder with final response
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def stream_response(client, prompt, settings, placeholder) -> str:
    """
    Stream the model's response to the UI

    Args:
        client: Dictionary containing Hugging Face model and tokenizer
        prompt: User prompt (potentially augmented with web search results)
        settings: User settings
        placeholder: Streamlit placeholder for displaying the response

    Returns:
        Full response as a string
    """
    full_response = ""

    try:
        # Load model on first use if needed
        model_name = settings["model"]

        # Initialize model if not already loaded
        if "hf_model" not in client or client["hf_model"] is None:
            with st.status(f"Loading {model_name}...", expanded=True):
                # Check if we have a tokenizer
                if "hf_tokenizer" not in client:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    client["hf_tokenizer"] = tokenizer
                else:
                    tokenizer = client["hf_tokenizer"]

                # Load the model with proper settings
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )

                # Create text generation pipeline
                client["hf_model"] = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=settings.get("max_tokens", 1000),
                    temperature=settings.get("temperature", 0.7),
                    repetition_penalty=1.1,
                    do_sample=True,
                )

        # Create prompt with chat format
        tokenizer = client["hf_tokenizer"]
        chat_prompt = format_prompt_for_model(prompt, model_name, tokenizer)

        # Generate text with streaming
        placeholder.markdown("Generating response...")
        streamer = TextIteratorStreamer(tokenizer)

        # Run generation in a separate thread to enable streaming
        generation_kwargs = {
            "input_ids": tokenizer.encode(chat_prompt, return_tensors="pt").to(
                model.device
            ),
            "streamer": streamer,
            "max_new_tokens": settings.get("max_tokens", 1000),
            "temperature": settings.get("temperature", 0.7),
            "repetition_penalty": 1.1,
            "do_sample": True,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the output
        for new_text in streamer:
            full_response += new_text
            placeholder.markdown(full_response + "▌")

        thread.join()

        # Finalize response
        return full_response.strip()

    except Exception as e:
        logging.error(f"Error streaming response: {str(e)}")
        placeholder.error(f"Error: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"


def format_prompt_for_model(prompt, model_name, tokenizer) -> str:
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
