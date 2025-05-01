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
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
import torch


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
    if "hf_tokenizer" not in clients or not settings.get("model"):
        return "Error: No model available. Please select a valid model and ensure you have a Hugging Face token if needed."

    # Ensure we have a model loaded
    if "hf_model" not in clients or clients["hf_model"] is None:
        # If tokenizer is available, try to load the model now
        if "hf_tokenizer" in clients and clients["hf_tokenizer"] is not None:
            try:
                # Get token for model loading
                hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv(
                    "HUGGING_FACE_TOKEN"
                )
                model_name = settings["model"]
                tokenizer = clients["hf_tokenizer"]

                # Determine if we need disk offloading based on model size
                need_disk_offload = any(
                    name in model_name.lower()
                    for name in ["llama", "mistral-7b", "zephyr-7b"]
                )

                # Choose the appropriate loading method
                if need_disk_offload:
                    # For larger models, use disk offloading
                    logging.info(
                        f"Loading large model {model_name} with disk offloading"
                    )

                    # Create model configuration
                    config = AutoModelForCausalLM.config_class.from_pretrained(
                        model_name, token=hf_token
                    )

                    # Initialize empty model to determine device map
                    with init_empty_weights():
                        empty_model = AutoModelForCausalLM.from_config(config)

                    # Calculate device map
                    device_map = infer_auto_device_map(
                        empty_model,
                        max_memory={0: "2GiB", "cpu": "16GiB", "disk": "50GiB"},
                        no_split_module_classes=[
                            "OPTDecoderLayer",
                            "LlamaDecoderLayer",
                            "MistralDecoderLayer",
                        ],
                    )

                    # Load model with disk offloading
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        offload_folder="offload_folder",
                        offload_state_dict=True,
                        token=hf_token,
                        trust_remote_code=True,
                    )
                else:
                    # For smaller models, load directly to device
                    logging.info(f"Loading smaller model {model_name} directly")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype="auto",
                        trust_remote_code=True,
                        token=hf_token,
                    )

                # Create text generation pipeline
                text_generation = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=settings.get("max_tokens", 1000),
                    temperature=settings.get("temperature", 0.7),
                    do_sample=True,
                )

                # Store in clients
                clients["hf_model"] = text_generation

                logging.info(f"Model {model_name} loaded successfully on demand")
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}"
                logging.error(error_msg)
                return error_msg
        else:
            return "Error: Model not initialized. Please select a model in the sidebar."

    # Prepare system message
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_message = (
        f"You are Ishtar AI, a helpful assistant. Current time: {current_time}."
    )

    # Check if this is a weather query
    weather_keywords = [
        "weather",
        "temperature",
        "forecast",
        "rain",
        "sunny",
        "cloudy",
        "snow",
        "hot",
        "cold",
    ]
    is_weather_query = any(
        keyword in user_input.lower() for keyword in weather_keywords
    )

    # Add web search results if enabled or if this is a weather query
    web_search_results = ""
    if (settings.get("web_search_enabled") or is_weather_query) and "tavily" in clients:
        try:
            with st.spinner("Searching the web for up-to-date information..."):
                # For weather queries, use advanced search with more results
                search_depth = (
                    "advanced"
                    if is_weather_query
                    else settings.get("search_depth", "basic")
                )
                max_results = 5 if is_weather_query else settings.get("max_results", 3)

                logging.info(f"Performing web search for: {user_input}")
                search_results = clients["tavily"].search(
                    query=user_input,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_answer=True,
                )

                # Initialize web_search_results as empty string
                web_search_results = ""

                # Safely extract results - handle both dict and TavilySearchResponse object types
                if search_results:
                    # First try to add the answer if available
                    if hasattr(search_results, "answer") and search_results.answer:
                        web_search_results += (
                            f"Web search answer: {search_results.answer}\n\n"
                        )
                    elif isinstance(search_results, dict) and search_results.get(
                        "answer"
                    ):
                        web_search_results += (
                            f"Web search answer: {search_results.get('answer')}\n\n"
                        )

                    # Then add the results list if available
                    results_list = []
                    if hasattr(search_results, "results"):
                        results_list = search_results.results
                    elif (
                        isinstance(search_results, dict) and "results" in search_results
                    ):
                        results_list = search_results["results"]

                    if results_list:
                        web_search_results += "Web search results:\n\n"
                        for i, result in enumerate(results_list, 1):
                            # Extract title and URL safely
                            title = (
                                result.get("title", "Untitled")
                                if isinstance(result, dict)
                                else getattr(result, "title", "Untitled")
                            )
                            url = (
                                result.get("url", "No URL")
                                if isinstance(result, dict)
                                else getattr(result, "url", "No URL")
                            )
                            content = (
                                result.get("content", "No content")
                                if isinstance(result, dict)
                                else getattr(result, "content", "No content")
                            )

                            web_search_results += f"{i}. {title}\n"
                            web_search_results += f"   URL: {url}\n"
                            web_search_results += f"   {content[:300]}...\n\n"

                        logging.info(f"Found {len(results_list)} search results")
                    else:
                        web_search_results += (
                            "Web search performed but no relevant results found.\n\n"
                        )
                        logging.info("No search results found")
                else:
                    web_search_results = (
                        "Web search performed but no relevant results found.\n\n"
                    )
                    logging.info("No search results found")
        except Exception as e:
            web_search_results = f"Error performing web search: {str(e)}\n\n"
            logging.error(f"Web search error: {str(e)}")

    # Add web search results to system message if available
    if web_search_results:
        system_message += "\n\n" + web_search_results

    # Create the full prompt with system and user messages
    full_prompt = f"{system_message}\n\nUser: {user_input}\n\nAssistant:"

    # Generate response from Hugging Face model
    try:
        if settings.get("debug_mode"):
            st.info(
                f"Using model: {settings['model']} with temperature: {settings['temperature']}"
            )

        with st.spinner("Thinking..."):
            # Trace with LangSmith if enabled
            if settings.get("enable_tracing") and clients.get("langsmith"):
                try:
                    import os
                    from langsmith import Client

                    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv(
                        "LANGSMITH_API_KEY"
                    )

                    if not langsmith_api_key:
                        logging.warning(
                            "LangSmith API key not found but tracing is enabled. Using direct generation."
                        )
                        # Fall back to direct generation without tracing
                        model_name = settings["model"]
                        formatted_prompt = format_prompt_for_model(
                            full_prompt, model_name
                        )
                        result = clients["hf_model"](formatted_prompt)
                        response_text = result[0]["generated_text"]
                        response_text = response_text.replace(
                            formatted_prompt, ""
                        ).strip()
                    else:
                        langsmith_client = Client(api_key=langsmith_api_key)

                        with langsmith_client.tracing(
                            project_name=os.getenv("LANGCHAIN_PROJECT", "ishtar_ai"),
                            name=f"huggingface_chat_{settings['model']}",
                        ) as tracer:
                            tracer.add_input(full_prompt)

                            # Format the prompt for the specific model
                            model_name = settings["model"]
                            formatted_prompt = format_prompt_for_model(
                                full_prompt, model_name
                            )

                            # Generate text
                            result = clients["hf_model"](formatted_prompt)
                            response_text = result[0]["generated_text"]

                            # Extract the assistant's reply from the generated text
                            response_text = response_text.replace(
                                formatted_prompt, ""
                            ).strip()

                            tracer.add_output(response_text)
                except Exception as e:
                    logging.error(f"Error with LangSmith tracing: {str(e)}")

                    # Generate without tracing if there was an error
                    model_name = settings["model"]
                    formatted_prompt = format_prompt_for_model(full_prompt, model_name)
                    result = clients["hf_model"](formatted_prompt)
                    response_text = result[0]["generated_text"]
                    response_text = response_text.replace(formatted_prompt, "").strip()
            else:
                # Generate without tracing
                model_name = settings["model"]
                formatted_prompt = format_prompt_for_model(full_prompt, model_name)
                result = clients["hf_model"](formatted_prompt)
                response_text = result[0]["generated_text"]
                response_text = response_text.replace(formatted_prompt, "").strip()

        return response_text
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

    # Show model status
    model_ready = "hf_model" in clients and clients["hf_model"] is not None
    current_model = settings.get("model", "None")

    if model_ready:
        st.success(f"✅ Model ready: {current_model}")
    else:
        st.warning(
            "⚠️ Model initializing. Please wait a moment before sending your first message."
        )
        # Check if tokenizer is available to load model on demand
        if "hf_tokenizer" in clients and clients["hf_tokenizer"] is not None:
            if st.button("Initialize Model Now"):
                with st.spinner(f"Loading model {current_model}..."):
                    try:
                        from transformers import AutoModelForCausalLM, pipeline
                        import os
                        import torch

                        # Get token for model loading
                        hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv(
                            "HUGGING_FACE_TOKEN"
                        )
                        model_name = settings["model"]
                        tokenizer = clients["hf_tokenizer"]

                        # Determine if we need disk offloading based on model size
                        need_disk_offload = any(
                            name in model_name.lower()
                            for name in ["llama", "mistral-7b", "zephyr-7b"]
                        )

                        # Choose the appropriate loading method
                        if need_disk_offload:
                            # For larger models, use disk offloading
                            logging.info(
                                f"Loading large model {model_name} with disk offloading"
                            )

                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                device_map="auto",
                                torch_dtype=torch.float16,
                                offload_folder="offload_folder",
                                token=hf_token,
                                trust_remote_code=True,
                            )
                        else:
                            # For smaller models, load directly to device
                            logging.info(f"Loading smaller model {model_name} directly")
                            model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                device_map="auto",
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                token=hf_token,
                            )

                        # Create text generation pipeline
                        text_generation = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=settings.get("max_tokens", 1000),
                            temperature=settings.get("temperature", 0.7),
                            do_sample=True,
                        )

                        # Store in clients
                        clients["hf_model"] = text_generation
                        st.success(f"Model {model_name} loaded successfully!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {str(e)}")
                        logging.error(f"Failed to load model: {str(e)}")

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

            # Check if model is ready
            if not model_ready:
                message_placeholder.warning(
                    "Model is not yet initialized. Attempting to load model now..."
                )
                try:
                    from transformers import AutoModelForCausalLM, pipeline
                    import os
                    import torch

                    # Get token for model loading
                    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv(
                        "HUGGING_FACE_TOKEN"
                    )
                    model_name = settings["model"]

                    # Check if we have a tokenizer
                    if "hf_tokenizer" not in clients or clients["hf_tokenizer"] is None:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name, token=hf_token
                        )
                        clients["hf_tokenizer"] = tokenizer
                    else:
                        tokenizer = clients["hf_tokenizer"]

                    # Try to load a smaller model if the selected one is too large
                    fallback_model = model_name
                    if any(
                        name in model_name.lower()
                        for name in ["llama-70b", "llama-4-48"]
                    ):
                        message_placeholder.info(
                            "Selected model is very large. Using TinyLlama as a fallback."
                        )
                        fallback_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                        tokenizer = AutoTokenizer.from_pretrained(
                            fallback_model, token=hf_token
                        )
                        clients["hf_tokenizer"] = tokenizer

                    # Load the model
                    message_placeholder.info(f"Loading model {fallback_model}...")

                    model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
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
                        max_new_tokens=settings.get("max_tokens", 1000),
                        temperature=settings.get("temperature", 0.7),
                        do_sample=True,
                    )

                    # Store in clients
                    clients["hf_model"] = text_generation
                    settings["model"] = fallback_model
                    model_ready = True
                    message_placeholder.success(
                        f"Model {fallback_model} loaded successfully!"
                    )
                except Exception as e:
                    error_msg = f"Failed to initialize model: {str(e)}"
                    message_placeholder.error(error_msg)
                    logging.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    return

            # Initialize response
            full_response = ""
            augmented_prompt = prompt

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
                    except Exception as e:
                        logging.error(f"Error in web search: {str(e)}")
                        status.update(label=f"⚠️ Search error: {str(e)}", state="error")

            # Display typing animation and stream response
            try:
                if model_ready:
                    # Check if tracing is enabled
                    if settings.get("enable_tracing") and "langsmith" in clients:
                        try:
                            from src.langsmith_integration import trace_huggingface_chat

                            # Stream response with tracing
                            for chunk in trace_huggingface_chat(
                                client=clients["hf_model"],
                                prompt=augmented_prompt,
                                model=settings["model"],
                                temperature=settings.get("temperature", 0.7),
                            ):
                                if chunk:
                                    content = chunk.get("generated_text", "")
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
                                clients["hf_model"],
                                augmented_prompt,
                                settings,
                                message_placeholder,
                            )
                    else:
                        # Stream response without tracing
                        full_response = stream_response(
                            clients["hf_model"],
                            augmented_prompt,
                            settings,
                            message_placeholder,
                        )
                else:
                    message_placeholder.error(
                        "No model or Hugging Face client available"
                    )
                    full_response = "I'm sorry, but I can't process your request right now. Please check if the model is loaded."
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
        client: Hugging Face model
        prompt: User prompt (potentially augmented with web search results)
        settings: User settings
        placeholder: Streamlit placeholder for displaying the response

    Returns:
        Full response as a string
    """
    full_response = ""

    try:
        # Check if client is None
        if client is None:
            placeholder.error("Model not initialized. Please select a model first.")
            return "Error: Model not initialized. Please select a model in the sidebar."

        # Format prompt based on model
        model_name = settings["model"]
        formatted_prompt = format_prompt_for_model(prompt, model_name)

        # Generate text
        placeholder.markdown("Generating response...")

        # Generate in a single call - for real streaming, you would need
        # to use a more advanced approach with TextIteratorStreamer
        try:
            result = client(
                formatted_prompt,
                max_new_tokens=settings.get("max_tokens", 1000),
                temperature=settings.get("temperature", 0.7),
                do_sample=True,
                repetition_penalty=1.1,
            )

            # Extract response and remove the prompt
            response_text = result[0]["generated_text"]
            response_text = response_text.replace(formatted_prompt, "").strip()

            # Simulate streaming for UI purposes
            words = response_text.split()
            chunks = [" ".join(words[i : i + 3]) for i in range(0, len(words), 3)]

            for chunk in chunks:
                full_response += chunk + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.05)  # Small delay for smoother appearance

        except TypeError as e:
            logging.error(f"Type error in response streaming: {str(e)}")
            placeholder.error("Error with model generation. Please try another model.")
            return "I encountered an error with this model. Please try selecting a different model from the sidebar."

        return full_response.strip()

    except Exception as e:
        logging.error(f"Error streaming response: {str(e)}")
        placeholder.error(f"Error: {str(e)}")
        return f"I encountered an error while generating a response: {str(e)}"


def format_prompt_for_model(prompt, model_name):
    """Format prompt based on the model's expected format"""
    if "meta-llama/Meta-Llama-3" in model_name or "meta-llama/Llama-3" in model_name:
        # Llama 3 format
        return f"<s>[INST] {prompt} [/INST]"
    elif "meta-llama/Meta-Llama-4" in model_name:
        # Llama 4 format
        return f"<s>[INST] {prompt} [/INST]"
    elif "mistral" in model_name.lower():
        # Mistral format
        return f"<s>[INST] {prompt} [/INST]"
    elif "gemma" in model_name.lower():
        # Gemma format
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"
    else:
        # Default format
        return f"USER: {prompt}\nASSISTANT:"
