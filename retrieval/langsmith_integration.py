#!/usr/bin/env python3
import os
import uuid
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Union, Generator
import logging

# Load environment variables with higher priority
load_dotenv(override=True)

# Configure logging
logger = logging.getLogger(__name__)


def get_langsmith_client():
    """Get a LangSmith client using environment variables"""
    # Try multiple keys for backwards compatibility
    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")

    if not api_key:
        logger.warning("No LangSmith API key found in environment variables")
        os.environ["LANGSMITH_TRACING"] = "false"
        return None

    # Get endpoint if available
    endpoint = os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get(
        "LANGCHAIN_ENDPOINT"
    )

    # Get project name
    project_name = os.environ.get("LANGCHAIN_PROJECT") or "ishtar_ai"

    # Set tracing environment variables
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name

    # Ensure API key is available in both formats for compatibility
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key

    try:
        from langsmith import Client

        # Log what we're using
        logger.info(f"LangSmith API Key found (length: {len(api_key)})")
        if endpoint:
            logger.info(f"Using custom LangSmith endpoint: {endpoint}")
        logger.info(f"Using project: {project_name}")

        # Initialize with proper parameters
        client_args = {"api_key": api_key}

        # Add endpoint if available
        if endpoint:
            client_args["api_url"] = endpoint

        # Create client
        client = Client(**client_args)

        # Verify connection by checking if project exists or creating it
        try:
            # Try to get or create the project
            projects = client.list_projects()
            project_exists = any(p.name == project_name for p in projects)

            if not project_exists:
                logger.info(f"Creating new LangSmith project: {project_name}")
                client.create_project(project_name=project_name)
            else:
                logger.info(f"Using existing LangSmith project: {project_name}")

            logger.info("LangSmith client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Error verifying LangSmith connection: {e}")
            os.environ["LANGSMITH_TRACING"] = "false"
            return None

    except ImportError:
        logger.warning("langsmith package not installed. Run 'pip install langsmith'")
        os.environ["LANGSMITH_TRACING"] = "false"
        return None
    except Exception as e:
        logger.error(f"Error initializing LangSmith client: {e}")
        os.environ["LANGSMITH_TRACING"] = "false"
        return None


def trace_huggingface_chat(
    client,
    prompt: str,
    model: str = "unknown",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
) -> dict:
    """
    Trace a Hugging Face chat interaction in LangSmith

    Args:
        client: The Hugging Face model or pipeline
        prompt: The input prompt for the model
        model: The model name
        temperature: The temperature setting for generation
        max_tokens: The maximum tokens to generate

    Returns:
        A dictionary containing the generated text
    """
    try:
        # Get API key and project name
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv(
            "LANGSMITH_API_KEY"
        )
        langsmith_tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        project_name = os.getenv("LANGCHAIN_PROJECT", "ishtar_ai")

        # Check if tracing is enabled and API key exists
        if not langsmith_tracing or not langchain_api_key:
            logger.info(
                "LangSmith tracing is disabled or no API key. Using direct generation."
            )
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            generated_text = result[0]["generated_text"]

            # Remove the prompt from the response if it's included
            if prompt in generated_text:
                clean_output = generated_text.replace(prompt, "", 1).strip()
            else:
                clean_output = generated_text

            yield {"generated_text": generated_text}
            return

        # Import langsmith with proper error handling
        try:
            from langsmith import Client
            from langsmith.run_trees import RunTree

            # Get endpoint if available
            endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv(
                "LANGCHAIN_ENDPOINT"
            )

            # Initialize client arguments
            client_args = {"api_key": langchain_api_key}
            if endpoint:
                client_args["api_url"] = endpoint

            # Create client with proper configuration
            langsmith_client = Client(**client_args)
            logger.info(
                f"Tracing HuggingFace chat with model {model} to project {project_name}"
            )

            # First, generate the response so we can stream it to the user
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            generated_text = result[0]["generated_text"]

            # Remove prompt from the output for cleaner logging
            if prompt in generated_text:
                clean_output = generated_text.replace(prompt, "", 1).strip()
            else:
                clean_output = generated_text

            # Create a run_id for tracking
            run_id = str(uuid.uuid4())

            # Post the run to LangSmith
            run_id = langsmith_client.create_run(
                project_name=project_name,
                name=f"Hugging Face Chat {model}",
                run_type="llm",
                inputs={"prompt": prompt},
                outputs={"response": clean_output},
                extra={
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                },
                run_id=run_id,
            )

            logger.info(
                f"Successfully traced HuggingFace chat to LangSmith project: {project_name} (Run ID: {run_id})"
            )

            # Return the result
            yield {"generated_text": generated_text}

        except ImportError as e:
            logger.error(f"Failed to import LangSmith: {e}")
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}

    except Exception as e:
        logger.error(f"Error tracing Hugging Face chat: {e}")
        try:
            # Fallback to direct generation if tracing fails
            logger.info("Falling back to direct generation without tracing")
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}
        except Exception as inner_e:
            logger.error(f"Error in fallback generation: {inner_e}")
            yield {"generated_text": f"Error generating response: {str(inner_e)}"}


if __name__ == "__main__":
    # Test the LangSmith integration
    client = get_langsmith_client()
    if client:
        try:
            # Get the project name from environment
            project = os.environ.get("LANGCHAIN_PROJECT", "default")
            print(f"Successfully connected to LangSmith with project: {project}")

            print("Check your LangSmith dashboard to view runs at:")
            print(f"https://smith.langchain.com/projects/{project}")
        except Exception as e:
            print(f"Error testing LangSmith: {e}")
    else:
        print("Failed to connect to LangSmith")
