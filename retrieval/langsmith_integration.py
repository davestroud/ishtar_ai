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

# Import settings - be careful about circular imports
try:
    from retrieval.config import settings

    TRACING_ENABLED = settings.langsmith_tracing
except (ImportError, AttributeError):
    # Fallback to environment variable if settings import fails
    TRACING_ENABLED = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def get_langsmith_client():
    """Get a LangSmith client using environment variables"""
    if not TRACING_ENABLED:
        logger.info("LangSmith tracing is disabled")
        return None

    # Try multiple keys for backwards compatibility
    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")

    # Get endpoint if available
    endpoint = os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get(
        "LANGCHAIN_ENDPOINT"
    )

    # Get project name
    project_name = os.environ.get("LANGCHAIN_PROJECT") or "ishtar_ai"

    if not api_key:
        logger.warning("No LangSmith API key found in environment variables")
        return None

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

            # Set as environment variable for other components
            os.environ["LANGCHAIN_PROJECT"] = project_name
            os.environ["LANGSMITH_TRACING"] = "true"

            return client
        except Exception as e:
            logger.error(f"Error verifying LangSmith connection: {e}")
            return None

    except ImportError:
        logger.warning("langsmith package not installed. Run 'pip install langsmith'")
        return None
    except Exception as e:
        logger.error(f"Error initializing LangSmith client: {e}")
        return None


def trace_huggingface_chat(
    client,
    prompt: str,
    model: str = "unknown",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs,
) -> Generator:
    """
    Trace a Hugging Face chat interaction in LangSmith

    Args:
        client: The Hugging Face model or pipeline
        prompt: The input prompt for the model
        model: The model name
        temperature: The temperature setting for generation
        max_tokens: The maximum tokens to generate

    Returns:
        Generator for the streamed response
    """
    try:
        # Get API key and project name
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv(
            "LANGSMITH_API_KEY"
        )
        project_name = os.getenv("LANGCHAIN_PROJECT", "ishtar_ai")

        # Check if tracing is enabled
        if (
            not langchain_api_key
            or os.getenv("LANGSMITH_TRACING", "").lower() != "true"
        ):
            logger.warning(
                "LangSmith tracing disabled or API key missing. Using direct generation."
            )
            # Just use the model directly without tracing
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}
            return

        # Import langsmith with proper error handling
        try:
            from langsmith import Client
        except ImportError:
            logger.error("Failed to import LangSmith. Using direct generation.")
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}
            return

        # Get endpoint if available
        endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")

        # Initialize client arguments
        client_args = {"api_key": langchain_api_key}
        if endpoint:
            client_args["api_url"] = endpoint

        # Create client with proper configuration
        try:
            langsmith_client = Client(**client_args)
            logger.info(
                f"Tracing HuggingFace chat with model {model} to project {project_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create LangSmith client: {e}")
            # Fall back to direct generation
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}
            return

        # Create metadata for better context
        metadata = {
            "model": model,
            "source": "huggingface",
            "application": "ishtar_ai",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        metadata.update(kwargs.pop("metadata", {}))

        # Use the create_run method directly instead of run_and_record
        try:
            # Run the model to get the response
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )

            # Extract text
            generated_text = result[0]["generated_text"]

            # Remove prompt from the output for cleaner logging
            if prompt in generated_text:
                clean_output = generated_text.replace(prompt, "").strip()
            else:
                clean_output = generated_text

            # Create a run_id for tracking
            run_id = str(uuid.uuid4())

            # Record the run in LangSmith using create_run instead of run_and_record
            langsmith_client.create_run(
                project_name=project_name,
                name=f"Hugging Face Chat {model}",
                run_type="llm",
                inputs={"prompt": prompt},
                outputs={"response": clean_output},
                extra=metadata,
                run_id=run_id,
            )

            logger.info(
                f"Successfully traced HuggingFace chat to LangSmith project: {project_name} (Run ID: {run_id})"
            )

            # Return the generated text
            yield {"generated_text": generated_text}

        except Exception as e:
            logger.error(f"Error tracing to LangSmith: {e}")
            # We already have the result, just return it
            yield {"generated_text": generated_text}

    except Exception as e:
        logger.error(f"Error tracing Hugging Face chat: {e}")

        # Fall back to direct generation
        try:
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
