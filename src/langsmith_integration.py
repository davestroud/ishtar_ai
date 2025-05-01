#!/usr/bin/env python3
import os
import uuid
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List, Union, Generator

# Load environment variables
load_dotenv()

# Import settings - be careful about circular imports
try:
    from src.config import settings

    TRACING_ENABLED = settings.langsmith_tracing
except (ImportError, AttributeError):
    # Fallback to environment variable if settings import fails
    TRACING_ENABLED = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def get_langsmith_client():
    """Get a LangSmith client using environment variables"""
    if not TRACING_ENABLED:
        print("LangSmith tracing is disabled")
        return None

    # Try multiple keys for backwards compatibility
    api_key = None

    # Try to get from environment variables with different names for backward compatibility
    api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")

    if not api_key:
        print(
            "Warning: No LangSmith API key found in environment variables (checked LANGCHAIN_API_KEY and LANGSMITH_API_KEY)"
        )
        return None

    try:
        from langsmith import Client

        # Explicitly pass the API key to avoid any loading issues
        client = Client(api_key=api_key)
        print(f"LangSmith client initialized with API key (length: {len(api_key)})")
        return client
    except ImportError:
        print("Warning: langsmith package not installed. Run 'pip install langsmith'")
        return None
    except Exception as e:
        print(f"Error initializing LangSmith client: {e}")
        return None


def trace_ollama_chat(
    query: str,
    model_name: str,
    messages: List[Dict[str, str]],
    response: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Trace an Ollama chat interaction in LangSmith

    Args:
        query: The user query
        model_name: The name of the model used
        messages: The conversation messages
        response: The model's response
        metadata: Additional metadata to include

    Returns:
        Optional run ID string
    """
    if not TRACING_ENABLED:
        return None

    client = get_langsmith_client()
    if not client:
        return None

    if metadata is None:
        metadata = {}

    # Add basic metadata
    metadata.update(
        {"model": model_name, "source": "ollama", "application": "ishtar-ai"}
    )

    # Use the project name from settings, env, or default
    try:
        from src.config import settings

        project_name = settings.langchain_project
    except (ImportError, AttributeError):
        # Fall back to environment variable
        project_name = os.environ.get("LANGCHAIN_PROJECT", "default")

    try:
        # Create a run in the existing project
        client.create_run(
            project_name=project_name,
            name=f"Ollama Chat {model_name}",
            run_type="llm",
            inputs={"query": query, "messages": messages},
            outputs={"response": response},
            extra=metadata,
        )

        # Generate a pseudo run ID for UI purposes
        run_id = str(uuid.uuid4())
        return run_id
    except Exception as e:
        print(f"Error tracing Ollama chat: {e}")
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
        # Ensure we have langsmith keys
        if not (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")):
            # Just use the model directly without tracing if not available
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}
            return

        # Import langsmith
        from langsmith import Client

        # Get client
        langsmith_client = Client()

        # Run via LangSmith
        name = kwargs.pop("name", f"Hugging Face Chat {model}")

        with langsmith_client.tracing(
            name=name,
            run_type="llm",
            project_name=os.environ.get("LANGCHAIN_PROJECT", "ishtar-ai"),
            metadata={
                "model": model,
                "source": "huggingface",
                "application": "ishtar-ai",
            },
        ) as tracer:
            # Record the input
            tracer.add_input(prompt)

            # Generate the response
            response = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )

            # Extract text
            generated_text = response[0]["generated_text"]

            # Add the output
            tracer.add_output(generated_text)

            # Return as a single output
            yield {"generated_text": generated_text}

    except Exception as e:
        print(f"Error tracing Hugging Face chat: {e}")

        # Fall back to direct generation
        try:
            result = client(
                prompt, max_new_tokens=max_tokens, temperature=temperature, **kwargs
            )
            yield {"generated_text": result[0]["generated_text"]}
        except Exception as inner_e:
            print(f"Error in fallback generation: {inner_e}")
            yield {"generated_text": f"Error generating response: {str(inner_e)}"}


if __name__ == "__main__":
    # Test the LangSmith integration
    client = get_langsmith_client()
    if client:
        try:
            # Get the project name from environment
            project = os.environ.get("LANGCHAIN_PROJECT", "default")
            print(f"Successfully connected to LangSmith with project: {project}")

            # Test trace function
            print("Testing trace_ollama_chat function...")
            run_id = trace_ollama_chat(
                query="Test query",
                model_name="test-model",
                messages=[{"role": "user", "content": "Test message"}],
                response="Test response",
                metadata={"test": True},
            )

            if run_id:
                print(f"Created test trace (pseudo ID: {run_id})")
                print("Check your LangSmith dashboard to view runs at:")
                print(f"https://smith.langchain.com/projects/{project}")
            else:
                print(
                    "No run ID returned, but check your LangSmith dashboard for the trace"
                )

        except Exception as e:
            print(f"Error testing LangSmith: {e}")
    else:
        print("Failed to connect to LangSmith")
