#!/usr/bin/env python3
import os
import uuid
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()


def get_langsmith_client():
    """Get a LangSmith client using environment variables"""
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        print("Warning: LANGCHAIN_API_KEY environment variable not set")
        return None

    try:
        client = Client()
        return client
    except Exception as e:
        print(f"Error initializing LangSmith client: {e}")
        return None


def trace_ollama_chat(query, model_name, messages, response, metadata=None):
    """
    Trace an Ollama chat interaction in LangSmith

    Args:
        query: The user query
        model_name: The name of the model used
        messages: The conversation messages
        response: The model's response
        metadata: Additional metadata to include
    """
    client = get_langsmith_client()
    if not client:
        return None

    if metadata is None:
        metadata = {}

    # Add basic metadata
    metadata.update(
        {"model": model_name, "source": "ollama", "application": "ishtar-ai"}
    )

    # Use the project name from env or default to default
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
