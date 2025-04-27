#!/usr/bin/env python3
import requests
import json
import os
import sys
from dotenv import load_dotenv
from config import get_client_config

# Load environment variables
load_dotenv()


class OllamaClient:
    def __init__(self, base_url=None):
        """Initialize the Ollama client with configuration from environment variables."""
        config = get_client_config()
        self.base_url = base_url or config["base_url"]
        self.api_url = f"{self.base_url}/api"
        self.default_model = config["default_model"]
        self.default_params = config["default_params"]

    def list_models(self):
        """List all available models"""
        response = requests.get(f"{self.api_url}/tags")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to list models: {response.text}")

    def generate(self, prompt, model=None, options=None):
        """Generate a response using the specified model"""
        model = model or self.default_model
        options = options or {}

        # Merge with default params
        merged_options = {**self.default_params, **options}
        payload = {"model": model, "prompt": prompt, **merged_options}

        response = requests.post(f"{self.api_url}/generate", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to generate: {response.text}")

    def chat(self, messages, model=None, options=None):
        """Chat with the model using a conversation format"""
        model = model or self.default_model
        options = options or {}

        # Merge with default params
        merged_options = {**self.default_params, **options}

        # Use streaming approach to avoid JSON parsing issues
        merged_options["stream"] = True

        payload = {"model": model, "messages": messages, **merged_options}

        print(
            f"Making chat request to {self.api_url}/chat with model {model}",
            file=sys.stderr,
        )

        try:
            response = requests.post(f"{self.api_url}/chat", json=payload, stream=True)

            if response.status_code != 200:
                error_msg = f"Failed to chat: {response.text}"
                print(error_msg, file=sys.stderr)
                return {
                    "message": {"role": "assistant", "content": f"Error: {error_msg}"}
                }

            # Process the streaming response line by line
            full_content = ""
            message_role = "assistant"

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    # Decode the line and parse it as JSON
                    line_str = line.decode("utf-8")
                    chunk = json.loads(line_str)

                    # Extract content from the chunk
                    if "message" in chunk and "content" in chunk["message"]:
                        content_chunk = chunk["message"]["content"]
                        full_content += content_chunk

                        # Update role if available
                        if "role" in chunk["message"]:
                            message_role = chunk["message"]["role"]

                except json.JSONDecodeError as e:
                    print(f"Error parsing chunk: {e}", file=sys.stderr)
                    # Try to extract any text content from the line
                    import re

                    text_matches = re.findall(r'"content"\s*:\s*"([^"]*)"', line_str)
                    if text_matches:
                        full_content += text_matches[0]
                except Exception as e:
                    print(f"Error processing chunk: {e}", file=sys.stderr)

            # Create a properly formatted response
            return {"message": {"role": message_role, "content": full_content}}

        except Exception as e:
            print(f"Request error: {e}", file=sys.stderr)
            return {
                "message": {
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                }
            }

    def pull_model(self, model_name):
        """Pull a model from the Ollama library"""
        payload = {"name": model_name}
        response = requests.post(f"{self.api_url}/pull", json=payload)
        if response.status_code == 200:
            return {
                "status": "success",
                "message": f"Model {model_name} pulled successfully",
            }
        else:
            raise Exception(f"Failed to pull model: {response.text}")

    def stream_generate(self, prompt, model=None, callback=None, options=None):
        """Stream generation results, calling the callback for each chunk"""
        model = model or self.default_model
        options = options or {}

        # Merge with default params and ensure streaming is enabled
        merged_options = {**self.default_params, **options, "stream": True}
        payload = {"model": model, "prompt": prompt, **merged_options}

        response = requests.post(f"{self.api_url}/generate", json=payload, stream=True)
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        text_chunk = chunk.get("response", "")
                        full_response += text_chunk
                        if callback:
                            callback(text_chunk, chunk)
                    except json.JSONDecodeError:
                        print(f"Error parsing chunk: {line}", file=sys.stderr)
            return full_response
        else:
            raise Exception(f"Failed to stream: {response.text}")


if __name__ == "__main__":
    client = OllamaClient()

    # Example: List available models
    try:
        models = client.list_models()
        print("Available models:")
        for model in models.get("models", []):
            print(f"- {model.get('name')}")
    except Exception as e:
        print(f"Error listing models: {e}")

    # Example: Generate text with a model
    try:
        response = client.generate(prompt="Explain quantum computing in simple terms")
        print("\nGenerated text:")
        print(response.get("response", ""))
    except Exception as e:
        print(f"Error generating text: {e}")

    # Example: Chat with a model
    try:
        chat_response = client.chat(
            messages=[
                {"role": "user", "content": "What are the three laws of robotics?"}
            ]
        )
        print("\nChat response:")
        print(chat_response.get("message", {}).get("content", ""))
    except Exception as e:
        print(f"Error in chat: {e}")

    # Example: Stream generation (uncomment to use)
    # def print_chunk(chunk, _):
    #     print(chunk, end="", flush=True)
    #
    # try:
    #     print("\nStreaming response:")
    #     client.stream_generate(
    #         prompt="Write a short poem about AI",
    #         callback=print_chunk
    #     )
    #     print("\n")
    # except Exception as e:
    #     print(f"\nError in streaming: {e}")
