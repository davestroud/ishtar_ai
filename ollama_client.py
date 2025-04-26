#!/usr/bin/env python3
import requests
import json
import os
from dotenv import load_dotenv
from config import get_client_config

# Load environment variables
load_dotenv()


class OllamaClient:
    def __init__(self, base_url=None):
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

        if options is None:
            options = {}

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

        if options is None:
            options = {}

        # Merge with default params
        merged_options = {**self.default_params, **options}

        payload = {"model": model, "messages": messages, **merged_options}

        response = requests.post(f"{self.api_url}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to chat: {response.text}")

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

        if options is None:
            options = {}

        # Merge with default params and ensure streaming is enabled
        merged_options = {**self.default_params, **options, "stream": True}

        payload = {"model": model, "prompt": prompt, **merged_options}

        response = requests.post(f"{self.api_url}/generate", json=payload, stream=True)
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    text_chunk = chunk.get("response", "")
                    full_response += text_chunk
                    if callback:
                        callback(text_chunk, chunk)
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
