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
        # Default timeout for requests (30 seconds)
        self.timeout = 30

    def list_models(self):
        """List all available models"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to list models: {response.text}")
        except requests.exceptions.Timeout:
            return {"models": [{"name": "llama3"}]}  # Default fallback
        except Exception as e:
            raise Exception(f"Error connecting to Ollama: {str(e)}")

    def generate(self, prompt, model=None, options=None, timeout=None):
        """Generate a response using the specified model"""
        model = model or self.default_model
        request_timeout = timeout or self.timeout

        if options is None:
            options = {}

        # Merge with default params
        merged_options = {**self.default_params, **options}

        payload = {"model": model, "prompt": prompt, **merged_options}

        try:
            response = requests.post(
                f"{self.api_url}/generate", json=payload, timeout=request_timeout
            )

            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    # Handle malformed JSON
                    print(
                        f"Warning: Received malformed JSON. Attempting to fix. Error: {e}"
                    )
                    # Return a simple dict with the text
                    return {"response": self._sanitize_response(response.text)}
            else:
                raise Exception(f"Failed to generate: {response.text}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {request_timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in generate request: {str(e)}")

    def chat(self, messages, model=None, options=None, timeout=None):
        """Chat with the model using a conversation format"""
        model = model or self.default_model
        request_timeout = timeout or self.timeout

        if options is None:
            options = {}

        # Merge with default params
        merged_options = {**self.default_params, **options}

        payload = {"model": model, "messages": messages, **merged_options}

        try:
            response = requests.post(
                f"{self.api_url}/chat", json=payload, timeout=request_timeout
            )

            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    # Handle malformed JSON
                    print(
                        f"Warning: Received malformed JSON. Attempting to fix. Error: {e}"
                    )
                    # Create a fallback response with the raw text
                    return {
                        "message": {
                            "role": "assistant",
                            "content": self._sanitize_response(response.text),
                        }
                    }
            else:
                raise Exception(f"Failed to chat: {response.text}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {request_timeout} seconds")
        except Exception as e:
            # Handle any other exceptions
            print(f"Error in chat request: {str(e)}")
            return {
                "message": {
                    "role": "assistant",
                    "content": f"I encountered an error processing your request. Please try again or simplify your query. Error: {str(e)}",
                }
            }

    def _sanitize_response(self, text):
        """Attempt to clean up and sanitize a malformed response"""
        if not text:
            return "No response received"

        # Try to find the useful part of the response
        # This handles cases where there might be multiple JSON objects or extra data
        lines = text.split("\n")

        # Try to find a valid JSON object in the response
        for line in lines:
            line = line.strip()
            if line:
                try:
                    # Try to parse as JSON
                    data = json.loads(line)
                    if isinstance(data, dict) and "response" in data:
                        return data["response"]
                except:
                    pass

        # If we couldn't find a valid JSON, just return the first line
        if lines and lines[0]:
            return lines[0]

        return text

    def pull_model(self, model_name, timeout=120):
        """Pull a model from the Ollama library"""
        payload = {"name": model_name}
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json=payload,
                timeout=timeout,  # Longer timeout for model pulling
            )

            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": f"Model {model_name} pulled successfully",
                }
            else:
                raise Exception(f"Failed to pull model: {response.text}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Model pull timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Error pulling model: {str(e)}")

    def stream_generate(
        self, prompt, model=None, callback=None, options=None, timeout=60
    ):
        """Stream generation results, calling the callback for each chunk"""
        model = model or self.default_model

        if options is None:
            options = {}

        # Merge with default params and ensure streaming is enabled
        merged_options = {**self.default_params, **options, "stream": True}

        payload = {"model": model, "prompt": prompt, **merged_options}

        try:
            response = requests.post(
                f"{self.api_url}/generate", json=payload, stream=True, timeout=timeout
            )

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
                            # Skip malformed lines
                            continue
                return full_response
            else:
                raise Exception(f"Failed to stream: {response.text}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Stream request timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Error in stream request: {str(e)}")


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
        response = client.generate(
            prompt="Explain quantum computing in simple terms",
            timeout=15,  # 15 second timeout
        )
        print("\nGenerated text:")
        print(response.get("response", ""))
    except TimeoutError as e:
        print(f"\nTimeout error: {e}")
    except Exception as e:
        print(f"\nError generating text: {e}")

    # Example: Chat with a model
    try:
        chat_response = client.chat(
            messages=[
                {"role": "user", "content": "What are the three laws of robotics?"}
            ],
            timeout=15,  # 15 second timeout
        )
        print("\nChat response:")
        print(chat_response.get("message", {}).get("content", ""))
    except TimeoutError as e:
        print(f"\nTimeout error: {e}")
    except Exception as e:
        print(f"\nError in chat: {e}")
