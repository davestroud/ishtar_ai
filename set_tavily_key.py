#!/usr/bin/env python3
import os
import sys
import re
import requests
from pathlib import Path
from dotenv import load_dotenv, set_key


def is_valid_api_key(api_key):
    """Check if the API key appears to be valid (correct format)"""
    # Tavily API keys typically start with "tvly-" followed by a string of characters
    return bool(re.match(r"^tvly-[a-zA-Z0-9]+$", api_key))


def test_tavily_api(api_key):
    """Test if the Tavily API key works by making a simple request"""
    print(f"Testing Tavily API key: {api_key[:5]}...")

    # Base URL for Tavily API
    base_url = "https://api.tavily.com"

    # Set up headers with proper authorization
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        # First try a simple status check
        status_url = f"{base_url}/status"
        print(f"Checking API status at: {status_url}")
        response = requests.get(status_url, timeout=5)
        print(f"Status check response: {response.status_code}")

        if response.status_code != 200:
            print(f"Warning: Status check failed with code {response.status_code}")

        # Now try a simple search to fully validate the key
        search_url = f"{base_url}/search"
        payload = {
            "query": "Test query from Ollama integration",
            "search_depth": "basic",
            "max_results": 1,
        }

        print(f"Making test search request to: {search_url}")
        response = requests.post(search_url, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            print("✅ Tavily API key is valid and working!")
            return True
        else:
            print(f"❌ API key test failed with status code: {response.status_code}")
            try:
                error_details = response.json()
                print(f"Error details: {error_details}")
            except:
                print(f"Error response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")
        return False


def main():
    # Check if we received the API key as a command-line argument
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        # Otherwise, prompt the user
        api_key = input("Enter your Tavily API key (starts with 'tvly-'): ")

    # Trim any whitespace
    api_key = api_key.strip()

    # Basic validation
    if not is_valid_api_key(api_key):
        print("❌ Invalid API key format. Tavily API keys typically start with 'tvly-'")
        print("You can get a free API key at: https://tavily.com")
        return 1

    # Find the .env file or create one
    env_path = Path(".env")

    # Create the .env file if it doesn't exist
    if not env_path.exists():
        print(f"Creating .env file at {env_path.absolute()}")
        env_path.touch()

    # Test the API key
    if test_tavily_api(api_key):
        # Save to .env file
        set_key(str(env_path), "TAVILY_API_KEY", api_key)
        print(f"✅ API key saved to {env_path.absolute()}")
        return 0
    else:
        print(
            "\n❌ API key validation failed. The key may be invalid or there might be network issues."
        )
        print("If you're sure the key is correct, you can still save it by running:")
        print(f"echo 'TAVILY_API_KEY={api_key}' > .env")
        return 1


if __name__ == "__main__":
    sys.exit(main())
