#!/usr/bin/env python3
"""
Script to save the Tavily API key to the environment or .env file
"""
import os
import sys
import dotenv
from pathlib import Path


def main():
    # Check if key is provided as command line argument
    if len(sys.argv) < 2:
        key = input("Enter your Tavily API key: ").strip()
        if not key:
            print("Error: No API key provided")
            sys.exit(1)
    else:
        key = sys.argv[1].strip()

    # Validate the key format (basic check)
    if not key.startswith("tvly-"):
        print(
            "Warning: Tavily API keys typically start with 'tvly-'. Your key might not be valid."
        )
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != "y":
            sys.exit(1)

    # First, try to find or create a .env file
    env_file = Path(".") / ".env"

    # Check if .env file exists
    if env_file.exists():
        # Update existing .env file
        dotenv.load_dotenv(env_file)
        dotenv.set_key(env_file, "TAVILY_API_KEY", key)
        print(f"Updated Tavily API key in {env_file}")
    else:
        # Create new .env file
        with open(env_file, "w") as f:
            f.write(f"TAVILY_API_KEY={key}\n")
        print(f"Created .env file with Tavily API key at {env_file}")

    # Set the environment variable for the current session
    os.environ["TAVILY_API_KEY"] = key
    print("Environment variable TAVILY_API_KEY set for current session")

    # Test the API key by importing TavilySearch and making a test query
    try:
        print("Testing API key with Tavily API...")
        sys.path.append("./src")  # Add src to path for importing
        from tavily_search import TavilySearch

        client = TavilySearch(key)
        result = client.search("test query", max_results=1)
        print("✅ API key is valid! Tavily API is working correctly.")
        return True
    except Exception as e:
        print(f"❌ Error testing Tavily API: {e}")
        print("The API key might be invalid or there might be connectivity issues.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nSetup complete! You can now run the app with:")
        print("  streamlit run src/streamlit_app.py")
    else:
        print("\nSetup failed. Please check your API key and try again.")
