#!/bin/bash
# Setup Tavily API key and run the app

# Activate virtual environment (if not already activated)
if [[ "$VIRTUAL_ENV" != *"llm_env"* ]]; then
    source bin/activate
fi

# Check if python-dotenv is installed
if ! pip show python-dotenv > /dev/null 2>&1; then
    echo "Installing python-dotenv..."
    pip install python-dotenv
fi

# Check if Tavily API key is provided as an argument
if [ -n "$1" ]; then
    # Run the setup script with the provided key
    python set_tavily_key.py "$1"
else
    # Ask for the API key interactively
    python set_tavily_key.py
fi

# Check if the setup was successful
if [ $? -ne 0 ]; then
    echo "Setup failed. Exiting."
    exit 1
fi

echo "✅ Tavily API key has been set up successfully!"
echo "You can now run the app with either:"
echo "  ./restart_app.sh"
echo "  or"
echo "  ./run_with_tavily.sh" 