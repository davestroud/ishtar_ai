#!/bin/bash
# Examples of interacting with Ollama API directly using curl

# Base URL for Ollama API
OLLAMA_API="http://localhost:11434/api"

# List all available models
echo "Listing available models..."
curl -s ${OLLAMA_API}/tags | jq

# Generate text with a model
echo -e "\nGenerating text with llama3..."
curl -s ${OLLAMA_API}/generate -d '{
  "model": "llama3",
  "prompt": "Explain how transformers work in AI in 3 sentences.",
  "temperature": 0.7,
  "max_tokens": 500
}' | jq

# Chat with a model
echo -e "\nChatting with llama3..."
curl -s ${OLLAMA_API}/chat -d '{
  "model": "llama3",
  "messages": [
    {
      "role": "user",
      "content": "What are some good practices for secure coding?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500
}' | jq

# Pull a model (this will take some time)
# Uncomment to use
# echo -e "\nPulling the mistral model..."
# curl -s ${OLLAMA_API}/pull -d '{
#   "name": "mistral"
# }' | jq

# Stream generation (shows tokens as they are generated)
echo -e "\nStreaming generation (press Ctrl+C to stop)..."
curl -s ${OLLAMA_API}/generate -d '{
  "model": "llama3",
  "prompt": "Write a haiku about coding.",
  "stream": true
}'

echo -e "\nDone!" 