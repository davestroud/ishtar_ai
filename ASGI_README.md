# Ishtar AI ASGI API

This document provides instructions for using the ASGI (Asynchronous Server Gateway Interface) API for the Ishtar AI project. The ASGI API allows for real-time communication via WebSockets and provides additional REST endpoints.

## Overview

The ASGI application provides:

1. A REST API for various functionalities
2. WebSocket support for real-time chat
3. A simple web interface to demo the API
4. Integration with the Streamlit app (launched as a subprocess)

## Running the ASGI Application

### Using Python directly

```bash
# Install the required dependencies
poetry add uvicorn fastapi starlette

# Run the ASGI app
python asgi_app.py
```

Or using uvicorn directly:

```bash
uvicorn asgi_app:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker Compose

```bash
docker-compose -f docker-compose-asgi.yml up
```

## Accessing the Applications

- ASGI API: http://localhost:8000
- Streamlit UI: http://localhost:8501
- Simple Web Demo: http://localhost:8000/static/index.html

## API Endpoints

### REST Endpoints

- `GET /` - Root endpoint with info
- `GET /api/health` - Health check
- `GET /api/models` - List available Ollama models
- `POST /api/search` - Search the web with Tavily
  - Payload: `{"query": "your search query"}`
- `POST /api/vector-search` - Search vectors in Pinecone
  - Payload: `{"vector": [0.1, 0.2, ...], "top_k": 3}`

### WebSocket Endpoint

- `WebSocket /ws/chat` - Real-time chat interface
  - Connect to this endpoint to establish a WebSocket connection
  - Send JSON messages in the format:
    ```json
    {
        "message": "Your message here",
        "model": "llama3",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    ```
  - Receive JSON responses in the format:
    ```json
    {
        "response": "AI response text",
        "model": "llama3"
    }
    ```

## Example WebSocket Usage

Here's a simple example of how to use the WebSocket API:

```javascript
// Connect to WebSocket
const socket = new WebSocket('ws://localhost:8000/ws/chat');

// Send a message
socket.onopen = function(e) {
    socket.send(JSON.stringify({
        message: "Hello, what can you tell me about the latest news in Syria?",
        model: "llama3",
        temperature: 0.7,
        max_tokens: 1000
    }));
};

// Receive response
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log("Response:", data.response);
};
```

## Why ASGI?

ASGI (Asynchronous Server Gateway Interface) provides several advantages over traditional WSGI:

1. **Asynchronous Support**: Handles WebSockets and long-lived connections
2. **Better Performance**: Efficiently manages multiple concurrent connections
3. **Modern Protocols**: Supports HTTP/2 and WebSockets
4. **Real-time Communication**: Enables bidirectional communication

For more information on ASGI, see the [ASGI Specification](https://asgi.readthedocs.io/en/latest/specs/index.html). 