#!/usr/bin/env python3
"""
ASGI application for Ishtar AI project
Provides both the Streamlit interface and additional API endpoints
"""

import os
import sys
import uvicorn
import subprocess
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ishtar AI",
    description="Ishtar AI Initiative API",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.pinecone_integration import get_pinecone_client
    from src.tavily_search import TavilySearch
    from src.langsmith_integration import get_langsmith_client

    pinecone_client = get_pinecone_client()
except ImportError as e:
    logger.warning(f"Failed to import modules: {e}")
    pinecone_client = None

# Streamlit subprocess
streamlit_process = None


@app.on_event("startup")
async def startup_event():
    """Start the Streamlit app in a subprocess when the ASGI app starts"""
    global streamlit_process
    try:
        # Start Streamlit in a subprocess
        streamlit_cmd = [
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port",
            "8501",
        ]
        streamlit_process = subprocess.Popen(
            streamlit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Started Streamlit subprocess")
    except Exception as e:
        logger.error(f"Failed to start Streamlit: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the Streamlit app when the ASGI app shuts down"""
    global streamlit_process
    if streamlit_process:
        streamlit_process.terminate()
        logger.info("Terminated Streamlit subprocess")


@app.get("/")
async def root():
    """Root endpoint that redirects to the Streamlit UI"""
    return {
        "message": "Welcome to Ishtar AI API",
        "streamlit_ui": "http://localhost:8501",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/models")
async def list_models():
    """List available models from Ollama"""
    try:
        from ollama_client import OllamaClient

        client = OllamaClient()
        models_response = client.list_models()
        return models_response
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to fetch models: {str(e)}"}
        )


@app.post("/api/search")
async def search(request: Request):
    """Search the web using Tavily"""
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            return JSONResponse(
                status_code=400, content={"error": "Missing 'query' parameter"}
            )

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=500, content={"error": "Tavily API key not configured"}
            )

        tavily_client = TavilySearch(api_key)
        search_result = tavily_client.search(
            query=query, search_depth="advanced", max_results=3
        )
        return search_result
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Search failed: {str(e)}"}
        )


@app.post("/api/vector-search")
async def vector_search(request: Request):
    """Search vectors in Pinecone"""
    try:
        if not pinecone_client or not pinecone_client.index:
            return JSONResponse(
                status_code=500, content={"error": "Pinecone client not initialized"}
            )

        data = await request.json()
        vector = data.get("vector")
        top_k = data.get("top_k", 3)

        if not vector:
            return JSONResponse(
                status_code=400, content={"error": "Missing 'vector' parameter"}
            )

        results = pinecone_client.query(
            vector=vector, top_k=top_k, include_metadata=True
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Vector search failed: {str(e)}"}
        )


# WebSocket connection for real-time chat
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        from ollama_client import OllamaClient

        client = OllamaClient()

        # Initialize chat history for this connection
        chat_history = []

        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            model_name = message_data.get("model", "llama3")
            temperature = message_data.get("temperature", 0.7)
            max_tokens = message_data.get("max_tokens", 1000)

            # Add to history
            chat_history.append({"role": "user", "content": user_message})

            # Get response from Ollama
            try:
                response = client.chat(
                    messages=chat_history,
                    model=model_name,
                    options={"temperature": temperature, "max_tokens": max_tokens},
                )
                assistant_response = response.get("message", {}).get("content", "")

                # Add assistant response to history
                chat_history.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # Send response back to client
                await websocket.send_json(
                    {"response": assistant_response, "model": model_name}
                )
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                await websocket.send_json(
                    {"error": f"Error generating response: {str(e)}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"error": f"Error: {str(e)}"})
        except:
            pass


if __name__ == "__main__":
    # Run the ASGI app with Uvicorn
    uvicorn.run(
        "asgi_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
