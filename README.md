# Ishtar AI

Ishtar AI is a Retrieval-Augmented Generation (RAG) assistant designed to help journalists and researchers by providing up-to-date information from various sources. The system can retrieve relevant documents from a local knowledge base and enhance answers with real-time web search capabilities.

## Core Features

-   **Local Knowledge Base:** Uses FAISS for efficient local similarity search over ingested documents.
-   **Large Language Model:** Leverages Meta Llama models via API for question answering and generation.
-   **Real-time Web Search:** Optionally integrates with Tavily API for current information.
-   **Data Ingestion:** Includes scripts to fetch and process data from sources like ReliefWeb, ACLED, and UNHCR (details in `ishtar_ai/data_ingestion/ingest.py`).
-   **Web Interface:** A Gradio app (`gradio_app.py`) provides an easy-to-use interface for queries.
-   **API Server:** A FastAPI backend (though the main interaction for RAG is currently via Gradio and `pipeline.py`).

## Setup and Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd ishtar_ai
    ```

2.  **Install Python 3.10+ and Poetry:**
    Ensure you have a compatible Python version (>=3.10, <3.14 as per `pyproject.toml`) and [Poetry](https://python-poetry.org/docs/#installation) installed.

3.  **Install dependencies:**
    This will create a virtual environment (`.venv`) in the project directory and install all necessary packages.
    ```bash
    poetry install
    ```

4.  **Set up environment variables:**
    Copy the example environment file and fill in your API keys and configurations.
    ```bash
    cp env.example .env
    ```
    Then edit `.env` with your actual credentials.

    **Required:**
    *   `OPENAI_API_KEY`: For OpenAI embeddings (used by FAISS).
    *   `LLAMA_API_KEY`: For Meta Llama models. (Alternatively, `LLM_API_KEY` can be used).

    **Optional:**
    *   `TAVILY_API_KEY`: For real-time web search.
    *   `FAISS_INDEX_PATH`: Path to store/load the FAISS index (defaults to `./faiss_index`).
    *   `LLAMA_API_URL`: Custom Llama API endpoint.
    *   `LLAMA_MODEL`: Specific Llama model to use.
    *   `OPENAI_EMBEDDING_MODEL`: Specific OpenAI embedding model.
    *   `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_ENDPOINT`: For LangSmith tracing.

5.  **Activate the virtual environment (optional but recommended for direct script execution):**
    ```bash
    source .venv/bin/activate
    ```
    Or, run commands via `poetry run <command>`.

## Running the Application

### 1. Ingest Data (Important First Step)

To populate your local FAISS vector store, you need to run the ingestion script. (You'll need to adapt or confirm the command based on how `ishtar_ai/data_ingestion/ingest.py` is structured to be called).

Example (assuming `ingest.py` has a main execution block):
```bash
poetry run python ishtar_ai/data_ingestion/ingest.py
```
This will create FAISS index files (e.g., `faiss_index.faiss`, `faiss_index.pkl`) at the path specified by `FAISS_INDEX_PATH` (or the default `./faiss_index`).

### 2. Run the Gradio Web Interface

Once you have ingested some data (or if you only want to use Tavily search initially), you can start the Gradio application:
```bash
poetry run python gradio_app.py
```
This will provide a local URL (usually `http://127.0.0.1:7860` or similar) to access the chat interface.

## Development Utilities

-   **Check Environment Variables:**
    A utility script `check_env.py` can help verify that your environment variables are loaded and basic API connectivity.
    ```bash
    poetry run python check_env.py
    ```

## Docker (Optional)

If a `Dockerfile` is configured for the current setup:
```bash
docker build -t ishtar-ai .
# Ensure .env file is available to Docker or secrets are managed appropriately
docker run -p 8000:8000 --env-file .env ishtar-ai 
```
*(Note: The existing `Dockerfile` might need updates to reflect changes from Pinecone to FAISS and dependency updates.)*
