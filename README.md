# Ishtar AI

Ishtar AI is a Retrieval-Augmented Generation (RAG) assistant designed to help journalists operating in conflict zones. The system retrieves relevant documents and uses a language model to generate grounded answers.

## Components
- **FastAPI** provides the API server.
- **LangChain** orchestrates the RAG pipeline with **Llama 4**.
- **Pinecone** stores vector embeddings for similarity search.
- **Gradio** offers a lightweight UI for manual queries.
- **Poetry** manages dependencies.

External data sources such as ReliefWeb, ACLED, and UNHCR can be ingested to keep the knowledge base up to date.

## Development
1. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
2. Activate the virtual environment (optional):
   ```bash
   source .venv/bin/activate
   ```
3. Copy the example environment file and fill in credentials:
   ```bash
   cp env.example .env
   ```
4. Run the API server:
   ```bash
   pkill -f 'uvicorn.*ishtar_ai.app.main' || true
   uvicorn ishtar_ai.app.main:app --reload
   ```
5. Launch the Gradio interface:
   ```bash
   python gradio_app.py
   ```

## Docker
Build and run with Docker:
```bash
docker build -t ishtar-ai .
docker run -p 8000:8000 ishtar-ai
```

# Environment variables (.env)

```
# Meta Llama Developer
LLAMA_API_KEY=llama-xxxxxxxxxxxxxxxxxxxxxxxx

# Pinecone (serverless SDK v3)
PINECONE_API_KEY=pcd-xxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_HOST=quickstart-xxxxxx.svc.us-east-1.aws.pinecone.io
PINECONE_INDEX=ishtar-ai

# Tavily (optional real-time web search)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx

# LangSmith tracing (optional)
LANGCHAIN_API_KEY=lsm-xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
```
