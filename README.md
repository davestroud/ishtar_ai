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

# Ishtar AI – example environment variables

# OpenAI
OPENAI_API_KEY=

# Pinecone
PINECONE_API_KEY=
PINECONE_ENV=
PINECONE_INDEX=ishtar-ai

# Hugging Face / Llama
HUGGINGFACEHUB_API_TOKEN=<your Hugging Face access token>
LLAMA_REPO=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8

from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace
from langchain_core.messages import HumanMessage

_hf_client = InferenceClient(
    model=LLAMA_REPO,
    token=HUGGINGFACEHUB_API_TOKEN,
    provider="hf-inference",   # valid provider name
)

chat_llm = ChatHuggingFace(
    llm=_hf_client,
    temperature=0.1,
    max_new_tokens=256,
)

async def query_pipeline(prompt: str) -> str:
    chat_response = await chat_llm.ainvoke([HumanMessage(content=prompt)])
    return chat_response.content.strip()
