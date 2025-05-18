# Ishtar AI

Ishtar AI is a Retrieval-Augmented Generation (RAG) assistant designed to help journalists operating in conflict zones. The system retrieves relevant documents and uses a language model to generate grounded answers.

## Components
- **FastAPI** provides the API server.
- **LangChain** orchestrates the RAG pipeline with **Llama 4**.
- **Pinecone** stores vector embeddings for similarity search.
- **Gradio** offers a lightweight UI for manual queries.
- **Poetry** manages dependencies.

External data sources such as ReliefWeb, ACLED, and UNHCR can be ingested to keep the knowledge base up to date.

## Development
1. Install Python 3.10 and project dependencies using [pyenv](https://github.com/pyenv/pyenv) and Poetry:

   ```bash
   pyenv install 3.10.x   # choose a patch version
   pyenv local 3.10.x
   poetry env use "$(pyenv which python)"
   poetry install
   ```

2. Run the API server:
   ```bash
   uvicorn ishtar_ai.app.main:app --reload
   ```

3. Launch the Gradio interface:
   ```bash
   python gradio_app.py
   ```

## Docker
Build and run with Docker:
```bash
docker build -t ishtar-ai .
docker run -p 8000:8000 ishtar-ai
```

## License
MIT
