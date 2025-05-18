"""Light-weight RAG pipeline (vector search + LLM).

If there is a local `.env` file (as generated from `env.example`) we load it
automatically so that variables like ``LLAMA_API_KEY`` and ``PINECONE_API_KEY``
are available during development without explicitly `export`-ing them.
"""

# Load env vars first (no hard dependency in production)
try:
    from dotenv import load_dotenv, find_dotenv

    _env_file = find_dotenv(usecwd=True)
    if _env_file:
        load_dotenv(_env_file, override=False)
    else:
        # .env not found—this is fine in production where variables are set
        pass
except ModuleNotFoundError:
    # python-dotenv is an optional dev dependency; ignore if missing
    pass

import os
import asyncio

# ---------------------------------------------------------------------------
# Pinecone (vector store) – TEMPORARILY DISABLED
# ---------------------------------------------------------------------------
# To get the FastAPI app running without external service errors, we wrap
# any Pinecone-related imports and initialisation behind a feature flag.  Set
# ``USE_PINECONE = True`` later when your credentials and index are ready.
# ---------------------------------------------------------------------------

# pylint: disable=invalid-name
USE_PINECONE = True  # Pinecone vector store enabled by default

try:
    from pinecone import Pinecone, ServerlessSpec  # SDK v3
except ImportError as _pc_err:  # legacy SDK installed
    raise RuntimeError(
        "Pinecone SDK v3+ is required. Run:  pip install --upgrade 'pinecone-client>=3'"
    ) from _pc_err

if USE_PINECONE:
    from langchain_openai.embeddings.base import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

# ----------------------------------------------------------------------------
# Pinecone configuration
# ----------------------------------------------------------------------------
# Required: PINECONE_API_KEY must be set in the environment.
# Optional: PINECONE_INDEX (defaults to "ishtar-ai").
# ----------------------------------------------------------------------------

if USE_PINECONE:
    # ------------------------------------------------------------------------
    #  Pinecone initialisation (SDK v3, serverless)
    # ------------------------------------------------------------------------
    _pc_api_key = os.environ.get("PINECONE_API_KEY")
    _pc_host = os.environ.get("PINECONE_HOST")  # serverless endpoint URL
    _pc_index = os.environ.get("PINECONE_INDEX", "ishtar-ai")

    if not _pc_api_key or not _pc_host:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_HOST are required.")

    pc = Pinecone(api_key=_pc_api_key)

    # Create index if missing (1536 dims → OpenAI text-embedding-3/text-ada)
    if _pc_index not in pc.list_indexes().names():
        pc.create_index(
            name=_pc_index,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(_pc_index, host=_pc_host)

    # LangChain wrapper (namespaces optional)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=OpenAIEmbeddings(),
        namespace="default",
    )
else:
    vectorstore = None  # type: ignore

# ---------------------------------------------------------------------------
# Meta Llama Developer client configuration
# ---------------------------------------------------------------------------

_llama_api_key = os.environ.get("LLAMA_API_KEY") or os.environ.get("LLM_API_KEY")
if not _llama_api_key:
    raise RuntimeError(
        "Missing API key. Set either LLAMA_API_KEY (preferred) or LLM_API_KEY."
    )

_llama_base_url = os.environ.get("LLAMA_API_URL", "https://api.llama.com/v1/")

_llama_model = os.environ.get("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")

# Meta Llama Developer SDK
from llama_api_client import LlamaAPIClient

llama_client = LlamaAPIClient(api_key=_llama_api_key, base_url=_llama_base_url)


# ---------------------------------------------------------------------------
# Tavily web search (real-time layer) – optional
# ---------------------------------------------------------------------------
from typing import List

try:
    from tavily import TavilyClient  # type: ignore

    _tavily_api_key = os.environ.get("TAVILY_API_KEY")
    USE_TAVILY = bool(_tavily_api_key)

    if USE_TAVILY:
        tavily_client = TavilyClient(api_key=_tavily_api_key)

except ModuleNotFoundError:
    USE_TAVILY = False

from langchain_core.documents import Document


# async helper to call Meta API in thread (SDK is sync)
async def _llama_chat(
    prompt: str, temperature: float = 0.1, max_tokens: int = 256
) -> str:
    def _sync_call():
        response = llama_client.chat.completions.create(
            model=_llama_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response

    resp = await asyncio.to_thread(_sync_call)

    # ---------------------------------------------------------------------
    # llama_api_client returns a ``CreateChatCompletionResponse`` with a
    # convenience field ``completion_message`` (plain string).  Older code
    # expected an OpenAI-style ``choices[0].message.content`` attribute.  We
    # support both for robustness.
    # ---------------------------------------------------------------------
    if hasattr(resp, "completion_message"):
        cm = resp.completion_message
        # Newer SDKs: completion_message is a model with a ``content`` attr
        if hasattr(cm, "content"):
            content = cm.content
            # content may be plain str or MessageTextContentItem
            if isinstance(content, str):
                return content.strip()
            if hasattr(content, "text"):
                return str(content.text).strip()
        # Older/edge versions: completion_message might be raw string
        if isinstance(cm, str):
            return cm.strip()

    if hasattr(resp, "choices") and resp.choices:
        return resp.choices[0].message.content.strip()

    raise RuntimeError(
        "Unexpected response schema from Llama API client: " f"{type(resp).__name__}"
    )


async def query_pipeline(query: str, *, top_k: int = 4) -> str:
    """Search the vector DB and answer using the LLM.

    Steps:
    1. similarity search → retrieve `top_k` documents
    2. build a simple prompt with the retrieved context
    3. ask the LLM for a grounded answer
    """

    # ------------------------------------------------------------------
    # 1. Retrieve documents from Pinecone and/or Tavily in parallel
    # ------------------------------------------------------------------

    async def _get_pine():
        if vectorstore is None:
            return []
        return vectorstore.similarity_search(query, k=top_k)  # sync but fast

    async def _get_tavily():
        if not USE_TAVILY:
            return []

        def _sync():
            # basic search depth, return dict with "results"
            res = tavily_client.search(query, max_results=top_k, search_depth="basic")
            docs: List[Document] = []
            for item in res.get("results", []):
                text = item.get("content") or item.get("title") or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "url": item.get("url"),
                            "similarity": item.get("score", 0.0),
                            "source": "tavily",
                        },
                    )
                )
            return docs

        return await asyncio.to_thread(_sync)

    pine_docs, web_docs = await asyncio.gather(_get_pine(), _get_tavily())

    docs_all = pine_docs + web_docs

    # simple sorting by similarity if present
    docs_all_sorted = sorted(
        docs_all,
        key=lambda d: d.metadata.get("similarity", 0.0),
        reverse=True,
    )[:top_k]

    if docs_all_sorted:
        context = "\n\n".join(d.page_content for d in docs_all_sorted)

        prompt = (
            "You are a helpful assistant. Use the CONTEXT below to answer the QUESTION.\n"
            "If the answer is not contained in the context, say 'I don't know'.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
        )
    else:
        # no retrieved docs
        prompt = (
            "You are a helpful assistant. Answer the QUESTION as best you can.\n\n"
            f"QUESTION: {query}\nANSWER:"
        )

    answer = await _llama_chat(prompt)
    return answer.strip()


# ---------------------------------------------------------------------------
# LangSmith tracing (optional)
# ---------------------------------------------------------------------------
try:
    from langsmith import traceable  # type: ignore

    def _traceable(name: str):  # wrapper to set default name easily
        return lambda fn: traceable(fn, name=name)

except ImportError:  # LangSmith not installed → no-op decorator

    def _traceable(name: str):
        def decorator(fn):
            return fn

        return decorator


# Apply LangSmith tracing (function wrapping) if available
_llama_chat = _traceable("LlamaChat")(_llama_chat)
query_pipeline = _traceable("QueryPipeline")(query_pipeline)


# ---------------------------------------------------------------------------
# Helper: ingest documents into Pinecone
# ---------------------------------------------------------------------------
async def ingest_documents(docs: list[str], metadatas: List[dict] | None = None):
    """Embed & upsert raw texts into Pinecone.

    Pass a list of raw strings (already chunked) and optional per-chunk metadata.
    """

    if not USE_PINECONE or vectorstore is None:
        raise RuntimeError("Pinecone is disabled.")

    _metas = metadatas or [{}] * len(docs)
    if len(_metas) != len(docs):
        raise ValueError("metadatas length must match docs length")

    # `add_texts` handles embeddings + upsert behind the scenes
    await asyncio.to_thread(vectorstore.add_texts, texts=docs, metadatas=_metas)
