"""Light-weight RAG pipeline (vector search + LLM).

If there is a local `.env` file (as generated from `env.example`) we load it
automatically so that variables like ``LLAMA_API_KEY`` and ``PINECONE_API_KEY``
are available during development without explicitly `export`-ing them.
"""

import os  # Ensure os is imported at the very top

# Load env vars first (no hard dependency in production)
try:
    from dotenv import load_dotenv, find_dotenv

    _env_file = find_dotenv(usecwd=True)
    if _env_file:
        print(f"[DEBUG] .env file found at: {_env_file}")
        load_dotenv(_env_file, override=True)
        print(
            f"[DEBUG] TAVILY_API_KEY from os.environ after load: {os.environ.get('TAVILY_API_KEY')}"
        )
    else:
        # find_dotenv(usecwd=True) already checks os.getcwd(), this extra check is redundant.
        # If it's not found by find_dotenv, it's not in the current working directory.
        print("[DEBUG] .env file not found by find_dotenv(usecwd=True).")

except ModuleNotFoundError:
    print("[DEBUG] python-dotenv module not found.")
    pass

import asyncio
import logging
from typing import Any, List, Dict, Optional, Union, Type

from langchain_core.documents import (
    Document,
)  # Moved import higher as it's used in ingest_documents

# Setup basic logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# API Keys and Configuration (excluding Pinecone)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
logger.info(f"[DEBUG] OPENAI_API_KEY loaded: {'SET' if OPENAI_API_KEY else 'NOT SET'}")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY") or os.getenv("LLM_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "https://api.llama.com/v1/")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


_embeddings: Any = None
vectorstore: Any = None

# Initialize FAISS vector store
logger.info("Attempting to initialize FAISS vector store with OpenAIEmbeddings.")
if OPENAI_API_KEY:
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS

        _embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        logger.info(
            f"Using OpenAIEmbeddings (model: {_embeddings.model}) and FAISS as vector store."
        )

        faiss_parent_dir = os.path.dirname(FAISS_INDEX_PATH)
        if faiss_parent_dir and not os.path.exists(faiss_parent_dir):
            os.makedirs(faiss_parent_dir, exist_ok=True)
            logger.info(f"Created directory for FAISS index: {faiss_parent_dir}")

        index_file_faiss = FAISS_INDEX_PATH + ".faiss"
        index_file_pkl = FAISS_INDEX_PATH + ".pkl"

        if os.path.exists(index_file_faiss) and os.path.exists(index_file_pkl):
            logger.info(f"Loading existing FAISS index from '{FAISS_INDEX_PATH}'.")
            # Determine folder_path correctly for load_local
            # If FAISS_INDEX_PATH is "./faiss_index", folder_path is "." and index_name is "faiss_index"
            load_folder_path = (
                os.path.dirname(FAISS_INDEX_PATH)
                if os.path.dirname(FAISS_INDEX_PATH)
                else "."
            )
            load_index_name = os.path.basename(FAISS_INDEX_PATH)

            vectorstore = FAISS.load_local(
                folder_path=load_folder_path,
                embeddings=_embeddings,
                index_name=load_index_name,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                f"FAISS index loaded. Index has {vectorstore.index.ntotal if vectorstore and hasattr(vectorstore, 'index') else 'N/A'} vectors."
            )
        else:
            logger.info(
                f"No existing FAISS index found at '{FAISS_INDEX_PATH}' (checked for .faiss and .pkl). Will be created on first ingestion."
            )
            vectorstore = None

    except ImportError as e:
        logger.error(
            f"Failed to import OpenAI or FAISS modules: {e}. FAISS setup failed. _embeddings and vectorstore will be None."
        )
        _embeddings = None
        vectorstore = None
    except Exception as e:
        logger.error(
            f"Error initializing FAISS: {e}. _embeddings and vectorstore will be None."
        )
        _embeddings = None
        vectorstore = None
else:
    logger.warning(
        "OPENAI_API_KEY not found. FAISS vector store cannot be initialized (requires OpenAIEmbeddings). _embeddings and vectorstore will be None."
    )
    _embeddings = None
    vectorstore = None

# ---------------------------------------------------------------------------
# Meta Llama Developer client configuration
# ---------------------------------------------------------------------------
if not LLAMA_API_KEY:
    # This was previously a RuntimeError, changing to a warning to allow app to run without LLM
    logger.warning(
        "Missing API key for Llama. Set either LLAMA_API_KEY (preferred) or LLM_API_KEY. LLM functionality will be disabled."
    )
    llama_client = None
else:
    from llama_api_client import LlamaAPIClient

    llama_client = LlamaAPIClient(api_key=LLAMA_API_KEY, base_url=LLAMA_API_URL)
    logger.info(f"Llama client initialized for model: {LLAMA_MODEL}")


# ---------------------------------------------------------------------------
# Tavily web search (real-time layer) – optional
# ---------------------------------------------------------------------------
USE_TAVILY = False
tavily_client = None
if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient

        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        USE_TAVILY = True
        logger.info(
            f"Tavily client initialized with API key ending: ...{TAVILY_API_KEY[-4:]}"
        )
    except ImportError:
        logger.warning("TavilyClient module not found. Tavily search will be disabled.")
    except Exception as e:
        logger.error(
            f"Error initializing Tavily client: {e}. Tavily search will be disabled."
        )
else:
    logger.info(
        "[Tavily Setup] TAVILY_API_KEY not set in environment. Tavily will be disabled."
    )


# async helper to call Meta API in thread (SDK is sync)
async def _llama_chat(
    prompt: str, temperature: float = 0.1, max_tokens: int = 256
) -> str:
    if not llama_client:
        logger.error("Llama client not initialized. Cannot fulfill chat request.")
        return "I am currently unable to process chat requests as the Llama client is not configured."

    def _sync_call():
        response = llama_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response

    try:
        resp = await asyncio.to_thread(_sync_call)
        if hasattr(resp, "completion_message"):
            cm = resp.completion_message
            if hasattr(cm, "content"):
                content = cm.content
                if isinstance(content, str):
                    return content.strip()
                if hasattr(content, "text"):  # For MessageTextContentItem
                    return str(content.text).strip()
            if isinstance(cm, str):  # Older SDKs
                return cm.strip()
        if hasattr(resp, "choices") and resp.choices:  # OpenAI-style
            return resp.choices[0].message.content.strip()

        logger.error(
            f"Unexpected response schema from Llama API client: {type(resp).__name__}, content: {resp}"
        )
        return "Error: Received an unexpected response from the Llama API."

    except Exception as e:
        logger.error(f"Error during Llama API call: {e}")
        return f"Error processing your request with the Llama API: {e}"


async def query_pipeline(query: str, *, top_k: int = 4) -> str:
    async def _get_vector_docs():
        if vectorstore is None:
            logger.info(
                "Vector store (FAISS) is not initialized. Skipping similarity search."
            )
            return []
        if not hasattr(vectorstore, "similarity_search"):
            logger.error(
                f"Vector store of type {type(vectorstore)} does not have similarity_search method."
            )
            return []
        logger.info(
            f"Performing similarity search in FAISS for query: '{query}' with k={top_k}"
        )
        try:
            return await asyncio.to_thread(
                vectorstore.similarity_search, query, k=top_k
            )
        except Exception as e:
            logger.error(f"Error during FAISS similarity search: {e}")
            return []

    async def _get_tavily_docs():  # Renamed for clarity
        if not USE_TAVILY or not tavily_client:
            logger.info("Skipping Tavily search (disabled or client not initialized).")
            return []
        logger.info(f"Fetching real-time info from Tavily for query: '{query}'")

        def _sync_tavily_search():
            try:
                res = tavily_client.search(
                    query, max_results=top_k, search_depth="basic"
                )
                logger.info(f"Tavily API raw response: {res}")
            except Exception as e:
                logger.error(f"Tavily API call failed: {e}")
                return []

            docs: List[Document] = []
            for item in res.get("results", []):
                text = item.get("content") or item.get("title") or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "url": item.get("url"),
                            "similarity": item.get(
                                "score", 0.0
                            ),  # Tavily calls it score
                            "source": "tavily",
                        },
                    )
                )
            return docs

        docs_returned = await asyncio.to_thread(_sync_tavily_search)
        logger.info(f"Documents processed from Tavily: {len(docs_returned)}")
        return docs_returned

    vector_docs, web_docs = await asyncio.gather(_get_vector_docs(), _get_tavily_docs())

    docs_all = vector_docs + web_docs
    docs_all_sorted = sorted(
        docs_all,
        key=lambda d: (
            d.metadata.get("similarity", 0.0) if isinstance(d.metadata, dict) else 0.0
        ),  # Guard against non-dict metadata
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
    from langsmith import traceable

    def _traceable(name: str):
        return lambda fn: traceable(fn, name=name)

except ImportError:

    def _traceable(name: str):
        def decorator(fn):
            return fn

        return decorator


_llama_chat = _traceable("LlamaChat")(_llama_chat)
query_pipeline = _traceable("QueryPipeline")(query_pipeline)


# ---------------------------------------------------------------------------
# Helper: ingest documents into FAISS
# ---------------------------------------------------------------------------
async def ingest_documents(docs: list[str], metadatas: List[dict] | None = None):
    global vectorstore
    if not _embeddings:
        logger.error(
            "Embeddings are not initialized (OpenAIEmbeddings). Cannot ingest documents."
        )
        return

    _metas = metadatas or [{}] * len(docs)
    if len(_metas) != len(docs):
        raise ValueError("metadatas length must match docs length")

    langchain_documents = [
        Document(page_content=t, metadata=m) for t, m in zip(docs, _metas)
    ]
    logger.info(
        f"Preparing to ingest {len(langchain_documents)} documents into FAISS vector store."
    )

    try:
        if vectorstore is None:
            logger.info(f"Creating new FAISS index at '{FAISS_INDEX_PATH}'.")
            new_vs = await asyncio.to_thread(
                FAISS.from_documents,
                documents=langchain_documents,
                embedding=_embeddings,
            )
            vectorstore = new_vs
            logger.info(
                f"New FAISS index created with {vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 'N/A'} vectors."
            )
        else:
            logger.info(
                f"Adding {len(langchain_documents)} documents to existing FAISS index."
            )
            await asyncio.to_thread(
                vectorstore.add_documents, documents=langchain_documents
            )
            logger.info(
                f"Documents added. FAISS index now has {vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 'N/A'} vectors."
            )

        if vectorstore:
            # Determine folder_path correctly for save_local
            save_folder_path = (
                os.path.dirname(FAISS_INDEX_PATH)
                if os.path.dirname(FAISS_INDEX_PATH)
                else "."
            )
            save_index_name = os.path.basename(FAISS_INDEX_PATH)

            await asyncio.to_thread(
                vectorstore.save_local,
                folder_path=save_folder_path,
                index_name=save_index_name,
            )
            logger.info(
                f"FAISS index saved to folder '{save_folder_path}' with name '{save_index_name}'. (Files: {save_index_name}.faiss, {save_index_name}.pkl)"
            )

    except Exception as e:
        logger.error(f"Failed to ingest documents into FAISS: {e}")
        # Optionally, re-raise or handle more gracefully depending on expected caller behavior
        raise
