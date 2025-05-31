"""Data ingestion from external sources."""

from typing import List
import httpx
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
import asyncio

# Import helper from the RAG pipeline so we re-use the existing Pinecone initialisation logic.
from ..rag.pipeline import ingest_documents

RELIEFWEB_API = "https://api.reliefweb.int/v1/reports"
ACLED_API = "https://api.acleddata.com"
UNHCR_API = "https://api.unhcr.org"

# ---------------------------------------------------------------------------
# Text splitter (OpenAI embeddings → 1536 tokens ≈ 4k chars ⇒ 1k chunk)
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# ---------------------------------------------------------------------------
# Fetch helpers – each returns a list[Document]
# ---------------------------------------------------------------------------


# ReliefWeb ------------------------------------------------------------------
async def fetch_reliefweb(query: str, *, limit: int = 20) -> List[Document]:
    """Return ReliefWeb reports matching *query* via HTTP GET.

    We stick to the public GET endpoint (query-string syntax) which is more
    permissive than the JSON-POST variant and avoids 400 errors for most
    simple searches.
    """

    # ReliefWeb occasionally refuses complex GET queries with a 400. We first
    # try the lighter-weight GET path; if it fails, transparently retry as a
    # POST with a JSON body that the API is more forgiving with.

    async def _parse(resp: httpx.Response) -> List[Document]:
        resp.raise_for_status()
        out: List[Document] = []
        for item in resp.json().get("data", []):
            fields = item.get("fields", {})
            text = (
                fields.get("body-html")
                or fields.get("body")
                or fields.get("description")
                or ""
            )
            out.append(
                Document(
                    page_content=text[:15000],
                    metadata={
                        "source": "reliefweb",
                        "url": fields.get("url"),
                        "title": fields.get("title"),
                    },
                )
            )
        return out

    async with httpx.AsyncClient() as client:
        try:
            params = {
                "appname": "ishtar-ai",
                "query[value]": query,
                "limit": limit,
                "profile": "lite",
                "format": "json",
            }
            get_resp = await client.get(
                RELIEFWEB_API,
                params=params,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            return await _parse(get_resp)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise  # unknown failure – propagate

            # Fallback: JSON POST
            payload = {
                "query": {"value": query, "operator": "AND"},
                "limit": limit,
                "profile": "lite",
            }
            post_resp = await client.post(
                f"{RELIEFWEB_API}?appname=ishtar-ai&format=json",
                json=payload,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            return await _parse(post_resp)


# ACLED ----------------------------------------------------------------------
async def fetch_acled(query: str, *, limit: int = 20) -> List[Document]:
    """Return ACLED events matching *query*.

    ACLED requires an API key provided via the `ACLED_API_KEY` environment
    variable. If it's missing we just return an empty list so that ingestion
    can continue with other data sources.
    """

    api_key = os.getenv("ACLED_API_KEY")
    if not api_key:
        # silently skip if no key – not fatal
        return []

    params = {"search": query, "limit": limit, "format": "json", "key": api_key}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{ACLED_API}/v1/events", params=params, timeout=30)
        resp.raise_for_status()
        events: List[Document] = []
        for item in resp.json().get("data", []):
            events.append(
                Document(
                    page_content=json.dumps(item),
                    metadata={"source": "acled", "event_id": item.get("event_id_cnty")},
                )
            )
        return events


# UNHCR ----------------------------------------------------------------------
async def fetch_unhcr(query: str, *, limit: int = 20) -> List[Document]:
    """Return UNHCR documents matching *query*.

    The UNHCR API is not officially documented; we tolerate failures and simply
    return an empty list in such cases so that ingestion continues smoothly.
    """

    params = {"q": query, "size": limit}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{UNHCR_API}/v1/search", params=params, timeout=30)
            resp.raise_for_status()
            docs: List[Document] = []
            for hit in resp.json().get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                docs.append(
                    Document(
                        page_content=src.get("content", ""),
                        metadata={"source": "unhcr", "id": hit.get("_id")},
                    )
                )
            return docs
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[Ingest] UNHCR fetch failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Chunk & ingest
# ---------------------------------------------------------------------------


def _chunk_documents(docs: List[Document]):
    texts: List[str] = []
    metadatas: List[dict] = []
    for doc in docs:
        for chunk in _splitter.split_text(doc.page_content):
            texts.append(chunk)
            metadatas.append(doc.metadata)
    return texts, metadatas


async def ingest_external_sources(query: str, *, max_docs: int = 20):
    """Fetch, chunk & store fresh material related to *query* in Pinecone."""

    # --------------------------------------------------------------------
    # 1. Gather documents concurrently from the 3 sources
    # --------------------------------------------------------------------

    results = await asyncio.gather(
        fetch_reliefweb(query, limit=max_docs),
        fetch_acled(query, limit=max_docs),
        fetch_unhcr(query, limit=max_docs),
        return_exceptions=True,
    )

    raw_docs: List[Document] = []
    for res in results:
        if isinstance(res, Exception):
            # Log & continue (network errors shouldn't abort the ingest)
            print(f"[Ingest] Warning: {res}")
        else:
            raw_docs.extend(res)

    if not raw_docs:
        print("[Ingest] No documents fetched – nothing to do.")
        return

    # --------------------------------------------------------------------
    # 2. Chunk & embed
    # --------------------------------------------------------------------
    texts, metas = _chunk_documents(raw_docs)

    # --------------------------------------------------------------------
    # 3. Upsert into Pinecone via the shared pipeline helper
    # --------------------------------------------------------------------
    await ingest_documents(texts, metas)

    print(f"[Ingest] Upserted {len(texts)} chunks into Pinecone.")


# ---------------------------------------------------------------------------
# CLI helper (python -m ishtar_ai.data_ingestion.ingest "query text")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ishtar_ai.data_ingestion.ingest '<query>'")
        sys.exit(1)

    q = sys.argv[1]

    asyncio.run(ingest_external_sources(q))
