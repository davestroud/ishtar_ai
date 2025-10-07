from __future__ import annotations
from typing import List, Dict
from ishtar.rag.embeddings import embed_docs
from ishtar.rag.vectorstore import VectorStore
from .normalize import normalize

def ingest_items(items: List[Dict], vs: VectorStore, batch_size: int = 64):
    normalized = [normalize(x) for x in items]
    texts = [n["text"] for n in normalized]
    ids = [n["id"] for n in normalized]
    metas = [n["meta"] for n in normalized]

    for i in range(0, len(texts), batch_size):
        chunk_texts = texts[i:i+batch_size]
        chunk_ids = ids[i:i+batch_size]
        chunk_metas = metas[i:i+batch_size]
        embs = embed_docs(chunk_texts)
        vs.upsert(chunk_ids, embs, chunk_metas)
