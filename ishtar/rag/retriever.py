from __future__ import annotations
from ishtar.rag.vectorstore import VectorStore
from ishtar.rag.embeddings import embed_query
from ishtar.rag.context import dedupe, compress
from ishtar.config.settings import settings

class Retriever:
    def __init__(self, vs: VectorStore, k: int = 12, rerank_top_k: int = 20):
        self.vs = vs
        self.k = k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, q: str, k: int | None = None) -> list[dict]:
        q_emb = embed_query(q)
        hits = self.vs.search(q_emb, k or self.rerank_top_k)
        hits = dedupe(hits)
        return hits[: (k or self.k)]

    def build_context(self, q: str, budget_tokens: int = 1500, k: int | None = None):
        docs = self.retrieve(q, k=k)
        return compress(docs, budget_tokens)
