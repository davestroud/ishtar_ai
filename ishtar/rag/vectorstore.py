from __future__ import annotations
from typing import List
import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

class VectorStore:
    def upsert(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        raise NotImplementedError
    def search(self, query_emb: np.ndarray, k: int) -> list[dict]:
        raise NotImplementedError

class FAISSStore(VectorStore):
    def __init__(self, dim: int, index_path: str = "data/faiss.index"):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dim) if faiss else None
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.embs = np.empty((0, dim), dtype="float32")

    def upsert(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms
        if self.index:
            self.index.add(embeddings)
        else:
            self.embs = np.vstack([self.embs, embeddings])
        self.ids.extend(ids)
        self.metas.extend(metadatas)

    def search(self, query_emb: np.ndarray, k: int) -> list[dict]:
        q = query_emb.astype("float32")
        q = q / (np.linalg.norm(q) + 1e-8)
        if self.index:
            D, I = self.index.search(q[None, :], k)
            I = I[0]; D = D[0]
        else:
            sims = (self.embs @ q)
            I = np.argsort(-sims)[:k]
            D = sims[I]
        hits = []
        for rank, (i, score) in enumerate(zip(I, D)):
            if i < 0 or i >= len(self.ids):
                continue
            hits.append({
                "id": self.ids[i],
                "score": float(score),
                "meta": self.metas[i],
                "rank": rank,
            })
        return hits

def make_vectorstore(backend: str) -> VectorStore:
    if backend == "faiss":
        return FAISSStore(dim=1536)
    raise ValueError(f"Unsupported backend: {backend}")
