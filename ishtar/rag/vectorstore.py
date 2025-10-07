from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

logger = logging.getLogger(__name__)

class VectorStore:
    def upsert(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        raise NotImplementedError
    def search(self, query_emb: np.ndarray, k: int) -> list[dict]:
        raise NotImplementedError

class FAISSStore(VectorStore):
    def __init__(self, dim: int, index_path: str = "data/faiss.index"):
        self.dim = dim
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = Path(f"{self.index_path}.meta.json")
        self.emb_path = Path(f"{self.index_path}.emb.npy")
        self.index = faiss.IndexFlatIP(dim) if faiss else None
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.embs = np.empty((0, dim), dtype="float32")
        self._load()

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
        self._persist()

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

    def _load(self):
        if self.index and self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read FAISS index at %s: %s", self.index_path, exc)
        if self.meta_path.exists():
            try:
                with self.meta_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.ids = list(payload.get("ids", []))
                self.metas = list(payload.get("metas", []))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read metadata for FAISS index at %s: %s", self.meta_path, exc)
        if not self.index and self.emb_path.exists():
            try:
                self.embs = np.load(self.emb_path).astype("float32")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to read fallback embeddings at %s: %s", self.emb_path, exc)
                self.embs = np.empty((0, self.dim), dtype="float32")

    def _persist(self):
        try:
            with self.meta_path.open("w", encoding="utf-8") as f:
                json.dump({"ids": self.ids, "metas": self.metas}, f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to write metadata for FAISS index at %s: %s", self.meta_path, exc)
        if self.index:
            try:
                faiss.write_index(self.index, str(self.index_path))  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to write FAISS index to %s: %s", self.index_path, exc)
        else:
            try:
                np.save(self.emb_path, self.embs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to persist fallback embeddings to %s: %s", self.emb_path, exc)


def make_vectorstore(backend: str, index_path: str | None = None) -> VectorStore:
    if backend == "faiss":
        return FAISSStore(dim=1536, index_path=index_path or "data/faiss.index")
    raise ValueError(f"Unsupported backend: {backend}")
