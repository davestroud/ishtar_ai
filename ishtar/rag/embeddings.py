from __future__ import annotations
import numpy as np
import tiktoken
from typing import List

# Dev-time placeholder embeddings using token hashing.
# Swap with OpenAI/HF embeddings for production.

enc = tiktoken.get_encoding("cl100k_base")

def _hash_tokens(tokens: list[int], dim: int = 1536) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    for i, t in enumerate(tokens):
        vec[(t + i) % dim] += 1.0
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm

def embed_query(text: str, dim: int = 1536) -> np.ndarray:
    tokens = enc.encode(text)[:2048]
    return _hash_tokens(tokens, dim)

def embed_docs(texts: List[str], dim: int = 1536) -> np.ndarray:
    arr = np.vstack([embed_query(t, dim) for t in texts])
    return arr
