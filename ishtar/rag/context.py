from __future__ import annotations
from typing import List, Dict, Any

def dedupe(hits: List[Dict[str, Any]], key="id") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hits:
        k = h.get(key)
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out

def compress(docs: List[Dict[str, Any]], budget_tokens: int = 1500) -> List[Dict[str, Any]]:
    # Placeholder: just return the hits. Real impl would token-compress.
    return docs
