import hashlib


def _fallback_id(source: str | None, title: str | None, body: str) -> str:
    if source:
        return source
    if title:
        return title
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
    return f"doc-{digest[:16]}"


def normalize(item: dict) -> dict:
    text = item.get("text") or (item.get("title", "") + "\n\n" + item.get("summary", ""))
    meta = item.get("meta") or {}
    meta.setdefault("source", item.get("link", ""))
    meta.setdefault("title", item.get("title", ""))
    doc_id = item.get("id") or _fallback_id(meta.get("source"), meta.get("title"), text)
    return {
        "id": doc_id,
        "text": text,
        "meta": meta,
    }
