def normalize(item: dict) -> dict:
    text = item.get("text") or (item.get("title","") + "\n\n" + item.get("summary",""))
    meta = item.get("meta") or {}
    meta.setdefault("source", item.get("link",""))
    meta.setdefault("title", item.get("title",""))
    return {
        "id": item.get("id") or meta.get("source") or meta.get("title"),
        "text": text,
        "meta": meta,
    }
