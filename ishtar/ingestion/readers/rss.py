from __future__ import annotations
import feedparser
from typing import List, Dict

def pull_rss(url: str, limit: int = 50) -> List[Dict]:
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:limit]:
        out.append({
            "id": getattr(e, "id", getattr(e, "link", "")),
            "title": e.get("title",""),
            "summary": e.get("summary",""),
            "link": e.get("link",""),
            "published": e.get("published",""),
            "text": f"{e.get('title','')}

{e.get('summary','')}",
            "meta": {"source": e.get("link",""), "title": e.get("title","")},
        })
    return out
