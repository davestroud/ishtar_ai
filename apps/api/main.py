from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from ishtar.config.settings import settings
from apps.api.schemas import ChatRequest, ChatResponse, Citation
from apps.api.deps import get_retriever, get_graph

app = FastAPI(title="Ishtar AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "env": settings.env}

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, retriever=Depends(get_retriever), graph=Depends(get_graph)):
    q = body.query
    ctx_docs = retriever.build_context(q, budget_tokens=settings.max_context_tokens, k=body.k)
    result = graph.invoke({"query": q, "context": ctx_docs})
    answer = result.get("final") or result.get("draft") or ""
    sorted_docs = sorted(
        ctx_docs,
        key=lambda d: (d.get("score") if isinstance(d.get("score"), (int, float)) else 0.0),
        reverse=True,
    )
    citations = []
    for doc in sorted_docs:
        meta = doc.get("meta") or {}
        safe_meta = {k: meta[k] for k in ("source", "title") if meta.get(k)}
        citations.append(
            Citation(
                id=str(doc.get("id", "")),
                source=safe_meta.get("source", ""),
                score=doc.get("score"),
                metadata=safe_meta or None,
            )
        )
    return ChatResponse(answer=answer, citations=citations)
