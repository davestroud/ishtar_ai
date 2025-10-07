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
    citations = [
        Citation(id=d.get("id",""), source=d.get("meta",{}).get("source",""), score=d.get("score"), metadata=d.get("meta"))
        for d in ctx_docs
    ]
    return ChatResponse(answer=result["final"], citations=citations)
