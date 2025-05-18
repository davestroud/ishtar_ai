from fastapi import FastAPI
from .schemas import QueryRequest, QueryResponse
from ..rag.pipeline import query_pipeline

app = FastAPI(title="Ishtar AI")

@app.post("/query", response_model=QueryResponse)
async def query_ishtar(req: QueryRequest):
    response_text = await query_pipeline(req.query)
    return QueryResponse(answer=response_text)
