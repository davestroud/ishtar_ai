from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    k: int | None = Field(default=None, description="Top-k to retrieve")

class Citation(BaseModel):
    id: str
    source: str
    score: float | None = None
    metadata: Dict[str, Any] | None = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
