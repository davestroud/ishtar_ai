import pytest

from ishtar_ai.rag import pipeline as pl

@pytest.mark.asyncio
async def test_query_pipeline(monkeypatch):
    async def fake_acall(q):
        return {"result": "foo"}

    monkeypatch.setattr(pl.qa_chain, "acall", fake_acall)
    result = await pl.query_pipeline("hi")
    assert result == "foo"
