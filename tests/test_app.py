import pytest
from httpx import AsyncClient

from ishtar_ai.app.main import app

@pytest.mark.asyncio
async def test_query_endpoint(monkeypatch):
    async def fake_pipeline(q: str) -> str:
        return "hello"

    monkeypatch.setattr("ishtar_ai.app.main.query_pipeline", fake_pipeline)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/query", json={"query": "test"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": "hello"}
