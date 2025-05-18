import pytest
from unittest.mock import AsyncMock

from ishtar_ai.rag import pipeline


@pytest.mark.asyncio
async def test_query_pipeline(monkeypatch):
    mock_acall = AsyncMock(return_value="mock result")
    monkeypatch.setattr(pipeline.qa_chain, "acall", mock_acall)
    result = await pipeline.query_pipeline("test")
    assert result == "mock result"
    mock_acall.assert_called_once_with("test")
