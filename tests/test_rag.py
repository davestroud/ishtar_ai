import numpy as np

from ishtar.ingestion.normalize import normalize
from ishtar.rag.vectorstore import FAISSStore


def test_normalize_generates_id_when_missing():
    doc = normalize({"summary": "Test summary", "meta": {}})
    assert doc["id"].startswith("doc-")
    assert doc["text"]


def test_vectorstore_persists_ids_and_metas(tmp_path):
    index_path = tmp_path / "test.index"
    store = FAISSStore(dim=4, index_path=str(index_path))
    store.upsert(
        ["doc-1"],
        np.ones((1, 4), dtype="float32"),
        [{"source": "http://example.com", "title": "Example"}],
    )

    restored = FAISSStore(dim=4, index_path=str(index_path))
    assert restored.ids == ["doc-1"]
    hits = restored.search(np.ones(4, dtype="float32"), k=1)
    assert hits
    assert hits[0]["meta"]["source"] == "http://example.com"
