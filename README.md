# Ishtar AI — RAG + Multi-Agent API (Bootstrap)

This is a minimal, production-minded scaffold for the Ishtar AI codebase you described.
It’s tuned for **Cursor** (IDE) + **Codex** workflows: clean module boundaries, small files,
and comments where AI pair-programmers add value.

## Quick start (dev)
```bash
# 1) Create .env (or export env vars)
cp .env.example .env

# 2) Build and run (dev services: api + prometheus)
docker compose -f infra/compose.dev.yml up --build

# 3) Call the API
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json"   -d '{"query":"What is Ishtar AI?", "k": 6}'
```

## Local (no Docker)
```bash
pip install -U pip
pip install -e .
uvicorn apps.api.main:app --reload --port 8000
```

## Repo map
```
ishtar_ai/
  apps/
    api/
      main.py
      deps.py
      schemas.py
  ishtar/
    config/settings.py
    ingestion/
      readers/rss.py
      normalize.py
      pipeline.py
    rag/
      vectorstore.py
      embeddings.py
      retriever.py
      context.py
    agents/
      graph.py
      prompts.py
      tools.py
      policies.py
    llm/
      client.py
      settings.py
    eval/
      ragas_eval.py
      gates.py
    obs/
      tracing.py
      metrics.py
  scripts/
    ingest_seed.py
  infra/
    docker/
      api.Dockerfile
      worker.Dockerfile
    compose.dev.yml
  tests/
    test_rag.py
    test_agents.py
  .env.example
  pyproject.toml
  README.md
```

## Cursor + Codex
- Add a `.cursor/rules` file (optional) to steer Codex (naming, patterns).
- Use small, single-purpose files — Codex follows them very well.
