FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
RUN pip install -U pip && pip install -e .
COPY . /app
CMD ["python", "-m", "scripts.ingest_seed"]
