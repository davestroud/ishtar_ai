FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install poetry && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-dev

COPY . .

CMD ["uvicorn", "ishtar_ai.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
