FROM python:3.11-slim

# Add metadata
LABEL org.opencontainers.image.title="Ishtar AI"
LABEL org.opencontainers.image.description="AI-powered insights for media and journalism"
LABEL org.opencontainers.image.vendor="Ishtar AI Initiative"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy poetry configuration
COPY pyproject.toml poetry.lock* /app/

# Configure poetry to not use virtualenvs inside Docker
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY . /app/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Default command
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 