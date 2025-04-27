FROM python:3.10-slim

# Add metadata
LABEL org.opencontainers.image.title="Ishtar AI"
LABEL org.opencontainers.image.description="AI-powered insights for media and journalism"
LABEL org.opencontainers.image.vendor="Ishtar AI Initiative"

WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV OLLAMA_HOST=ollama
ENV OLLAMA_PORT=11434
ENV DEFAULT_MODEL=llama3
ENV LANGCHAIN_PROJECT=default
ENV LANGSMITH_TRACING=true
# LANGCHAIN_API_KEY and OPENAI_API_KEY should be provided at runtime or via docker-compose

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"] 