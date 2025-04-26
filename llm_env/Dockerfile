FROM python:3.10-slim

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

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"] 