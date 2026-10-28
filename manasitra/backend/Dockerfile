# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed for compiling some python packages (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first
COPY backend/requirements.txt /app/backend/requirements.txt

# Install PyTorch CPU-only version first to keep the image size small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining backend requirements
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Note: Codebase folders (ai-engine, backend, database) are mounted as volumes in docker-compose.yml to avoid duplicating heavy model weights in Docker image layers.

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV DATABASE_PATH=/app/database/manasitra.db
ENV OLLAMA_URL=http://ollama:11434

# Run the FastAPI app
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]
