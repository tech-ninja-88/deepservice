FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsqlite3-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY rag-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ChromaDB ONNX embedding model to avoid first-request timeout
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['test'])"

COPY rag-service/ .

RUN mkdir -p /app/data /app/logs /app/chroma_db

COPY rag-service/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
