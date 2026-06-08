FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsqlite3-dev && rm -rf /var/lib/apt/lists/*

# 从仓库根目录构建，所以路径要加 rag-service/ 前缀
COPY rag-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag-service/ .

RUN mkdir -p /app/data /app/logs /app/chroma_db

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
