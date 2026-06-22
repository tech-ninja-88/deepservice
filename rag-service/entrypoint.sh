#!/bin/sh
set -e

echo "==> DeepService Backend Starting..."

# Auto-seed knowledge base if empty
python -c "
from data_layer import VectorStoreManager
store = VectorStoreManager()
stats = store.get_collection_stats()
if stats['total_chunks'] == 0:
    import sys
    sys.path.insert(0, '/app')
    from main import seed_knowledge_base
    seed_knowledge_base()
else:
    print(f'Knowledge base: {stats[\"total_chunks\"]} chunks, skipping seed.')
"

echo "==> Starting API server on port ${PORT:-8000}"
exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}
