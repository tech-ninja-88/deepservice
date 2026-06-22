#!/bin/sh
set -e

# In Sealos: leave NEXT_PUBLIC_API_URL empty so the frontend uses relative URLs
# and Next.js rewrites proxy /api/* to the internal backend service (no CORS needed).
# For direct backend access: set NEXT_PUBLIC_API_URL to the backend's public URL.
API_URL="${NEXT_PUBLIC_API_URL:-}"
if [ -z "$API_URL" ]; then
  echo "==> Using relative API URLs (Next.js rewrite proxy mode)"
else
  echo "==> Configuring API endpoint: $API_URL"
fi

# Replace the placeholder in compiled JS bundles
find /app -name '*.js' -type f -exec sed -i "s|{{NEXT_PUBLIC_API_URL}}|$API_URL|g" {} +

echo "==> Starting frontend server on port ${PORT:-3000}"
exec node server.js
