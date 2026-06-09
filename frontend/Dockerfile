FROM node:20-alpine AS base
WORKDIR /app

FROM base AS deps
COPY package.json ./
RUN npm install

FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# 构建时不需要 API URL —— 运行时通过 /api/config 动态获取
ENV NEXT_PUBLIC_API_URL=""
ENV NEXT_TELEMETRY_DISABLED 1
RUN npm run build

FROM base AS runner
ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
ENV PORT 3000
# API_URL 在 Sealos 环境变量中设置，前端运行时读取
CMD ["node", "server.js"]
