# DeepService 部署运维手册

> 版本 V1.0 | 2026-06-06

---

## 一、环境准备清单

### 1.1 必需服务

| 服务 | 用途 | 获取 |
|------|------|------|
| DeepSeek API Key | 对话生成 | https://platform.deepseek.com/api_keys |
| OpenAI API Key | Embedding（可选） | https://platform.openai.com/api-keys |
| GitHub 仓库 | 代码托管 + CI/CD | https://github.com |
| Vercel 账号 | 前端部署 | https://vercel.com (GitHub 登录) |
| Render 账号 | 后端部署 | https://render.com (GitHub 登录) |

### 1.2 本地开发环境

```bash
# Python 3.11+
python --version

# Node.js 18+
node --version

# 包管理器
pip --version
npm --version
```

---

## 二、部署步骤详解

### 2.1 方案一：Vercel + Render（推荐，全免费）

**Step 1: 推送代码到 GitHub**

```bash
cd deepservice
git init
git add .
git commit -m "DeepService V1.0 — 企业级智能客服系统"
git remote add origin https://github.com/YOUR_USERNAME/deepservice.git
git push -u origin main
```

**Step 2: 部署后端 (Render)**

1. 访问 https://dashboard.render.com
2. New → **Web Service**
3. 连接 GitHub 仓库
4. 填写配置：

```
Name:          deepservice-api
Runtime:       Python 3
Region:        Singapore
Root Dir:      rag-service
Build:         pip install -r requirements.txt
Start:         uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

5. 添加环境变量：

```
DEEPSEEK_API_KEY=sk-xxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxx
PORT=8000
```

6. 选择 **Free Instance** → Create Web Service
7. 等待构建完成（约 3-5 分钟）
8. 获取 URL：`https://deepservice-api.onrender.com`

**Step 3: 部署前端 (Vercel)**

1. 访问 https://vercel.com/new
2. 导入 GitHub 仓库
3. 配置项目：

```
Framework:     Next.js
Root Dir:      frontend
Build:         npm run build
Output:        .next
```

4. 添加环境变量：

```
NEXT_PUBLIC_API_URL=https://deepservice-api.onrender.com
```

5. 点击 **Deploy**
6. 获取 URL：`https://deepservice.vercel.app`

**Step 4: 初始化知识库**

```bash
# 通过 API 导入知识库文档
curl -X POST https://deepservice-api.onrender.com/api/knowledge/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "退换货政策",
    "content": "自签收之日起7天内可申请退货...",
    "category": "售后政策"
  }'
```

### 2.2 方案二：Docker Compose（本地/服务器）

**docker-compose.yml** (项目根目录):

```yaml
version: '3.8'
services:
  api:
    build: ./rag-service
    ports:
      - "8000:8000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./rag-service/chroma_db:/app/chroma_db
      - ./rag-service/knowledge_base:/app/knowledge_base
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
    depends_on:
      - api
```

```bash
# 启动
export DEEPSEEK_API_KEY=sk-xxx
export OPENAI_API_KEY=sk-xxx
docker compose up -d --build

# 查看日志
docker compose logs -f api

# 停止
docker compose down
```

### 2.3 方案三：仅后端部署 + 前端本地

适合面试演示——前端本地运行速度快，后端在线上。

```bash
# 先按方案一部署 Render 后端
# 然后本地运行前端：

cd deepservice/frontend
echo "NEXT_PUBLIC_API_URL=https://deepservice-api.onrender.com" > .env.local
npm install
npm run dev
# 打开 http://localhost:3000
```

---

## 三、配置管理

### 3.1 环境变量清单

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | — | DeepSeek API 密钥 |
| `OPENAI_API_KEY` | — | — | OpenAI Embedding 密钥 |
| `REDIS_URL` | — | `memory://` | Redis 连接串（留空=内存存储） |
| `PORT` | — | `8000` | API 服务端口 |
| `ALLOWED_ORIGINS` | — | `*` | CORS 允许的来源 |
| `LOG_LEVEL` | — | `INFO` | 日志级别 |
| `CHUNK_SIZE` | — | `512` | 知识库分块大小 |
| `SIMILARITY_THRESHOLD` | — | `0.70` | 检索相似度阈值 |

### 3.2 运行时配置更新

部分配置支持运行时通过 API 热更新（无需重启）：

```bash
# 更新相似度阈值
curl -X PUT https://deepservice-api.onrender.com/api/admin/settings \
  -H "Content-Type: application/json" \
  -d '{"retrieval.vector_similarity_threshold": 0.75}'
```

---

## 四、监控告警方案

### 4.1 健康检查

```bash
# 定时 ping 服务（使用 UptimeRobot 或 Cron）
curl https://deepservice-api.onrender.com/health

# 预期响应:
# {"status":"healthy","components":{"session_mgr":"ok","orchestrator":"ok"}}
```

### 4.2 关键监控指标

| 指标 | 告警阈值 | 检查频率 | 处理方式 |
|------|----------|----------|----------|
| API 可用性 | < 99% (月度) | 1 分钟 | 检查服务日志 |
| P95 延迟 | > 8s | 持续 | 检查 LLM API 状态 |
| 错误率 | > 5% | 5 分钟 | 回滚最近部署 |
| 幻觉率 | > 15% | 每日 | 检查知识库质量 |
| DeepSeek API | 不可用 | 实时 | 切换备用模型 |

### 4.3 Render 休眠防止

免费 Render 实例 15 分钟无流量会休眠：

```bash
# 方法1: UptimeRobot 每 10 分钟 ping
# 注册 https://uptimerobot.com → 添加监控 → URL: https://xxx.onrender.com/health

# 方法2: 自建 cron (GitHub Actions)
# .github/workflows/keep-alive.yml
name: Keep Render Alive
on:
  schedule:
    - cron: '*/10 * * * *'
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -s https://deepservice-api.onrender.com/health
```

---

## 五、故障排查指南

### 5.1 常见问题速查表

| 现象 | 可能原因 | 排查步骤 |
|------|----------|----------|
| 前端加载白屏 | API URL 配置错误 | 检查 `NEXT_PUBLIC_API_URL`；浏览器 F12 → Network |
| API 返回 500 | DeepSeek API Key 无效 | `curl /health` 检查组件状态 |
| 回答不准确 | 知识库内容过时/不足 | 检查知识库状态 `GET /api/admin/stats` |
| 响应很慢(>10s) | DeepSeek API 限流 | 检查并发会话数；添加请求队列 |
| CORS 错误 | 来源未在白名单 | 检查 `ALLOWED_ORIGINS` 环境变量 |
| ChromaDB 数据丢失 | 服务重启 | 使用持久化存储或 Render Disk |
| SSE 连接中断 | 超时/网络问题 | 增加客户端重连逻辑 |

### 5.2 调试命令

```bash
# 1. 检查所有组件状态
curl https://deepservice-api.onrender.com/health | jq .

# 2. 测试知识库检索
curl "https://deepservice-api.onrender.com/api/knowledge/search?query=退货&top_k=3" | jq .

# 3. 测试对话（非流式）
curl -X POST https://deepservice-api.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"如何退货？","stream":false}' | jq .

# 4. 查看对话日志
curl "https://deepservice-api.onrender.com/api/admin/logs?limit=10" | jq .

# 5. 查看系统统计
curl https://deepservice-api.onrender.com/api/admin/stats | jq .
```

### 5.3 日志分析

```bash
# 后端日志位置
# Render: Dashboard → deepservice-api → Logs
# Docker: docker compose logs -f api
# 本地: rag-service/logs/deepservice_YYYY-MM-DD.log

# 关键日志模式
grep "ERROR" logs/deepservice_*.log        # 所有错误
grep "hallucination_risk" logs/*.log       # 幻觉风险日志
grep "GuardDecision" logs/*.log            # 防护系统决策日志
```

### 5.4 性能优化建议

| 优化项 | 预期收益 | 实施难度 |
|--------|----------|----------|
| 启用 Redis 缓存会话 | 减少内存占用 60% | 低 |
| 知识库查询结果缓存 | 检索延迟降低 80% | 低 |
| 本地部署 Embedding 模型 | 消除 OpenAI API 延迟 | 中 |
| 异步处理对话摘要 | 减少等待时间 1-2s | 中 |
| API 请求队列限流 | 避免 DeepSeek 429 错误 | 中 |

---

## 六、安全加固清单

- [ ] 生产环境修改 `ALLOWED_ORIGINS` 为具体域名
- [ ] 启用 API Key 认证（`/api/chat` 等接口）
- [ ] 配置 HTTPS 证书（Vercel/Render 默认支持）
- [ ] 定期轮换 API Key
- [ ] 开启 Render 的 IP 白名单
- [ ] 知识库文档去除敏感信息后再导入
- [ ] 日志中手机号/邮箱自动脱敏

---

*文档版本 V1.0 | 最后更新 2026-06-06*
