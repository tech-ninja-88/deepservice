# DeepService 线上部署指南

> 将 DeepService 企业级智能客服系统部署到线上环境。

---

## 🗂 部署架构

```
┌─────────────────────────────────────────────────────┐
│                    用户浏览器                         │
│              https://deepservice.vercel.app          │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
    ┌─────▼──────┐        ┌──────▼──────┐
    │  Vercel    │        │  Render     │
    │  Frontend  │  ───→  │  Backend    │
    │  Next.js   │  SSE   │  FastAPI    │
    │  (免费)    │        │  (免费)     │
    └────────────┘        └──────┬──────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐            ┌──────▼──────┐
              │ DeepSeek   │            │  ChromaDB   │
              │ API        │            │  (Render     │
              │ (外部)     │            │   Disk)      │
              └────────────┘            └─────────────┘
```

---

## 📋 部署前准备

### 1. 获取 API Keys

| 服务 | 用途 | 获取地址 | 免费额度 |
|------|------|----------|----------|
| **DeepSeek** | 对话生成 + Embedding | https://platform.deepseek.com/api_keys | 注册赠送额度 |
| **OpenAI** (可选) | Embedding 向量化 | https://platform.openai.com/api-keys | $5 免费额度 |
| **Vercel** | 前端托管 | https://vercel.com (GitHub 登录) | 100小时/月免费 |
| **Render** | 后端 API 托管 | https://render.com (GitHub 登录) | 750小时/月免费 |

### 2. 准备知识库文档

将企业文档放入 `rag-service/knowledge_base/` 目录：

```bash
cd deepservice/rag-service/knowledge_base/
# 放入你的 Markdown/PDF/Word 文档
# 示例：
# - product_faq.md      (产品FAQ)
# - return_policy.md    (退换货政策)
# - vip_benefits.md     (VIP权益)
```

### 3. 本地测试

```bash
# 后端
cd deepservice/rag-service
pip install -r requirements.txt
python api_server.py --seed   # 初始化知识库 + 启动API

# 前端 (新终端)
cd deepservice/frontend
npm install
npm run dev                   # 访问 http://localhost:3000
```

---

## 🚀 方案一：Vercel + Render（推荐 — 全免费）

### Step 1: 后端部署到 Render

**1.1 创建 GitHub 仓库**

```bash
cd deepservice
git init
git add .
git commit -m "DeepService initial commit"
git remote add origin https://github.com/YOUR_USERNAME/deepservice.git
git push -u origin main
```

**1.2 在 Render 创建 Web Service**

1. 登录 https://render.com (GitHub 账号)
2. 点击 **New + → Web Service**
3. 连接 GitHub 仓库 `YOUR_USERNAME/deepservice`
4. 配置：

```
Name:           deepservice-api
Environment:    Python 3
Region:         Singapore (亚洲延迟低)
Branch:         main
Root Directory: rag-service
Build Command:  pip install -r requirements.txt
Start Command:  uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

5. **环境变量** (Environment → Add Environment Variable)：

```
DEEPSEEK_API_KEY=sk-your-deepseek-key
OPENAI_API_KEY=sk-your-openai-key   # 可选，用于 Embedding
REDIS_URL=                           # 留空使用内存存储
PORT=8000
```

6. 选择 **Free** 计划
7. 点击 **Create Web Service**

部署成功后会得到 URL：`https://deepservice-api.onrender.com`

**⚠️ Render 休眠问题**：免费实例 15 分钟无请求会休眠，下次请求需 30-60 秒唤醒。
**解决方法**：使用 [UptimeRobot](https://uptimerobot.com) 每 10 分钟 ping 一次 `/health`。

### Step 2: 前端部署到 Vercel

**2.1 在 Vercel 导入项目**

1. 登录 https://vercel.com (GitHub 账号)
2. 点击 **Add New → Project**
3. 导入 `YOUR_USERNAME/deepservice`
4. 配置：

```
Framework:        Next.js
Root Directory:   frontend
Build Command:    npm run build
Output Directory: .next
```

5. **环境变量**：

```
NEXT_PUBLIC_API_URL=https://deepservice-api.onrender.com
```

6. 点击 **Deploy**

部署成功后会得到：`https://deepservice.vercel.app`

### Step 3: 验证部署

```bash
# 测试后端健康检查
curl https://deepservice-api.onrender.com/health

# 测试知识库检索
curl "https://deepservice-api.onrender.com/api/knowledge/search?query=退货"

# 在浏览器打开
open https://deepservice.vercel.app
```

---

## 🐳 方案二：Docker + 云服务器（更稳定）

### Dockerfile (后端)

创建 `deepservice/rag-service/Dockerfile`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# 知识库数据
RUN mkdir -p /app/data /app/logs /app/chroma_db

EXPOSE 8000

# 启动命令
CMD ["sh", "-c", "python api_server.py --seed && uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### docker-compose.yml

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
      - ./rag-service/data:/app/data
      - ./rag-service/chroma_db:/app/chroma_db
      - ./rag-service/knowledge_base:/app/knowledge_base
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
    depends_on:
      - api
```

### 部署到云服务器

```bash
# 在服务器上
git clone https://github.com/YOUR_USERNAME/deepservice.git
cd deepservice

# 设置环境变量
export DEEPSEEK_API_KEY=sk-xxx
export OPENAI_API_KEY=sk-xxx

# 启动
docker compose up -d --build

# 配置 Nginx 反向代理 (可选)
# /etc/nginx/sites-available/deepservice
```

**推荐云服务器**：
- **阿里云 ECS**：2核4G 约 ¥68/月
- **腾讯云轻量**：2核2G 约 ¥58/月
- **AWS Lightsail**：$5/月（新加坡区域）

---

## ⚠️ 常见问题与解决方案

### 1. Vercel 504 超时

**问题**：流式对话超过 60 秒被 Vercel 截断

**原因**：Vercel Serverless Functions 最大执行时间为 60 秒（Hobby 计划）

**解决方案**：

方案 A：使用 Edge Functions（30s 限制，但更快）
```javascript
// next.config.js
export const config = { runtime: 'edge' };
```

方案 B：**推荐** — 前端直连后端 API，绕过 Vercel 代理
```typescript
// 修改 src/lib/api.ts
// 不使用 rewrites，直接指向 Render URL
const API_BASE = "https://deepservice-api.onrender.com";
```

方案 C：升级到 Vercel Pro ($20/月，300s 超时)

### 2. CORS 错误

**问题**：前端访问后端时浏览器报 CORS 错误

**解决**：后端已配置 CORS 中间件，检查环境变量：
```bash
# Render 环境变量中添加
ALLOWED_ORIGINS=https://deepservice.vercel.app
```

### 3. DeepSeek API 调用失败

**问题**：`DEEPSEEK_API_KEY` 无效或余额不足

**解决**：
```bash
# 测试 API Key
curl https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello"}]}'
```

### 4. ChromaDB 数据丢失 (Render)

**问题**：Render 免费实例重启后数据丢失

**解决**：
- 使用 Render Disk（每月 $1 起）
- 或使用 Supabase pgvector（免费 500MB）
- 或每次启动时自动重建索引：`python main.py --seed`

### 5. 移动端输入框被键盘遮挡

**问题**：iOS Safari 上输入框被虚拟键盘遮挡

**解决**：已在 CSS 中添加 `position: fixed` 处理，如仍有问题：
```css
/* globals.css */
@media (max-width: 640px) {
  .chat-input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
  }
}
```

---

## 🔗 部署后 URL 示例

部署完成后，你将拥有：

| 环境 | URL | 说明 |
|------|-----|------|
| **前端** | `https://deepservice.vercel.app` | 用户聊天界面 |
| **后端 API** | `https://deepservice-api.onrender.com` | REST + SSE 接口 |
| **API 文档** | `https://deepservice-api.onrender.com/docs` | Swagger 自动生成 |
| **管理后台** | `https://deepservice.vercel.app/admin` | 对话日志/知识库管理 |

---

## 部署验证清单

部署后确认以下功能正常：

- [ ] 打开前端 URL，能看到聊天界面
- [ ] 输入"你好"，机器人能正常回复
- [ ] 问一个知识库内的问题（如"如何退货"），能看到来源标注
- [ ] 问一个知识库外的问题（如"今天天气"），机器人回复"不确定"
- [ ] 连续多轮对话正常
- [ ] 点击"新对话"能开启新会话
- [ ] 管理后台能正常访问
- [ ] 移动端界面适配正常
