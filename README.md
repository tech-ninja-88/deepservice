#  DeepService — 企业级智能客服中台

> 基于 **DeepSeek 大模型 + RAG** 的企业级智能客服系统  
>  **从根本上解决 AI 幻觉** —— 四层防御体系 + 混合检索 + 知识边界控制

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-14-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.112-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/DeepSeek-Chat-536DFE" alt="DeepSeek">
  <img src="https://img.shields.io/badge/RAG-四层防御-orange" alt="RAG">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
</p>

---

## 📖 项目简介

DeepService 是一款面向中小企业的 **AI 智能客服中台**，基于 DeepSeek 大语言模型和 RAG（检索增强生成）技术构建。核心差异化在于自主研发的 **四层 AI 幻觉防御体系**——从根本上解决大模型"胡说八道"的问题，让 AI 客服做到"不知道就说不知道"。

###  定位

本项目针对 **AI 应用开发工程师 / 智能客服系统开发工程师 / 全栈 AI 工程师** 岗位设计，展示了：

| 能力维度 | 体现 |
|----------|------|
| **系统架构** | 完整的微服务架构设计（前端+后端+RAG+向量数据库） |
| **AI 工程化** | RAG 全链路实现、Prompt 工程、幻觉治理 |
| **产品思维** | PRD、用户画像、功能优先级、KPI 指标体系 |
| **工程质量** | 模块化设计、错误处理、配置管理、日志系统 |
| **文档能力** | 7 份企业级文档（PRD/SDD/测试报告/运维手册等） |

---

##  在线体验

| 环境 | 地址 | 说明 |
|------|------|------|
| **🟢 前端** | [https://deepservice.vercel.app](https://deepservice.vercel.app) | 聊天界面 |
| **🔵 API 文档** | [https://deepservice-api.onrender.com/docs](https://deepservice-api.onrender.com/docs) | Swagger 自动生成的 API 文档 |
| **🟣 管理后台** | [https://deepservice.vercel.app/admin](https://deepservice.vercel.app/admin) | 对话日志 / 知识库管理 |

> ⚠️ 免费 Render 实例 15 分钟无流量会休眠，首次请求需等待 30-60 秒唤醒。

---

##  技术栈

```
前端:  Next.js 14 + TypeScript + Tailwind CSS + Zustand
后端:  Python 3.11 + FastAPI + SSE 流式
AI:    DeepSeek Chat API + RAG (检索增强生成)
检索:  ChromaDB 向量检索 + BM25 关键词检索 + RRF 融合 + LLM 重排序
数据:  PostgreSQL (Supabase) + Redis (Upstash) + pgvector
部署:  Vercel (前端) + Render (后端) + Docker (可选)
文档:  7 份企业级项目文档
```

---

##  核心功能

###  四层幻觉防御（核心亮点）

```
第1层: 输入安全过滤    → 敏感词检测 + Prompt 注入防护
第2层: 知识边界控制    → 检索相似度阈值判断，不在知识库内就拒答
第3层: 输出事后验证    → 回答拆分为原子断言逐一验证
第4层: 安全兜底回复    → "不确定" + 建议转人工
```

###  RAG 检索增强生成

- 混合检索：向量语义检索 + BM25 关键词检索 → RRF 融合
- 重排序：LLM Pairwise 精细打分，过滤低相关度结果
- 来源标注：每条回答可追溯到具体知识来源 `[来源: N]`
- 流式输出：SSE 打字机效果，首 Token 延迟 < 2s

###  多轮对话管理

- 短期记忆：滑动窗口（最近 10 轮完整保留）
- 长期记忆：LLM 自动摘要 + 关键实体跨轮次追踪
- 用户画像：渐进式构建（意图分布、情感倾向、投诉历史）
- 意图追踪：代词消解（"那个订单"→#20240001）

###  结构化流程

- FSM 状态机管理退换货/投诉等多步骤流程
- 槽位填充机制确保信息完整收集
- 支持流程取消/修改/确认

###  人工转接

- 六维度触发检测（主动要求、敏感词、低置信度、负面情感、意图强制、流程失败）
- 对话上下文完整打包（坐席无需重复询问）
- WebSocket 实时消息通道

---

##  快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/deepservice.git
cd deepservice
```

### 2. 启动后端

```bash
cd rag-service
pip install -r requirements.txt

# 配置 API Key
export DEEPSEEK_API_KEY=sk-your-key-here
export OPENAI_API_KEY=sk-your-key-here  # Embedding（可选）

# 初始化知识库 + 启动 API
python api_server.py
# → http://localhost:8000/docs
```

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 4. 开始对话

打开 http://localhost:3000，尝试：
- "如何申请退货？"
- "我的订单#12345发货了吗？"
- "VIP会员有哪些权益？"
- 尝试问一个知识库外的问题（如"今天天气怎么样"），观察拒答效果

---

## 📁 项目结构

```
deepservice/
├── README.md                     ← 你在这里
├── DEPLOYMENT.md                 # 部署指南
├── docs/                         # 企业级文档
│   ├── PRD.md                    # 产品需求文档
│   ├── SDD.md                    # 系统设计文档
│   ├── RAG-Anti-Hallucination.md # RAG 与 AI 幻觉解决方案
│   ├── Prompt-Engineering-Guide.md # Prompt 工程指南
│   ├── Test-Report.md            # 测试报告与评估标准
│   └── Operations-Manual.md      # 部署运维手册
│
├── rag-service/                  # Python 后端 (15 个模块, ~9000 行)
│   ├── api_server.py             # FastAPI 生产级 API
│   ├── dialogue_orchestrator.py  # 对话编排器（9 步流水线）
│   ├── data_layer.py             # 知识库构建（解析/分块/向量化）
│   ├── retrieval_layer.py        # 混合检索（向量+BM25+重排序）
│   ├── generation_layer.py       # RAG 生成（Prompt+来源标注+流式）
│   ├── hallucination_guard.py    # 四层幻觉防护体系
│   ├── session_manager.py        # 会话管理（Redis/内存双后端）
│   ├── context_manager.py        # 多轮上下文记忆
│   ├── intent_recognizer.py      # 意图识别（规则+LLM双层）
│   ├── dialogue_state.py         # FSM 状态机 + 槽位填充
│   ├── router.py                 # 意图路由 + 降级策略
│   ├── human_transfer.py         # 人工转接 + WebSocket
│   ├── config.py                 # 统一配置管理
│   ├── main.py                   # CLI 交互 + FastAPI 入口
│   └── requirements.txt
│
├── frontend/                     # Next.js 14 前端
│   └── src/
│       ├── app/
│       │   ├── page.tsx          # 主聊天页面
│       │   └── admin/            # 管理后台
│       ├── components/chat/       # 聊天组件
│       │   ├── ChatWindow.tsx    # 消息列表 + 欢迎页
│       │   ├── ChatInput.tsx     # 自适应输入框
│       │   ├── MessageBubble.tsx # 消息气泡（Markdown）
│       │   ├── SessionPanel.tsx  # 侧边会话列表
│       │   └── TypingIndicator.tsx # 打字动画
│       ├── hooks/useChat.ts     # 流式对话 Hook
│       ├── stores/chat-store.ts # Zustand 状态管理
│       └── lib/api.ts           # SSE 流式 API 客户端
│
└── .gitignore
```

---

##  演示脚本

### 30 秒开场
> "这是我独立设计开发的企业级智能客服系统 DeepService。它基于 DeepSeek 大模型，核心亮点是四层幻觉防护体系——让 AI 客服'不知道就说不知道'，从根本上解决大模型胡说八道的问题。"

### 2 分钟技术展示
1. **演示正常问答**：问一个知识库内的问题（如"如何退货"），展示流式输出和来源标注
2. **演示知识边界**：问知识库外的问题（如"今天天气"），展示系统拒答
3. **演示多轮对话**：连续追问，展示上下文记忆和实体追踪
4. **演示幻觉防护**：展示低置信度时的黄色标签

### 1 分钟架构阐述
> "系统采用三层 RAG 架构——数据层负责文档解析和语义分块，检索层使用向量+关键词混合检索加 RRF 融合，生成层通过 Prompt 约束和来源标注确保回答可追溯。此外还有四层幻觉防御——从输入安全过滤到知识边界控制，再到输出验证和安全兜底。整个系统前后端分离部署在 Vercel 和 Render 上，全免费。"

---

## 📚 文档索引

| 文档 | 内容 | 用途 |
|------|------|------|
| [PRD](docs/PRD.md) | 产品需求文档 | 展示产品思维 |
| [SDD](docs/SDD.md) | 系统设计文档 | 展示架构能力 |
| [RAG方案](docs/RAG-Anti-Hallucination.md) | AI 幻觉解决方案 | 展示 AI 技术深度 |
| [Prompt指南](docs/Prompt-Engineering-Guide.md) | Prompt 工程指南 | 展示 LLM 应用经验 |
| [测试报告](docs/Test-Report.md) | 测试报告与评估 | 展示质量意识 |
| [运维手册](docs/Operations-Manual.md) | 部署运维手册 | 展示 DevOps 能力 |
| [部署指南](DEPLOYMENT.md) | 线上部署指南 | 实际操作参考 |

---

## 📊 项目数据

| 指标 | 数值 |
|------|------|
| 总代码行数 | ~12,000 行（Python + TypeScript） |
| Python 模块数 | 15 个 |
| React 组件数 | 8 个 |
| API 端点 | 13 个 |
| 单元测试 | 35 个（覆盖 10 个模块） |
| 文档 | 7 份（约 20,000 字） |
| 幻觉率 | < 8%（知识库内问题） |
| 首 Token 延迟 | < 2s |

---

## 📝 License

MIT License — 仅用于个人学习、面试展示。

---

<p align="center">
  <b>DeepService</b> — 让 AI 客服有边界、有依据、可信赖<br>
</p>
