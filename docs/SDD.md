# DeepService 系统设计文档 (SDD)

> 版本 V1.0 | 2026-06-06

---

## 一、总体架构设计

### 1.1 架构全景图

```
                              ┌──────────────────────────────┐
                              │        CDN / DNS              │
                              │   deepservice.vercel.app      │
                              └──────────────┬───────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
              ┌─────▼─────┐          ┌──────▼──────┐          ┌──────▼──────┐
              │  Vercel    │          │   Render     │          │  Supabase   │
              │  Next.js   │  SSE/    │   FastAPI    │  SQL/    │  PostgreSQL │
              │  Frontend  │◄─REST──►│   Backend    │◄───────►│  + pgvector │
              │  (Edge)    │          │   (Docker)   │          │  (Managed)  │
              └────────────┘          └──────┬───────┘          └─────────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                        ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
                        │ DeepSeek  │  │  Upstash   │  │  MinIO    │
                        │ API       │  │  Redis     │  │  Storage  │
                        │ (LLM/Emb) │  │  (Cache)   │  │  (Files)  │
                        └───────────┘  └───────────┘  └───────────┘
```

### 1.2 分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     接入层 (Access Layer)                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ Web Client │ │ Mobile Web │ │ REST API   │ │ WebSocket    │  │
│  │ (React)    │ │ (PWA)      │ │ (External) │ │ (Agent)      │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                     网关层 (Gateway Layer)                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ 限流       │ │ 认证/鉴权  │ │ 请求路由    │ │ 日志/追踪    │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                     业务服务层 (Business Layer)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │ 对话编排器    │ │ 意图路由器    │ │ 人工转接服务  │              │
│  │ Orchestrator │ │ Router       │ │ TransferSvc  │              │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘              │
│         │                │                │                       │
│  ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐              │
│  │ 会话管理器    │ │ RAG 引擎     │ │ FSM 状态机   │              │
│  │ SessionMgr   │ │ RAG Engine   │ │ StateMachine │              │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘              │
│         │                │                │                       │
│  ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐              │
│  │ 上下文管理器  │ │ 幻觉防护系统  │ │ 意图识别器   │              │
│  │ ContextMgr   │ │ HalluGuard   │ │ IntentRecog  │              │
│  └──────────────┘ └──────────────┘ └──────────────┘              │
├──────────────────────────────────────────────────────────────────┤
│                     LLM 网关层 (LLM Gateway)                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ 统一调用    │ │ 自动重试    │ │ 成本追踪    │ │ Token 计数   │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                     数据层 (Data Layer)                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │ PostgreSQL │ │  Redis     │ │  ChromaDB  │ │  MinIO/S3    │  │
│  │ (业务数据) │ │ (缓存/会话)│ │ (向量)     │ │  (文件)      │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 核心数据流

```
用户消息
  │
  ▼
[API Gateway] 限流检查 → 参数校验
  │
  ▼
[InputSafetyFilter] ←── 第1层幻觉防御
  │ 通过
  ▼
[IntentRecognizer] 规则匹配(60%) → LLM分类(35%) → 未知(5%)
  │
  ▼
[ContextManager] 构建记忆上下文（短期窗口 + 长期摘要 + 用户画像）
  │
  ▼
[KnowledgeBoundary] ←── 第2层幻觉防御
  │ 通过(相似度 > 阈值)
  ▼
[HybridRetriever] 向量检索 + BM25 → RRF融合 → 重排序
  │
  ▼
[RAGGenerator] Prompt构建 → DeepSeek生成(流式)
  │
  ▼
[OutputValidator] ←── 第3层幻觉防御（事实核查）
  │
  ▼
[ConfidenceScorer] 多维度置信度评分
  │
  ▼
[FallbackHandler] ←── 第4层幻觉防御（低置信度兜底）
  │
  ▼
最终响应 → SSE 流式返回前端
```

---

## 二、技术选型与理由

| 层级 | 技术 | 版本 | 选型理由 |
|------|------|------|----------|
| **前端框架** | Next.js 14 | App Router | SSR/SSG/API Routes 一体化，Vercel 原生优化 |
| **UI 样式** | Tailwind CSS | 3.4 | 原子化 CSS，开发效率高，无运行时开销 |
| **状态管理** | Zustand | 4.5 | 轻量（<1KB），API 简洁，TS 友好 |
| **后端框架** | FastAPI | 0.112+ | 异步高性能，自动 Swagger，Python 生态 |
| **LLM API** | DeepSeek Chat | — | 中文能力强，性价比高（约为 GPT-4 的 1/20） |
| **Embedding** | OpenAI text-embedding-3-small | — | 1536维，$0.02/1M tokens，效果稳定 |
| **向量数据库** | ChromaDB | 0.5+ | 开源、Python 原生、零配置、支持持久化 |
| **关键词检索** | rank-bm25 | 0.2 | BM25 经典算法，无需额外服务 |
| **数据库** | PostgreSQL (Supabase) | 15 | 免费 500MB，内置 pgvector |
| **缓存** | Redis (Upstash) | — | Serverless 友好，免费 256MB |
| **部署前端** | Vercel | Hobby | 100小时/月免费，全球 CDN |
| **部署后端** | Render | Free | 750小时/月，Docker 支持 |
| **文档解析** | LangChain + pypdf + python-docx | — | 多格式统一接口 |

---

## 三、核心模块详细设计

### 3.1 对话编排器 (DialogueOrchestrator)

**设计模式**：Pipeline + Strategy

```
class DialogueOrchestrator:
    """
    9步处理流水线
    """
    def process(user_message, conversation_id) → DialogueResponse:
        1. session = SessionManager.get_or_create(conversation_id)
        2. if TransferDetector.check(user_message) → handle_transfer()
        3. recognition = IntentRecognizer.recognize(user_message)
        4. if recognition.is_multi_intent → decompose()
        5. memory = ContextManager.build_memory(conversation_id)
        6. if StateTracker.is_in_structured_flow() → handle_flow_step()
        7. decision = IntentRouter.route(recognition)
        8. response = RouteExecutor.execute(decision)
        9. SessionManager.save_message() → update_profile() → return response
```

**关键设计决策**：
- 每步独立可测试，可通过依赖注入替换任一环节
- 流式输出使用 Generator 模式，内存友好
- 错误隔离：任一步骤失败不影响其他步骤

### 3.2 RAG 引擎架构

```
                    ┌──────────────────────┐
                    │     RAG 引擎总览      │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────▼─────┐      ┌──────▼──────┐      ┌──────▼──────┐
    │  数据层    │      │   检索层     │      │   生成层     │
    │ DataLayer │ ───► │RetrievalLayer│ ───► │GenerationLayer│
    └───────────┘      └─────────────┘      └─────────────┘
         │                    │                     │
    ┌────▼────┐        ┌─────▼──────┐        ┌─────▼──────┐
    │文档解析  │        │ 向量检索    │        │ Prompt构建 │
    │语义分块  │        │ BM25检索    │        │ 来源标注   │
    │向量化    │        │ RRF融合     │        │ 流式输出   │
    │ChromaDB │        │ 重排序      │        │ 拒答规则   │
    └─────────┘        └────────────┘        └────────────┘
```

### 3.3 幻觉防护系统

四层防御架构的详细实现：

```
Layer 1: InputSafetyFilter
  ├── 敏感词黑名单
  ├── Prompt 注入模式匹配（正则）
  ├── 输入长度限制（防 DoS）
  └── 无意义输入检测

Layer 2: KnowledgeBoundaryGuard
  ├── 检索相似度阈值判断（默认 0.70）
  ├── 检索结果为空 → 直接 FALLBACK
  └── 所有结果低于阈值 → FLAG 标记

Layer 3: OutputValidator (事后事实核查)
  ├── 回答拆分为原子断言
  ├── 逐一在检索结果中验证
  ├── 检测与知识库的矛盾
  └── 计算 hallucinations_risk (0-1)

Layer 4: FallbackHandler
  ├── 不确定性回复模板
  ├── 超出范围回复模板
  ├── 敏感话题回复模板
  └── 技术错误回复模板
```

---

## 四、数据库设计

### 4.1 ER 图

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Conversation │       │   Message    │       │  UserProfile │
├──────────────┤       ├──────────────┤       ├──────────────┤
│ id (PK)      │──┐    │ id (PK)      │       │ user_id (PK) │
│ user_id      │  │    │ conv_id (FK) │◄──────│ preferences  │
│ status       │  │    │ role         │       │ created_at   │
│ created_at   │  └───►│ content      │       └──────────────┘
│ updated_at   │       │ metadata     │
│ message_count│       │ created_at   │
└──────────────┘       └──────┬───────┘
                              │
┌──────────────┐       ┌──────▼───────┐       ┌──────────────┐
│  KnowledgeDoc│       │MsgFeedback   │       │  ApiCallLog  │
├──────────────┤       ├──────────────┤       ├──────────────┤
│ id (PK)      │──┐    │ message_id   │       │ id (PK)      │
│ title        │  │    │ helpful      │       │ model        │
│ content      │  │    │ comment      │       │ tokens       │
│ category     │  └───►│              │       │ cost         │
│ status       │       └──────────────┘       │ latency      │
└──────┬───────┘                              └──────────────┘
       │
┌──────▼───────┐
│KnowledgeChunk│
├──────────────┤
│ id (PK)      │
│ doc_id (FK)  │
│ content      │
│ embedding    │ ← pgvector(1536)
│ chunk_index  │
└──────────────┘
```

### 4.2 Redis 缓存结构

```
# 会话级缓存
session:{id}:info        → Hash    {status, user_id, created_at, ...}
session:{id}:messages    → List    [JSON messages]
session:{id}:summary     → String  对话摘要
session:{id}:entities    → Hash    {order_id, product, ...}

# 限流
ratelimit:{user_id}:chat → String  窗口内请求计数

# 转接队列
transfer:queue           → List    待处理转接请求
```

---

## 五、API 接口规范

### 5.1 对话接口

**POST /api/chat/stream** (核心接口)

```
Request:
{
  "message": "我的订单#12345发货了吗？",
  "conversation_id": "sess_abc123",  // 可选
  "stream": true,
  "user_id": "anonymous"
}

Response (SSE Stream):
event: token
data: {"content":"您好"}

event: token
data: {"content":"，让我帮您查询订单#12345的状态"}

event: metadata
data: {"conversation_id":"sess_abc123","intent":"order_status","confidence":0.92}

event: done
data: {}
```

### 5.2 API 端点清单

| 方法 | 端点 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/health` | 无 | 服务健康检查 |
| `POST` | `/api/chat` | API Key | 同步对话 |
| `POST` | `/api/chat/stream` | API Key | **SSE 流式对话** |
| `GET` | `/api/conversations` | API Key | 会话列表 |
| `GET` | `/api/conversations/{id}` | API Key | 会话详情 |
| `DELETE` | `/api/conversations/{id}` | API Key | 删除会话 |
| `POST` | `/api/conversations/{id}/rate` | API Key | 评价(1-5) |
| `GET` | `/api/knowledge/search` | API Key | 知识库检索 |
| `GET` | `/api/knowledge/documents` | Admin | 文档列表 |
| `POST` | `/api/knowledge/documents` | Admin | 创建文档 |
| `DELETE` | `/api/knowledge/documents/{id}` | Admin | 删除文档 |
| `GET` | `/api/admin/stats` | Admin | 系统统计 |
| `GET` | `/api/admin/logs` | Admin | 对话日志 |

### 5.3 错误响应格式

```json
{
  "detail": "错误描述",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 30
}
```

---

## 六、安全与隐私设计

| 层面 | 措施 |
|------|------|
| **传输安全** | 全链路 HTTPS，HSTS 启用 |
| **API 认证** | API Key + 请求签名（V2），IP 白名单 |
| **数据隔离** | 会话级隔离，conversation_id 随机不可猜测 |
| **敏感数据处理** | 手机号/邮箱日志自动脱敏 `139****1234` |
| **内容安全** | 输入敏感词过滤 + Prompt 注入检测 |
| **依赖安全** | Dependabot 自动更新，定期 `pip audit` |
| **密钥管理** | 环境变量注入，不提交代码仓库 |

---

## 七、系统扩展性设计

### 7.1 水平扩展

```
                    ┌──────────────┐
                    │  Nginx/LB    │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ API #1    │   │ API #2    │   │ API #3    │
    │ (Docker)  │   │ (Docker)  │   │ (Docker)  │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────▼───────┐
                    │ Redis Cluster│ (共享会话状态)
                    └──────────────┘
```

### 7.2 模块化插件机制

```python
# 所有核心组件通过注册表管理，支持运行时替换
class ComponentRegistry:
    intent_recognizer: IntentRecognizer
    retrieval_service: RetrievalService
    llm_client: LLMClient
    
    def register(self, name, component):
        """注册自定义实现"""
```

---

*文档版本 V1.0 | 最后更新 2026-06-06*
