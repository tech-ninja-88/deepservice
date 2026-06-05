"""
=============================================================================
DeepService RAG — 主入口 & 完整调用链
=============================================================================
这是整个 RAG 系统的编排层，整合了所有模块：

  data_layer.py        → 知识库构建（文档解析、分块、向量化）
  retrieval_layer.py   → 混合检索（向量 + BM25 + 重排序）
  generation_layer.py  → RAG 生成（Prompt 构建、流式输出、来源标注）
  hallucination_guard.py → 幻觉防护（四层防御、置信度评分）

完整调用链：
  用户输入 → [第1层:安全过滤] → [意图识别] → [混合检索] → [第2层:知识边界]
  → [第3层:生成回答] → [第4层:输出验证] → [置信度评分] → 返回用户

启动方式：
  python main.py                      # 命令行交互模式
  python main.py --api                # 启动 FastAPI 服务（TODO）
  python main.py --seed               # 初始化示例知识库
=============================================================================
"""

import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)
logger.add(
    Path(__file__).parent / "logs" / "deepservice_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
)

# 导入各模块
from config import get_config
from data_layer import (
    Document, VectorStoreManager, EmbeddingGenerator,
    SemanticChunker, DocumentParserRegistry,
)
from retrieval_layer import RetrievalService, SearchResult, RetrievalResult
from generation_layer import (
    RAGGenerator, IntentClassifier, PromptTemplates,
    RAGResponse, ResponseType, DeepSeekClient,
)
from hallucination_guard import (
    HallucinationDefenseSystem, GuardDecision, GuardResult,
    ConfidenceScore, FactCheckResult, FallbackHandler,
)


# ============================================================================
# 1. 完整的 RAG 编排器
# ============================================================================
class RAGOrchestrator:
    """
    RAG 编排器 — 整合完整调用链

    职责：
      - 串联所有模块的执行顺序
      - 处理各种决策分支（拒答、转人工、正常回答）
      - 记录完整的调用日志（用于分析和审计）
      - 计算和返回置信度

    面试展示要点：
      这个类体现了企业级系统的核心设计理念 ——
      不是简单调用 LLM，而是有多层防护和可控的流程。
    """

    def __init__(self, scenario: str = "general"):
        """
        初始化编排器

        参数:
          scenario: 场景标识 ("general" | "sales" | "it")
        """
        self.scenario = scenario
        self.config = get_config()

        # 初始化各子系统
        self.retrieval_service = RetrievalService()
        self.generator = RAGGenerator(scenario=scenario)
        self.classifier = IntentClassifier()
        self.defense = HallucinationDefenseSystem()

        # 对话记忆（简化版 - 生产环境用 Redis）
        self._conversation_store: Dict[str, List[Dict]] = {}
        self._summary_store: Dict[str, str] = {}

        logger.info(f"[Orchestrator] RAG 编排器初始化完成 (scenario={scenario})")

    def process_query(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        stream: bool = False,
    ) -> Dict:
        """
        ┌──────────────────────────────────────────────────┐
        │          完整的 RAG 查询处理流水线                 │
        │                                                  │
        │  用户输入 message ────► 最终响应 response          │
        │       │                      ▲                   │
        │       ▼                      │                   │
        │  ┌─────────┐    ┌─────────┐   │                   │
        │  │Layer1   │    │Layer4   │   │                   │
        │  │安全过滤 │    │兜底回复 │   │                   │
        │  └────┬────┘    └─────────┘   │                   │
        │       │ (pass)        ▲       │                   │
        │       ▼               │ (block/fallback)           │
        │  ┌─────────┐          │       │                   │
        │  │意图识别 │          │       │                   │
        │  └────┬────┘          │       │                   │
        │       │               │       │                   │
        │       ▼               │       │                   │
        │  ┌─────────┐          │       │                   │
        │  │混合检索 │          │       │                   │
        │  └────┬────┘          │       │                   │
        │       │               │       │                   │
        │       ▼               │       │                   │
        │  ┌─────────┐          │       │                   │
        │  │Layer2   │──────────┘       │                   │
        │  │知识边界 │ (low confidence)  │                   │
        │  └────┬────┘                  │                   │
        │       │ (pass)                │                   │
        │       ▼                       │                   │
        │  ┌─────────┐                  │                   │
        │  │Layer3   │                  │                   │
        │  │LLM生成  │                  │                   │
        │  └────┬────┘                  │                   │
        │       │                       │                   │
        │       ▼                       │                   │
        │  ┌─────────┐                  │                   │
        │  │Layer3   │                  │                   │
        │  │输出验证 │                  │                   │
        │  └────┬────┘                  │                   │
        │       │                       │                   │
        │       ▼                       │                   │
        │  ┌─────────┐                  │                   │
        │  │置信度   │                  │                   │
        │  │评分     │                  │                   │
        │  └────┬────┘                  │                   │
        │       │                       │                   │
        │       ▼                       │                   │
        │  最终响应 ◄────────────────────┘                   │
        └──────────────────────────────────────────────────┘
        """
        start_time = time.time()
        conversation_id = conversation_id or self._generate_conversation_id()

        logger.info("=" * 60)
        logger.info(f"[Orchestrator] 处理查询: '{user_message[:80]}...'")
        logger.info(f"[Orchestrator] 会话: {conversation_id}")

        # ──── Phase 1: 输入安全过滤（第1层防御）────
        logger.info("[Phase 1] 输入安全过滤...")
        safety_check = self.defense.layer1.check(user_message)

        if safety_check.decision == GuardDecision.BLOCK:
            logger.warning(f"[Phase 1] 输入被阻止: {safety_check.reason}")
            return self._build_response(
                content=safety_check.suggestion or "抱歉，我无法处理这个请求。",
                conversation_id=conversation_id,
                response_type="blocked",
                confidence=0.0,
                metadata={"guard_layer": 1, "reason": safety_check.reason},
                elapsed=time.time() - start_time,
            )

        sanitized_query = safety_check.sanitized_query or user_message

        # ──── Phase 2: 意图识别 ────
        logger.info("[Phase 2] 意图识别...")
        intent_result = self.classifier.classify(sanitized_query)
        logger.info(
            f"[Phase 2] 意图: {intent_result['intent']} "
            f"(置信度: {intent_result['confidence']:.2f})"
        )

        # 特定意图处理
        if intent_result["intent"] == "greeting":
            return self._build_response(
                content="您好！我是 DeepService 智能客服助手。请问有什么可以帮您的？",
                conversation_id=conversation_id,
                response_type="greeting",
                confidence=1.0,
                metadata={"intent": "greeting"},
                elapsed=time.time() - start_time,
            )

        if intent_result["intent"] == "complaint":
            # 投诉直接建议转人工
            return self._build_response(
                content=(
                    "非常理解您的心情，对于给您带来的不便我们深表歉意。\n\n"
                    "为了更快地解决您的问题，我建议您直接联系人工客服。"
                    "回复"人工"即可为您转接。"
                ),
                conversation_id=conversation_id,
                response_type="escalate",
                confidence=0.9,
                metadata={"intent": "complaint", "suggest_transfer": True},
                elapsed=time.time() - start_time,
            )

        # ──── Phase 3: 混合检索 ────
        logger.info("[Phase 3] 混合检索...")
        search_result = self.retrieval_service.search(
            query=sanitized_query,
            strategy="rrf",
            enable_rerank=True,
        )
        logger.info(
            f"[Phase 3] 检索完成: {search_result.result_count} 条结果, "
            f"top_similarity={search_result.top_similarity:.3f}"
        )

        # ──── Phase 4: 知识边界检测（第2层防御）────
        logger.info("[Phase 4] 知识边界检测...")
        boundary_check = self.defense.layer2.check(sanitized_query, search_result)

        if boundary_check.decision in (GuardDecision.FALLBACK, GuardDecision.ESCALATE):
            logger.warning(f"[Phase 4] 知识边界触发: {boundary_check.decision.value}")
            fallback_content = self.defense.layer4.get_response(
                boundary_check, sanitized_query
            )
            return self._build_response(
                content=fallback_content,
                conversation_id=conversation_id,
                response_type="uncertain" if boundary_check.decision == GuardDecision.FALLBACK else "escalate",
                confidence=boundary_check.confidence,
                metadata={
                    "guard_layer": 2,
                    "reason": boundary_check.reason,
                    "top_similarity": search_result.top_similarity,
                    "intent": intent_result["intent"],
                },
                sources=[r.to_context_string(i + 1) for i, r in enumerate(search_result.results)],
                elapsed=time.time() - start_time,
            )

        # ──── Phase 5: RAG 生成（第3层防御的一部分）────
        logger.info("[Phase 5] RAG 生成...")

        if stream:
            # 流式模式：返回生成器句柄
            response_data = self._build_response(
                content="",  # 流式内容由上层填充
                conversation_id=conversation_id,
                response_type="knowledge_based",
                confidence=search_result.top_similarity,
                metadata={
                    "intent": intent_result["intent"],
                    "stream": True,
                },
                sources=[r.to_context_string(i + 1) for i, r in enumerate(search_result.results)],
                elapsed=time.time() - start_time,
            )
            # 附加流式生成器信息
            response_data["_stream_context"] = {
                "query": sanitized_query,
                "search_result": search_result,
            }
            return response_data

        # 同步模式
        # 获取对话记忆
        summary, recent = self._get_conversation_context(conversation_id)

        rag_response = self.generator.generate(
            user_query=sanitized_query,
            search_result=search_result,
            conversation_summary=summary,
            recent_messages=recent,
            stream=False,
        )

        logger.info(
            f"[Phase 5] 生成完成: type={rag_response.response_type.value}, "
            f"confidence={rag_response.confidence:.3f}"
        )

        # ──── Phase 6: 输出验证（第3层防御）────
        logger.info("[Phase 6] 输出验证...")
        validation = self.defense.layer3.validate(
            rag_response.content,
            search_result,
        )
        logger.info(
            f"[Phase 6] 验证: is_factual={validation.is_factual}, "
            f"risk={validation.hallucination_risk:.3f}"
        )

        # ──── Phase 7: 置信度评分 ────
        logger.info("[Phase 7] 置信度评分...")
        confidence = self.defense.score_confidence(
            search_result,
            rag_response.content,
            validation,
        )
        logger.info(f"[Phase 7] 综合置信度: {confidence.overall:.3f}")

        # ──── Phase 8: 保存对话记录 ────
        self._save_conversation_turn(
            conversation_id, user_message, rag_response.content,
            intent_result, confidence,
        )

        # ──── Phase 9: 构建最终响应 ────
        elapsed = time.time() - start_time
        logger.info(f"[Phase 9] 总耗时: {elapsed:.2f}s")

        return self._build_response(
            content=rag_response.content,
            conversation_id=conversation_id,
            response_type=rag_response.response_type.value,
            confidence=confidence.overall,
            metadata={
                "intent": intent_result["intent"],
                "intent_confidence": intent_result["confidence"],
                "retrieval_count": search_result.result_count,
                "top_similarity": search_result.top_similarity,
                "hallucination_risk": validation.hallucination_risk,
                "confidence_factors": confidence.factors,
                "citations": [
                    {"index": c.index, "title": c.document_title, "score": c.relevance_score}
                    for c in rag_response.citations
                ],
                "tokens": rag_response.usage,
                "guard_results": {
                    "layer1": safety_check.decision.value,
                    "layer2": boundary_check.decision.value,
                    "layer3_valid": validation.is_factual,
                },
            },
            sources=[r.to_context_string(i + 1) for i, r in enumerate(search_result.results)],
            elapsed=elapsed,
        )

    def process_query_stream(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
    ):
        """
        流式处理查询 — 生成器模式

        用于 SSE (Server-Sent Events) 流式输出。
        每次 yield 一个事件字典。

        用法（FastAPI）:
            @app.post("/api/chat")
            async def chat(request: ChatRequest):
                orchestrator = get_orchestrator()
                return StreamingResponse(
                    orchestrator.process_query_stream(request.message),
                    media_type="text/event-stream",
                )
        """
        conversation_id = conversation_id or self._generate_conversation_id()

        # 发送开始事件
        yield {"event": "start", "data": json.dumps({"conversation_id": conversation_id})}

        # Phase 1-3: 安全过滤 + 意图 + 检索
        safety_check = self.defense.layer1.check(user_message)
        if safety_check.decision == GuardDecision.BLOCK:
            yield {"event": "blocked", "data": json.dumps({"reason": safety_check.reason})}
            return

        sanitized = safety_check.sanitized_query or user_message

        intent_result = self.classifier.classify(sanitized)
        yield {
            "event": "intent",
            "data": json.dumps({"intent": intent_result["intent"], "confidence": intent_result["confidence"]}),
        }

        search_result = self.retrieval_service.search(query=sanitized, strategy="rrf")

        # Phase 4: 知识边界
        boundary_check = self.defense.layer2.check(sanitized, search_result)
        if boundary_check.decision in (GuardDecision.FALLBACK, GuardDecision.ESCALATE):
            fallback = self.defense.layer4.get_response(boundary_check, sanitized)
            yield {"event": "token", "data": json.dumps({"content": fallback})}
            yield {"event": "done", "data": json.dumps({"type": "fallback"})}
            return

        # Phase 5: 流式生成
        summary, recent = self._get_conversation_context(conversation_id)

        try:
            stream_gen = self.generator.generate_stream(
                user_query=sanitized,
                search_result=search_result,
                conversation_summary=summary,
                recent_messages=recent,
            )

            full_content = ""
            for chunk, meta in stream_gen:
                if chunk:
                    full_content += chunk
                    yield {"event": "token", "data": json.dumps({"content": chunk})}
                if meta.get("type") == "done":
                    break

            # Phase 6: 输出验证
            validation = self.defense.layer3.validate(full_content, search_result)
            confidence = self.defense.score_confidence(search_result, full_content, validation)

            # 发送元数据
            yield {
                "event": "metadata",
                "data": json.dumps({
                    "confidence": confidence.overall,
                    "hallucination_risk": validation.hallucination_risk,
                    "intent": intent_result["intent"],
                }),
            }

            # 保存对话
            self._save_conversation_turn(
                conversation_id, user_message, full_content,
                intent_result, confidence,
            )

            yield {"event": "done", "data": json.dumps({"conversation_id": conversation_id})}

        except Exception as e:
            logger.error(f"[Orchestrator] 流式错误: {e}")
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    # ──── 辅助方法 ────

    def _build_response(
        self,
        content: str,
        conversation_id: str,
        response_type: str,
        confidence: float,
        metadata: Dict,
        sources: List[str] = None,
        elapsed: float = 0.0,
    ) -> Dict:
        """构建标准化的 API 响应"""
        return {
            "conversation_id": conversation_id,
            "content": content,
            "response_type": response_type,
            "confidence": round(confidence, 4),
            "metadata": metadata,
            "sources": sources or [],
            "elapsed_seconds": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_conversation_context(
        self,
        conversation_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """获取对话上下文记忆"""
        history = self._conversation_store.get(conversation_id, [])
        summary = self._summary_store.get(conversation_id)

        if history:
            recent = "\n".join(
                f"{'用户' if m['role'] == 'user' else '客服'}: {m['content'][:200]}"
                for m in history[-self.config.app.max_recent_messages:]
            )
        else:
            recent = None

        return summary, recent

    def _save_conversation_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        intent_result: Dict,
        confidence: ConfidenceScore,
    ):
        """保存一轮对话"""
        if conversation_id not in self._conversation_store:
            self._conversation_store[conversation_id] = []

        self._conversation_store[conversation_id].extend([
            {"role": "user", "content": user_message, "timestamp": time.time()},
            {"role": "assistant", "content": assistant_response, "timestamp": time.time(),
             "intent": intent_result["intent"], "confidence": confidence.overall},
        ])

        # 触发摘要生成
        total_rounds = len(self._conversation_store[conversation_id]) // 2
        if total_rounds >= self.config.app.summary_trigger_rounds:
            self._generate_summary_async(conversation_id)

    def _generate_summary_async(self, conversation_id: str):
        """异步生成对话摘要（生产环境用 Celery/BackgroundTasks）"""
        history = self._conversation_store.get(conversation_id, [])
        if len(history) < 10:
            return

        try:
            llm = DeepSeekClient()
            messages_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '客服'}: {m['content'][:300]}"
                for m in history[:-4]  # 保留最近2轮不做摘要
            )
            response = llm.chat(
                messages=[
                    {"role": "system", "content": "请用2-3句话总结以下对话的关键信息，包括：用户的问题、已经给出的解决方案、未解决的问题。"},
                    {"role": "user", "content": messages_text},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            self._summary_store[conversation_id] = response["content"]
            logger.debug(f"[Orchestrator] 对话摘要已生成: {conversation_id}")
        except Exception as e:
            logger.error(f"[Orchestrator] 摘要生成失败: {e}")

    @staticmethod
    def _generate_conversation_id() -> str:
        """生成会话 ID"""
        import uuid
        return f"conv_{uuid.uuid4().hex[:12]}"


# ============================================================================
# 2. 示例数据初始化
# ============================================================================
def seed_knowledge_base():
    """
    初始化示例知识库

    用途：
      - 快速启动和演示
      - 面试时展示 RAG 效果
    """
    logger.info("开始初始化示例知识库...")

    store = VectorStoreManager()

    # 检查是否已经初始化
    stats = store.get_collection_stats()
    if stats["total_chunks"] > 0:
        logger.info(f"知识库已有 {stats['total_chunks']} 个 chunk，跳过初始化")
        logger.info("如需重新初始化，请先调用 store.reset_collection()")
        return

    sample_docs = [
        Document(
            title="退换货政策",
            content=(
                "# 退换货政策\n\n"
                "## 退货条件\n"
                "1. 自签收之日起 **7天** 内，商品未经使用且不影响二次销售，可申请退货。\n"
                "2. 特殊商品（内衣、食品、定制商品）不支持退货。\n"
                "3. 退货商品需包含完整包装、配件和赠品。\n\n"
                "## 换货条件\n"
                "1. 自签收之日起 **15天** 内，商品出现质量问题，可申请换货。\n"
                "2. 换货商品需保持商品完好，附上购买凭证。\n\n"
                "## 退换货流程\n"
                "1. 登录账号，进入'我的订单'页面。\n"
                "2. 选择需退换货的订单，点击'申请售后'。\n"
                "3. 选择退换货原因，上传凭证照片（质量问题需提供清晰照片）。\n"
                "4. 提交后等待客服审核，审核时间为 **1-3个工作日**。\n"
                "5. 审核通过后，按系统指引寄回商品。\n\n"
                "## 运费规则\n"
                "- 质量问题退换货：运费由商家承担（上限15元）。\n"
                "- 非质量问题退货：运费由买家承担。\n"
                "- 换货运费：商家承担寄回商品的运费。\n\n"
                "## 退款时效\n"
                "- 审核通过后，退款在 **3-7个工作日** 内原路返回。\n"
                "- 若超过7个工作日未收到退款，请联��客服。\n"
                "- 退款金额以实际支付金额为准，优惠券部分不退回。\n\n"
                "## 注意事项\n"
                "- 同一订单的多次退款申请将合并处理。\n"
                "- 恶意退货行为将被限制售后服务。\n"
                "- 跨境商品退货流程稍有不同，请参考跨境购说明。\n"
            ),
            source_type="md",
        ),
        Document(
            title="VIP会员权益",
            content=(
                "# VIP会员权益说明\n\n"
                "## 会员等级\n"
                "- **普通会员**：注册即可，享受基础服务。\n"
                "- **银卡会员**：年消费满2000元自动升级。\n"
                "- **金卡会员**：年消费满5000元自动升级。\n"
                "- **钻石会员**：年消费满20000元自动升级。\n\n"
                "## 各等级权益\n\n"
                "### 银卡会员\n"
                "- 全场商品 **9.5折** 优惠\n"
                "- 每月2张免邮券\n"
                "- 专属客服优先接入\n"
                "- 生日当月双倍积分\n\n"
                "### 金卡会员\n"
                "- 全场商品 **9折** 优惠\n"
                "- 每月5张免邮券\n"
                "- 24小时专属客服\n"
                "- 退货免运费（每月2次）\n"
                "- 新品优先试用权\n\n"
                "### 钻石会员\n"
                "- 全场商品 **8.5折** 优惠\n"
                "- 无限免邮\n"
                "- 一对一专属管家\n"
                "- 无限次退货免运费\n"
                "- 每年2次免费定制服务\n"
                "- 线下门店VIP休息室\n\n"
                "## 会员升级规则\n"
                "- 每年1月1日根据上年度消费重新计算等级。\n"
                "- 升级即时生效，降级在次年1月1日生效。\n"
                "- 会员专属活动以实际页面提示为准。\n"
            ),
            source_type="md",
        ),
        Document(
            title="常见技术问题FAQ",
            content=(
                "# 常见技术问题 FAQ\n\n"
                "## 账号相关\n"
                "### Q: 忘记密码怎么办？\n"
                "A: 在登录页面点击'忘记密码'，输入注册手机号获取验证码，验证通过后可设置新密码。\n"
                "密码要求：8-20位，包含字母和数字，不能使用连续或重复字符。\n\n"
                "### Q: 账号被锁定怎么办？\n"
                "A: 连续输错密码5次，账号将被临时锁定30分钟。30分钟后自动解锁。\n"
                "若急需登录，可联系客服人工解锁。\n\n"
                "### Q: 如何修改绑定手机号？\n"
                "A: 进入'账号设置' → '安全设置' → '修改手机号'，需验证当前手机号和新手机号。\n\n"
                "## APP 问题\n"
                "### Q: APP 闪退怎么办？\n"
                "A: 尝试以下方案（按优先级排序）：\n"
                "1. 重启APP\n"
                "2. 检查APP版本是否为最新（应用商店更新）\n"
                "3. 清理手机缓存\n"
                "4. 卸载后重新安装\n"
                "5. 若以上均无效，联系客服并告知手机型号和系统版本。\n\n"
                "### Q: 支付失败怎么办？\n"
                "A: 请检查以下情况：\n"
                "1. 银行卡余额是否充足\n"
                "2. 是否超过单笔/单日支付限额\n"
                "3. 网络连接是否稳定\n"
                "4. 如使用优惠券，检查是否满足使用条件\n"
                "若多次失败，建议更换支付方式或联系银行确认。\n"
            ),
            source_type="md",
        ),
    ]

    store.index_documents(sample_docs)

    # 重新加载 BM25 索引
    from retrieval_layer import RetrievalService
    RetrievalService().rebuild_bm25_index()

    logger.info(f"示例知识库初始化完成: {store.get_collection_stats()}")


# ============================================================================
# 3. 命令行交互模式
# ============================================================================
def interactive_mode():
    """
    命令行交互模式 — 快速测试 RAG 效果

    使用方式:
        python main.py

    支持的命令:
        /help    — 显示帮助
        /stats   — 显示知识库统计
        /reset   — 重置当前会话
        /exit    — 退出
    """
    print("\n" + "=" * 60)
    print("  DeepService — 企业级智能客服系统")
    print("  基于 DeepSeek + RAG 的智能问答")
    print("=" * 60)
    print("  输入问题开始对话，输入 /exit 退出")
    print("  输入 /help 查看更多命令")
    print("=" * 60 + "\n")

    orchestrator = RAGOrchestrator(scenario="general")
    conversation_id = None

    while True:
        try:
            user_input = input("🧑 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 感谢使用，再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()
            if cmd == "exit" or cmd == "quit":
                print("👋 感谢使用，再见！")
                break
            elif cmd == "help":
                print("""
可用命令:
  /help    — 显示此帮助
  /stats   — 显示知识库统计信息
  /reset   — 开始新的对话会话
  /exit    — 退出程序
  人工      — 转接人工客服
                """)
                continue
            elif cmd == "stats":
                store = VectorStoreManager()
                stats = store.get_collection_stats()
                print(f"知识库统计: {stats}\n")
                continue
            elif cmd == "reset":
                conversation_id = None
                print("✅ 已开始新的对话会话\n")
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看帮助\n")
                continue

        # 处理转人工
        if user_input.strip() == "人工":
            print("\n🤖 DeepService: 正在为您转接人工客服...")
            print("   预计等待时间: 约 30 秒")
            print("   请稍候，人工坐席将接入此对话。\n")
            continue

        # 处理正常查询
        print("\n🤖 DeepService: ", end="", flush=True)

        try:
            result = orchestrator.process_query(
                user_message=user_input,
                conversation_id=conversation_id,
            )
            conversation_id = result["conversation_id"]

            # 打印结果
            print(result["content"])
            print()  # 空行

            # 打印元信息
            meta = result["metadata"]
            intent = meta.get("intent", "unknown")
            confidence = result["confidence"]
            elapsed = result["elapsed_seconds"]

            info_parts = [f"意图: {intent}", f"置信度: {confidence:.2f}"]
            if result.get("sources"):
                info_parts.append(f"参考来源: {len(result['sources'])} 条")
            info_parts.append(f"耗时: {elapsed:.2f}s")

            # 颜色标记
            if confidence < 0.5:
                confidence_mark = "⚠️"
            elif confidence < 0.7:
                confidence_mark = "🟡"
            else:
                confidence_mark = "🟢"

            print(f"  {confidence_mark} {' | '.join(info_parts)}")

            # 低置信度提示
            if confidence < 0.5:
                print("  💡 此回答置信度较低，建议确认后使用或联系人工客服。")

        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            print(f"\n⚠️ 抱歉，处理您的问题时遇到错误。请稍后再试。\n")


# ============================================================================
# 4. API 服务入口（FastAPI）
# ============================================================================
def create_app():
    """
    创建 FastAPI 应用

    使用方式:
        python main.py --api
        或
        uvicorn main:app --reload
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError:
        logger.error("请安装 FastAPI 依赖: pip install fastapi uvicorn sse-starlette")
        raise

    app = FastAPI(
        title="DeepService RAG API",
        description="企业级智能客服系统 — 基于 DeepSeek + RAG",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局编排器实例
    orchestrator = RAGOrchestrator()

    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=2000)
        conversation_id: Optional[str] = None
        stream: bool = Field(default=True, description="是否流式输出")

    class ChatResponse(BaseModel):
        conversation_id: str
        content: str
        response_type: str
        confidence: float
        metadata: dict = {}
        elapsed_seconds: float = 0.0

    @app.get("/")
    async def root():
        return {
            "name": "DeepService RAG API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": orchestrator.config.llm.chat_model}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        对话接口 — 非流式模式

        返回完整的 RAG 回答，包含来源标注和置信度。
        """
        try:
            result = orchestrator.process_query(
                user_message=request.message,
                conversation_id=request.conversation_id,
                stream=False,
            )
            return ChatResponse(
                conversation_id=result["conversation_id"],
                content=result["content"],
                response_type=result["response_type"],
                confidence=result["confidence"],
                metadata=result["metadata"],
                elapsed_seconds=result["elapsed_seconds"],
            )
        except Exception as e:
            logger.error(f"API 错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """
        流式对话接口 — SSE 模式

        返回 Server-Sent Events 流。
        """
        async def event_generator():
            try:
                for event in orchestrator.process_query_stream(
                    user_message=request.message,
                    conversation_id=request.conversation_id,
                ):
                    event_type = event.get("event", "message")
                    data = event.get("data", "")
                    yield f"event: {event_type}\ndata: {data}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        """获取会话历史"""
        history = orchestrator._conversation_store.get(conversation_id, [])
        return {"conversation_id": conversation_id, "messages": history}

    @app.get("/api/admin/stats")
    async def get_stats():
        """获取知识库和系统统计"""
        store = VectorStoreManager()
        kb_stats = store.get_collection_stats()
        return {
            "knowledge_base": kb_stats,
            "active_conversations": len(orchestrator._conversation_store),
            "model": orchestrator.config.llm.chat_model,
        }

    @app.post("/api/knowledge/search")
    async def search_knowledge(
        query: str = Query(..., min_length=1),
        top_k: int = Query(default=5, ge=1, le=20),
    ):
        """知识库检索接口"""
        search_result = orchestrator.retrieval_service.search(
            query=query, top_k=top_k, strategy="rrf"
        )
        return {
            "query": query,
            "results": [
                {
                    "content": r.content,
                    "score": r.final_score,
                    "metadata": r.metadata,
                }
                for r in search_result.results
            ],
            "top_similarity": search_result.top_similarity,
            "result_count": search_result.result_count,
        }

    return app


# ============================================================================
# 5. 程序入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepService RAG — 企业级智能客服系统")
    parser.add_argument("--api", action="store_true", help="启动 FastAPI 服务")
    parser.add_argument("--host", default="0.0.0.0", help="API 服务主机")
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口")
    parser.add_argument("--seed", action="store_true", help="初始化示例知识库")
    parser.add_argument("--scenario", default="general", help="场景: general|sales|it")

    args = parser.parse_args()

    # 环境变量检查
    import os as _os
    if not _os.getenv("DEEPSEEK_API_KEY"):
        print("=" * 60)
        print("⚠️  未检测到 DEEPSEEK_API_KEY 环境变量。")
        print("   请设置环境变量或在 .env 文件中配置：")
        print("   DEEPSEEK_API_KEY=your_api_key_here")
        print("   获取 API Key: https://platform.deepseek.com/api_keys")
        print("=" * 60)
        print()
        print("继续以演示模式运行（部分功能需要 API Key）...\n")

    # 初始化知识库
    if args.seed:
        logger.info("初始化示例知识库...")
        seed_knowledge_base()

    # 启动模式选择
    if args.api:
        logger.info(f"启动 FastAPI 服务: http://{args.host}:{args.port}")
        logger.info(f"API 文档: http://{args.host}:{args.port}/docs")
        try:
            import uvicorn
            app = create_app()
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        except ImportError:
            logger.error("FastAPI 未安装。请运行: pip install fastapi uvicorn sse-starlette")
            sys.exit(1)
    else:
        # 默认：命令行交互模式
        if args.seed:
            print("\n✅ 示例知识库已初始化。现在可以提问了！\n")
        interactive_mode()
