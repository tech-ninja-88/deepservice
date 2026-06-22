"""
DeepService RAG main entry point — orchestrates data_layer, retrieval_layer, generation_layer, and hallucination_guard.
Pipeline: Input -> L1 Safety -> Intent -> Hybrid Retrieval -> L2 Boundary -> L3 Generate -> L4 Validate -> Confidence -> Response.
Usage: python main.py (interactive) | python main.py --seed (init knowledge base)
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
    RAGGenerator, PromptTemplates,
    RAGResponse, ResponseType, DeepSeekClient,
)
from hallucination_guard import (
    HallucinationDefenseSystem, GuardDecision, GuardResult,
    ConfidenceScore, FactCheckResult, FallbackHandler,
)


# >>> RAG Orchestrator
class RAGOrchestrator:
    """
    Orchestrates the full RAG pipeline: safety -> intent -> retrieval -> generation -> validation.
    """

    def __init__(self, scenario: str = "general"):
        """
        Initialize orchestrator. scenario: "general" | "sales" | "it"
        """
        self.scenario = scenario
        self.config = get_config()

        # Initialize subsystems
        self.retrieval_service = RetrievalService()
        self.generator = RAGGenerator(scenario=scenario)
        from intent_recognizer import get_intent_recognizer
        self.classifier = get_intent_recognizer()
        self.defense = HallucinationDefenseSystem()

        # In-memory conversation store; swap for Redis when scaling horizontally
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
        RAG pipeline: Input → [L1: Safety Filter] → [Intent] → [Hybrid Retrieval]
        → [L2: Knowledge Boundary] → [L3: LLM Generation] → [L4: Output Validation]
        → [Confidence Score] → Response
        """
        start_time = time.time()
        conversation_id = conversation_id or self._generate_conversation_id()

        logger.info("=" * 60)
        logger.info(f"[Orchestrator] Processing query: '{user_message[:80]}...'")
        logger.info(f"[Orchestrator] Session: {conversation_id}")

        # Phase 1: Input safety filter (Layer 1)
        logger.info("[Phase 1] Input safety filter...")
        safety_check = self.defense.layer1.check(user_message)

        if safety_check.decision == GuardDecision.BLOCK:
            logger.warning(f"[Phase 1] Input blocked: {safety_check.reason}")
            return self._build_response(
                content=safety_check.suggestion or "抱歉，我无法处理这个请求。",
                conversation_id=conversation_id,
                response_type="blocked",
                confidence=0.0,
                metadata={"guard_layer": 1, "reason": safety_check.reason},
                elapsed=time.time() - start_time,
            )

        sanitized_query = safety_check.sanitized_query or user_message

        # Phase 2: Intent recognition
        logger.info("[Phase 2] Intent recognition...")
        recognition = self.classifier.recognize(sanitized_query)
        top = recognition.get_top_intent()
        intent_result = {
            "intent": top.intent.value if top else "unknown",
            "confidence": top.confidence if top else 0.0,
        }
        logger.info(
            f"[Phase 2] Intent: {intent_result['intent']} "
            f"(confidence: {intent_result['confidence']:.2f})"
        )

        # Specific intent handling
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
            # Complaint -> suggest direct human transfer
            return self._build_response(
                content=(
                    "非常理解您的心情，对于给您带来的不便我们深表歉意。\n\n"
                    "为了更快地解决您的问题，我建议您直接联系人工客服。"
                    '回复"人工"即可为您转接。'
                ),
                conversation_id=conversation_id,
                response_type="escalate",
                confidence=0.9,
                metadata={"intent": "complaint", "suggest_transfer": True},
                elapsed=time.time() - start_time,
            )

        # Phase 3: Hybrid retrieval
        logger.info("[Phase 3] Hybrid retrieval...")
        search_result = self.retrieval_service.search(
            query=sanitized_query,
            strategy="rrf",
            enable_rerank=True,
        )
        logger.info(
            f"[Phase 3] Retrieval complete: {search_result.result_count} results, "
            f"top_similarity={search_result.top_similarity:.3f}"
        )

        # Phase 4: Knowledge boundary check (Layer 2)
        logger.info("[Phase 4] Knowledge boundary check...")
        boundary_check = self.defense.layer2.check(sanitized_query, search_result)

        if boundary_check.decision in (GuardDecision.FALLBACK, GuardDecision.ESCALATE):
            logger.warning(f"[Phase 4] Knowledge boundary triggered: {boundary_check.decision.value}")
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

        # Phase 5: RAG generation (part of Layer 3 defense)
        logger.info("[Phase 5] RAG generation...")

        if stream:
            # Streaming mode: return generator handle
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

        # Sync mode
        # Get conversation memory
        summary, recent = self._get_conversation_context(conversation_id)

        rag_response = self.generator.generate(
            user_query=sanitized_query,
            search_result=search_result,
            conversation_summary=summary,
            recent_messages=recent,
            stream=False,
        )

        logger.info(
            f"[Phase 5] Generation complete: type={rag_response.response_type.value}, "
            f"confidence={rag_response.confidence:.3f}"
        )

        # Phase 6: Output validation (Layer 3)
        logger.info("[Phase 6] Output validation...")
        validation = self.defense.layer3.validate(
            rag_response.content,
            search_result,
        )
        logger.info(
            f"[Phase 6] Validation: is_factual={validation.is_factual}, "
            f"risk={validation.hallucination_risk:.3f}"
        )

        # Phase 7: Confidence scoring
        logger.info("[Phase 7] Confidence scoring...")
        confidence = self.defense.score_confidence(
            search_result,
            rag_response.content,
            validation,
        )
        logger.info(f"[Phase 7] Overall confidence: {confidence.overall:.3f}")

        # Phase 8: Save conversation turn
        self._save_conversation_turn(
            conversation_id, user_message, rag_response.content,
            intent_result, confidence,
        )

        # Phase 9: Build final response
        elapsed = time.time() - start_time
        logger.info(f"[Phase 9] Total elapsed: {elapsed:.2f}s")

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

        recognition = self.classifier.recognize(sanitized)
        top = recognition.get_top_intent()
        intent_result = {
            "intent": top.intent.value if top else "unknown",
            "confidence": top.confidence if top else 0.0,
        }
        yield {
            "event": "intent",
            "data": json.dumps({"intent": intent_result["intent"], "confidence": intent_result["confidence"]}),
        }

        search_result = self.retrieval_service.search(query=sanitized, strategy="rrf")

        # Phase 4: Knowledge boundary
        boundary_check = self.defense.layer2.check(sanitized, search_result)
        if boundary_check.decision in (GuardDecision.FALLBACK, GuardDecision.ESCALATE):
            fallback = self.defense.layer4.get_response(boundary_check, sanitized)
            yield {"event": "token", "data": json.dumps({"content": fallback})}
            yield {"event": "done", "data": json.dumps({"type": "fallback"})}
            return

        # Phase 5: Streaming generation
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

            # Phase 6: Output validation
            validation = self.defense.layer3.validate(full_content, search_result)
            confidence = self.defense.score_confidence(search_result, full_content, validation)

            # Send metadata
            yield {
                "event": "metadata",
                "data": json.dumps({
                    "confidence": confidence.overall,
                    "hallucination_risk": validation.hallucination_risk,
                    "intent": intent_result["intent"],
                }),
            }

            # Save conversation
            self._save_conversation_turn(
                conversation_id, user_message, full_content,
                intent_result, confidence,
            )

            yield {"event": "done", "data": json.dumps({"conversation_id": conversation_id})}

        except Exception as e:
            logger.error(f"[Orchestrator] Streaming error: {e}")
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    # >>> Helper methods

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
        """Build a standardized API response"""
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
        """Retrieve conversation context memory"""
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
        """Save one turn of conversation"""
        if conversation_id not in self._conversation_store:
            self._conversation_store[conversation_id] = []

        self._conversation_store[conversation_id].extend([
            {"role": "user", "content": user_message, "timestamp": time.time()},
            {"role": "assistant", "content": assistant_response, "timestamp": time.time(),
             "intent": intent_result["intent"], "confidence": confidence.overall},
        ])

        # Trigger summary generation
        total_rounds = len(self._conversation_store[conversation_id]) // 2
        if total_rounds >= self.config.app.summary_trigger_rounds:
            self._generate_summary_async(conversation_id)

    def _generate_summary_async(self, conversation_id: str):
        """Generate conversation summary asynchronously (use Celery/BackgroundTasks in production)"""
        history = self._conversation_store.get(conversation_id, [])
        if len(history) < 10:
            return

        try:
            llm = DeepSeekClient()
            messages_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '客服'}: {m['content'][:300]}"
                for m in history[:-4]  # keep last 2 turns unsummarized
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
            logger.debug(f"[Orchestrator] Summary generated: {conversation_id}")
        except Exception as e:
            logger.error(f"[Orchestrator] Summary generation failed: {e}")

    @staticmethod
    def _generate_conversation_id() -> str:
        """Generate a conversation ID"""
        import uuid
        return f"conv_{uuid.uuid4().hex[:12]}"


# >>> Seed knowledge base
def seed_knowledge_base():
    """
    Initialize sample knowledge base for quick start and RAG validation.
    """
    logger.info("Initializing sample knowledge base...")

    store = VectorStoreManager()

    # Check if already initialized
    stats = store.get_collection_stats()
    if stats["total_chunks"] > 0:
        logger.info(f"Knowledge base already has {stats['total_chunks']} chunks, skipping init")
        logger.info("To re-initialize, call store.reset_collection() first")
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

    # Reload BM25 index
    from retrieval_layer import RetrievalService
    RetrievalService().rebuild_bm25_index()

    logger.info(f"Sample knowledge base initialized: {store.get_collection_stats()}")


# >>> CLI interactive mode
def interactive_mode():
    """
    CLI interactive mode for quick RAG testing.
    Commands: /help, /stats, /reset, /exit.
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

        # Handle commands
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

        # Handle human transfer request
        if user_input.strip() == "人工":
            print("\n🤖 DeepService: 正在为您转接人工客服...")
            print("   预计等待时间: 约 30 秒")
            print("   请稍候，人工坐席将接入此对话。\n")
            continue

        # Handle normal query
        print("\n🤖 DeepService: ", end="", flush=True)

        try:
            result = orchestrator.process_query(
                user_message=user_input,
                conversation_id=conversation_id,
            )
            conversation_id = result["conversation_id"]

            # Print result
            print(result["content"])
            print()

            # Print metadata
            meta = result["metadata"]
            intent = meta.get("intent", "unknown")
            confidence = result["confidence"]
            elapsed = result["elapsed_seconds"]

            info_parts = [f"意图: {intent}", f"置信度: {confidence:.2f}"]
            if result.get("sources"):
                info_parts.append(f"参考来源: {len(result['sources'])} 条")
            info_parts.append(f"耗时: {elapsed:.2f}s")

            # Confidence indicator
            if confidence < 0.5:
                confidence_mark = "⚠️"
            elif confidence < 0.7:
                confidence_mark = "🟡"
            else:
                confidence_mark = "🟢"

            print(f"  {confidence_mark} {' | '.join(info_parts)}")

            # Low confidence warning
            if confidence < 0.5:
                print("  💡 此回答置信度较低，建议确认后使用或联系人工客服。")

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            print(f"\n⚠️ 抱歉，处理您的问题时遇到错误。请稍后再试。\n")


# >>> CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeepService RAG — CLI & Seed Utility")
    parser.add_argument("--seed", action="store_true", help="Initialize sample knowledge base")
    parser.add_argument("--scenario", default="general", help="Scenario: general|sales|it")

    args = parser.parse_args()

    # Check environment variables
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

    # Initialize knowledge base
    if args.seed:
        logger.info("Initializing sample knowledge base...")
        seed_knowledge_base()

    # CLI interactive mode (for API server: uvicorn api_server:app)
    if args.seed:
        print("\n✅ 示例知识库已初始化。现在可以提问了！\n")
    interactive_mode()
