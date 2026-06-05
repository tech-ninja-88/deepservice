"""
=============================================================================
DeepService 对话管理 — 对话编排器 (Dialogue Orchestrator)
=============================================================================
职责：
  1. 整合六大对话管理模块为统一的对话引擎
  2. 提供完整的多轮对话处理流水线
  3. 多意图任务的编排执行

整合的模块：
  session_manager.py     → 会话生命周期管理
  context_manager.py     → 多轮上下文记忆
  intent_recognizer.py   → 意图识别 + 实体抽取
  dialogue_state.py      → FSM 状态机 + 槽位填充
  router.py              → 意图路由分发
  human_transfer.py      → 人工转接

完整处理流水线：
  用户消息
    │
    ▼
  [SessionManager] 获取/创建会话
    │
    ▼
  [TransferDetector] 是否需要转人工？
    │
    ├── 是 → [HumanTransferService] 打包上下文 → 分配坐席
    │
    └── 否
        │
        ▼
  [IntentRecognizer] 意图识别 + 实体抽取
        │
        ▼
  [ContextManager] 构建对话记忆（短期+长期+画像）
        │
        ▼
  [DialogueStateTracker] 当前是否在结构化流程中？
        │
        ├── 是 → 继续槽位填充
        │
        └── 否
            │
            ▼
  [IntentRouter] 意图 → 处理策略路由
        │
        ├── RAG_RETRIEVAL → [RetrievalService + RAGGenerator]
        ├── STRUCTURED_FLOW → [DialogueStateTracker.start_flow]
        ├── TRANSFER_HUMAN → [HumanTransferService]
        ├── RULE_RESPONSE → 直接返回模板
        └── CLARIFY → 追问澄清
            │
            ▼
  [HallucinationDefense] 输出验证 + 置信度评分
            │
            ▼
  [SessionManager] 保存消息 → 返回响应
=============================================================================
"""

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path

from loguru import logger

from config import get_config

# 导入所有对话管理模块
from session_manager import (
    SessionManager, Session, SessionStatus, Message, MessageRole,
    get_session_manager,
)
from context_manager import (
    ContextManager, ConversationMemory, UserProfile,
    get_context_manager,
)
from intent_recognizer import (
    HybridIntentRecognizer, IntentCategory, IntentResult, RecognitionResult,
    get_intent_recognizer,
)
from dialogue_state import (
    DialogueStateTracker, DialogueState, FlowType,
    get_dialogue_state_tracker,
)
from router import (
    IntentRouter, RouteExecutor, RouteAction, RouteDecision,
    get_router, get_route_executor, build_default_handlers,
)
from human_transfer import (
    HumanTransferService, TransferTrigger, TransferContext, TransferStatus,
    get_human_transfer_service,
)


# ============================================================================
# 统一响应格式
# ============================================================================
@dataclass
class DialogueResponse:
    """
    标准化的对话响应

    无论走哪种处理路径，最终都返回此格式。
    """
    # 核心字段
    conversation_id: str
    content: str
    response_type: str                      # knowledge_based / uncertain / greeting / transfer / flow_step
    confidence: float = 1.0

    # 意图信息
    intent: str = ""
    is_multi_intent: bool = False
    intents_detail: List[Dict] = field(default_factory=list)

    # 路由信息
    route_action: str = ""

    # 结构化流程信息
    flow_step: Optional[Dict] = None        # 流程中的步骤信息

    # 转接信息
    transfer_info: Optional[Dict] = None

    # 来源引用
    sources: List[Dict] = field(default_factory=list)

    # 元数据
    entities: List[Dict] = field(default_factory=list)
    tracked_entities: Dict[str, str] = field(default_factory=dict)
    memory_context_used: bool = False
    elapsed_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = {
            "conversation_id": self.conversation_id,
            "content": self.content,
            "response_type": self.response_type,
            "confidence": round(self.confidence, 4),
            "intent": self.intent,
            "is_multi_intent": self.is_multi_intent,
            "metadata": {
                **self.metadata,
                "route_action": self.route_action,
                "memory_used": self.memory_context_used,
            },
        }

        if self.intents_detail:
            result["intents_detail"] = self.intents_detail

        if self.entities:
            result["entities"] = self.entities

        if self.tracked_entities:
            result["tracked_entities"] = self.tracked_entities

        if self.sources:
            result["sources"] = self.sources

        if self.flow_step:
            result["flow_step"] = self.flow_step

        if self.transfer_info:
            result["transfer_info"] = self.transfer_info

        result["elapsed_ms"] = round(self.elapsed_ms, 2)
        return result


# ============================================================================
# 核心编排器
# ============================================================================
class DialogueOrchestrator:
    """
    对话编排器 — 完整的对话处理引擎

    使用示例：
        orchestrator = DialogueOrchestrator()
        response = orchestrator.process("你好，我想退货", conversation_id="...")
        print(response.content)
    """

    def __init__(self):
        self.config = get_config()

        # 初始化所有子系统
        self.session_mgr = get_session_manager()
        self.context_mgr = get_context_manager()
        self.intent_recognizer = get_intent_recognizer()
        self.state_tracker = get_dialogue_state_tracker()
        self.router = get_router()
        self.transfer_service = get_human_transfer_service()

        # 可选的 RAG 依赖（延迟导入避免循环依赖）
        self._retrieval_service = None
        self._rag_generator = None
        self._defense_system = None

        logger.info("[DialogueOrchestrator] 编排器初始化完成")

    @property
    def retrieval_service(self):
        if self._retrieval_service is None:
            from retrieval_layer import RetrievalService
            self._retrieval_service = RetrievalService()
        return self._retrieval_service

    @property
    def rag_generator(self):
        if self._rag_generator is None:
            from generation_layer import RAGGenerator
            self._rag_generator = RAGGenerator()
        return self._rag_generator

    @property
    def defense_system(self):
        if self._defense_system is None:
            from hallucination_guard import HallucinationDefenseSystem
            self._defense_system = HallucinationDefenseSystem()
        return self._defense_system

    def process(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: str = "anonymous",
        channel: str = "web",
        locale: str = "zh-CN",
        **kwargs,
    ) -> DialogueResponse:
        """
        ┌─────────────────────────────────────────────────┐
        │        处理单条用户消息（完整流水线）              │
        │                                                 │
        │  这是整个对话系统的核心入口方法。                  │
        │  面试时可以逐步骤讲解这个方法的处理流程。           │
        └─────────────────────────────────────────────────┘
        """
        start_time = time.time()
        logger.info(f"[Orchestrator] ====== 开始处理消息 ======")
        logger.info(f"[Orchestrator] 用户输入: '{user_message[:80]}...'")

        # ──── Step 1: 会话管理 ────
        session, is_new = self.session_mgr.get_or_create_session(
            session_id=conversation_id,
            user_id=user_id,
            channel=channel,
            locale=locale,
        )
        conversation_id = session.id
        logger.info(
            f"[Step 1] 会话: {conversation_id[:12]}... "
            f"(新会话={is_new}, 第{session.turn_count}轮)"
        )

        # 保存用户消息
        user_msg = Message(role=MessageRole.USER, content=user_message)
        self.session_mgr.append_message(conversation_id, user_msg)

        # ──── Step 2: 转人工检测 ────
        intent = ""
        confidence = 1.0

        # 先做快速意图识别用于转接检测
        quick_result = self.intent_recognizer.recognize(
            user_message, session_id=conversation_id
        )
        if quick_result.intents:
            intent = quick_result.intents[0].intent.value
            confidence = quick_result.intents[0].confidence

        transfer_trigger = self.transfer_service.check_trigger(
            session_id=conversation_id,
            user_message=user_message,
            intent=intent,
            confidence=confidence,
        )

        if transfer_trigger:
            logger.info(f"[Step 2] 触发转人工: {transfer_trigger.value}")
            return self._handle_transfer(
                conversation_id, user_id, transfer_trigger,
                quick_result, start_time,
            )

        # ──── Step 3: 意图识别（完整版）────
        logger.info("[Step 3] 意图识别...")
        recognition = quick_result  # 复用快速识别结果
        primary_intent = recognition.primary_intent
        top_intent_name = primary_intent.intent.value if primary_intent else "unknown"

        logger.info(
            f"[Step 3] 意图: {top_intent_name} "
            f"(置信度: {primary_intent.confidence if primary_intent else 0:.2f})"
        )
        if recognition.is_multi_intent:
            logger.info(
                f"[Step 3] 多意图: {[i.intent.value for i in recognition.intents]}"
            )

        # ──── Step 4: 构建对话记忆 ────
        logger.info("[Step 4] 构建对话记忆...")
        memory = self.context_mgr.build_memory(conversation_id, self.session_mgr)

        # 检查是否有跨轮次实体可用于补充当前查询
        relevant_entities = self.context_mgr.get_relevant_entities_for_query(
            memory, user_message
        )

        if relevant_entities:
            logger.info(
                f"[Step 4] 跨轮次实体: "
                f"{[(e.type.value, e.value) for e in relevant_entities]}"
            )

        # ──── Step 5: 检查是否在结构化流程中 ────
        state = self.state_tracker.fsm.get_current_state(conversation_id)
        logger.info(f"[Step 5] 当前对话状态: {state.value}")

        if self.state_tracker.fsm.is_in_structured_flow(conversation_id):
            # 在流程中，继续处理槽位
            return self._handle_flow_step(
                conversation_id, user_message, recognition, memory, start_time
            )

        # ──── Step 6: 路由决策 ────
        logger.info("[Step 6] 路由决策...")
        route_decisions = self.router.route(recognition, conversation_id)

        if not route_decisions:
            return self._build_fallback_response(
                conversation_id, "无法确定如何处理您的请求。", recognition, start_time
            )

        primary_decision = route_decisions[0]
        logger.info(f"[Step 6] 路由: {primary_decision.action.value}")

        # ──── Step 7: 按路由执行 ────
        logger.info("[Step 7] 执行处理...")
        response = self._execute_route(
            primary_decision,
            user_message=user_message,
            conversation_id=conversation_id,
            recognition=recognition,
            memory=memory,
            start_time=start_time,
        )

        # ──── Step 8: 保存助手消息 ────
        asst_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            metadata={
                "intent": top_intent_name,
                "confidence": response.confidence,
                "route_action": response.route_action,
            },
        )
        self.session_mgr.append_message(conversation_id, asst_msg)

        # ──── Step 9: 更新用户画像 ────
        self.context_mgr.update_profile_from_turn(
            user_id=user_id,
            intent=top_intent_name,
            sentiment=0.5,  # TODO: 接入情感分析
            session_id=conversation_id,
        )

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[Orchestrator] ====== 处理完成 ({elapsed:.0f}ms) ======")

        return response

    def process_stream(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: str = "anonymous",
    ) -> Generator[str, None, DialogueResponse]:
        """
        流式处理（SSE 模式）

        用于实时打字机效果输出。
        """
        # 简化版：先做完整处理，再逐字输出
        # 生产环境应在 RAGGenerator.generate_stream() 层级实现真正的流式
        full_response = self.process(user_message, conversation_id, user_id)

        # 逐块 yield 内容（模拟流式）
        content = full_response.content
        chunk_size = 10  # 每次输出 10 个字符

        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)

        # 最后输出元数据
        yield json.dumps({"type": "metadata", "data": full_response.to_dict()}, ensure_ascii=False)
        yield json.dumps({"type": "done"}, ensure_ascii=False)

        return full_response

    # ──── 路由处理分支 ────

    def _execute_route(
        self,
        decision: RouteDecision,
        **context,
    ) -> DialogueResponse:
        """根据路由决策执行对应的处理"""
        action = decision.action
        conversation_id = context.get("conversation_id", "")
        user_message = context.get("user_message", "")
        recognition = context.get("recognition")
        memory = context.get("memory")
        start_time = context.get("start_time", time.time())

        if action == RouteAction.RAG_RETRIEVAL:
            return self._handle_rag(conversation_id, user_message, recognition, memory, decision, start_time)

        elif action == RouteAction.RAG_ORDER_LOOKUP:
            return self._handle_rag_order(conversation_id, user_message, recognition, memory, decision, start_time)

        elif action == RouteAction.STRUCTURED_FLOW:
            return self._handle_start_flow(conversation_id, user_message, recognition, decision, start_time)

        elif action == RouteAction.RULE_RESPONSE:
            return self._handle_rule_response(conversation_id, decision, recognition, start_time)

        elif action == RouteAction.CLARIFY:
            return self._handle_clarify(conversation_id, user_message, recognition, start_time)

        elif action == RouteAction.TRANSFER_HUMAN:
            return self._handle_transfer(
                conversation_id, context.get("user_id", "anonymous"),
                TransferTrigger.INTENT_FORCED, recognition, start_time,
            )

        elif action == RouteAction.LLM_DIRECT:
            return self._handle_llm_direct(conversation_id, user_message, memory, start_time)

        else:
            return self._build_fallback_response(
                conversation_id,
                "抱歉，我暂时无法处理这类问题。请尝试换个方式描述。",
                recognition,
                start_time,
            )

    def _handle_rag(
        self,
        conversation_id: str,
        user_message: str,
        recognition: RecognitionResult,
        memory: ConversationMemory,
        decision: RouteDecision,
        start_time: float,
    ) -> DialogueResponse:
        """
        RAG 检索生成处理

        集成 hallucination_guard 进行知识边界控制和输出验证。
        """
        # 1. 检索
        search_result = self.retrieval_service.search(
            query=user_message,
            strategy="rrf",
            enable_rerank=True,
        )

        # 2. 知识边界预检
        pre_check = self.defense_system.pre_generation_check(user_message, search_result)
        if pre_check.decision.value in ("block", "fallback"):
            content = self.defense_system.get_fallback_response(pre_check, user_message)
            return DialogueResponse(
                conversation_id=conversation_id,
                content=content,
                response_type="uncertain",
                confidence=pre_check.confidence,
                intent=recognition.get_top_intent().intent.value if recognition.get_top_intent() else "unknown",
                route_action="rag_retrieval",
                elapsed_ms=(time.time() - start_time) * 1000,
                metadata={"fallback_reason": pre_check.reason},
            )

        # 3. 生成（注入上下文记忆）
        memory_context = memory.to_prompt_context()

        rag_response = self.rag_generator.generate(
            user_query=user_message,
            search_result=search_result,
            conversation_summary=memory.summary,
            recent_messages=(
                "\n".join(
                    f"{'用户' if m.role == MessageRole.USER else '客服'}: {m.content[:200]}"
                    for m in memory.recent_messages[-6:]
                ) if memory.recent_messages else None
            ),
        )

        # 4. 输出验证
        validation = self.defense_system.post_generation_validate(
            rag_response.content, search_result
        )
        final_confidence = self.defense_system.score_confidence(
            search_result, rag_response.content, validation
        )

        # 5. 如果幻觉风险高，切换为兜底回复
        if validation.hallucination_risk > 0.5:
            logger.warning(f"[Orchestrator] 幻觉风险过高: {validation.hallucination_risk:.3f}")
            content = self.defense_system.get_fallback_response(
                self.defense_system.layer2.check(user_message, search_result),
                user_message,
            )
        else:
            content = rag_response.content

        return DialogueResponse(
            conversation_id=conversation_id,
            content=content,
            response_type=rag_response.response_type.value,
            confidence=final_confidence.overall,
            intent=recognition.get_top_intent().intent.value if recognition.get_top_intent() else "unknown",
            route_action="rag_retrieval",
            sources=[
                {"index": c.index, "title": c.document_title, "score": c.relevance_score}
                for c in rag_response.citations
            ],
            entities=[e.to_dict() for e in recognition.entities],
            tracked_entities={k: v.value for k, v in memory.tracked_entities.items()},
            memory_context_used=True,
            elapsed_ms=(time.time() - start_time) * 1000,
            metadata={
                "top_similarity": search_result.top_similarity,
                "result_count": search_result.result_count,
                "hallucination_risk": validation.hallucination_risk,
            },
        )

    def _handle_rag_order(self, *args, **kwargs) -> DialogueResponse:
        """RAG + 订单查询（预留数据库查询接口）"""
        # TODO: 集成订单数据库查询
        return self._handle_rag(*args, **kwargs)

    def _handle_start_flow(
        self,
        conversation_id: str,
        user_message: str,
        recognition: RecognitionResult,
        decision: RouteDecision,
        start_time: float,
    ) -> DialogueResponse:
        """
        启动结构化流程
        """
        flow_type = decision.metadata.get("flow_type", FlowType.RETURN_EXCHANGE)
        result = self.state_tracker.start_flow(conversation_id, flow_type)

        return DialogueResponse(
            conversation_id=conversation_id,
            content=result["message"],
            response_type="flow_step",
            confidence=0.95,
            intent=recognition.get_top_intent().intent.value if recognition.get_top_intent() else "",
            route_action="structured_flow",
            flow_step={
                "flow_type": flow_type.value if isinstance(flow_type, FlowType) else flow_type,
                "status": result["status"],
                "next_slot": result.get("next_slot"),
            },
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _handle_flow_step(
        self,
        conversation_id: str,
        user_message: str,
        recognition: RecognitionResult,
        memory: ConversationMemory,
        start_time: float,
    ) -> DialogueResponse:
        """
        处理结构化流程中的用户输入
        """
        result = self.state_tracker.process_user_input(conversation_id, user_message)

        response_type = "flow_step"
        if result["status"] == "flow_completed":
            response_type = "flow_completed"
        elif result["status"] == "flow_cancelled":
            response_type = "knowledge_based"  # 回到自由对话模式
        elif result.get("transfer_human"):
            return self._handle_transfer(
                conversation_id, "anonymous",
                TransferTrigger.FLOW_FAILED, recognition, start_time,
            )

        return DialogueResponse(
            conversation_id=conversation_id,
            content=result["message"],
            response_type=response_type,
            confidence=0.9,
            intent=recognition.get_top_intent().intent.value if recognition.get_top_intent() else "",
            route_action="structured_flow",
            flow_step=result,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _handle_clarify(
        self,
        conversation_id: str,
        user_message: str,
        recognition: RecognitionResult,
        start_time: float,
    ) -> DialogueResponse:
        """追问澄清"""
        return DialogueResponse(
            conversation_id=conversation_id,
            content="抱歉，我没有完全理解您的问题。能否请您再详细描述一下？",
            response_type="clarification",
            confidence=0.3,
            intent=recognition.get_top_intent().intent.value if recognition.get_top_intent() else "unknown",
            route_action="clarify",
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _handle_rule_response(
        self,
        conversation_id: str,
        decision: RouteDecision,
        recognition: RecognitionResult,
        start_time: float,
    ) -> DialogueResponse:
        """规则直接回复"""
        from router import IntentRouter
        content = IntentRouter.RULE_RESPONSES.get(
            decision.intent,
            "您好！请问有什么可以帮您的？",
        )
        return DialogueResponse(
            conversation_id=conversation_id,
            content=content,
            response_type="greeting",
            confidence=1.0,
            intent=decision.intent.value,
            route_action="rule_response",
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _handle_llm_direct(
        self,
        conversation_id: str,
        user_message: str,
        memory: ConversationMemory,
        start_time: float,
    ) -> DialogueResponse:
        """直接 LLM 对话（不经过 RAG）"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
            )
            memory_context = memory.to_prompt_context()
            response = client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": f"你是一个友好的客服助手。\n\n{memory_context}"},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            content = response.choices[0].message.content or ""
            return DialogueResponse(
                conversation_id=conversation_id,
                content=content,
                response_type="chitchat",
                confidence=0.8,
                intent="chitchat",
                route_action="llm_direct",
                memory_context_used=True,
                elapsed_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return self._build_fallback_response(
                conversation_id,
                "抱歉，我暂时无法回答这个问题。",
                None,
                start_time,
                error=str(e),
            )

    def _handle_transfer(
        self,
        conversation_id: str,
        user_id: str,
        trigger: TransferTrigger,
        recognition: Optional[RecognitionResult],
        start_time: float,
    ) -> DialogueResponse:
        """
        执行人工转接
        """
        # 构建上下文
        memory = self.context_mgr.build_memory(conversation_id, self.session_mgr)
        summary = memory.summary or ""
        recent_messages = [
            {
                "role": m.role.value,
                "content": m.content[:200],
                "timestamp": m.timestamp,
            }
            for m in memory.recent_messages[-10:]
        ]
        entities = {k: v.value for k, v in memory.tracked_entities.items()}

        # 发起转接
        transfer = self.transfer_service.initiate_transfer(
            session_id=conversation_id,
            trigger=trigger,
            user_id=user_id,
            conversation_summary=summary,
            recent_messages=recent_messages,
            tracked_entities=entities,
        )

        # 构建用户响应
        if transfer.status == TransferStatus.QUEUED:
            queue_pos = self.transfer_service.agent_manager.get_queue_length()
            content = (
                f"正在为您转接人工客服...\n\n"
                f"当前排队位置：第 {queue_pos} 位\n"
                f"预计等待：约 {queue_pos * 2} 分钟\n\n"
                f"您也可以继续描述问题，我会为您记录。排队期间随时可以取消。"
            )
        else:
            content = (
                f"已为您接通人工客服。\n\n"
                f"坐席 {transfer.agent_name} 将为您服务。"
                f"我将把我们的对话记录发送给坐席，您无需重复描述问题。"
            )

        return DialogueResponse(
            conversation_id=conversation_id,
            content=content,
            response_type="transfer",
            confidence=0.95,
            intent=recognition.get_top_intent().intent.value if recognition and recognition.get_top_intent() else "",
            route_action="transfer_human",
            transfer_info={
                "transfer_id": transfer.transfer_id,
                "trigger": trigger.value,
                "status": transfer.status.value,
                "agent_name": transfer.agent_name,
                "queue_position": self.transfer_service.agent_manager.get_queue_length(),
            },
            elapsed_ms=(time.time() - start_time) * 1000,
        )

    def _build_fallback_response(
        self,
        conversation_id: str,
        content: str,
        recognition: Optional[RecognitionResult],
        start_time: float,
        error: str = "",
    ) -> DialogueResponse:
        """构建兜底响应"""
        return DialogueResponse(
            conversation_id=conversation_id,
            content=content,
            response_type="fallback",
            confidence=0.1,
            intent=recognition.get_top_intent().intent.value if recognition and recognition.get_top_intent() else "unknown",
            route_action="fallback",
            elapsed_ms=(time.time() - start_time) * 1000,
            metadata={"error": error} if error else {},
        )


# ============================================================================
# 多轮对话演示
# ============================================================================
class MultiTurnDemo:
    """
    多轮对话演示 — 用于面试展示

    演示场景：
      1. 基础问答（带实体追踪）
      2. 多意图处理
      3. 退换货流程（FSM + 槽位填充）
      4. 转人工流程
    """

    def __init__(self):
        self.orchestrator = DialogueOrchestrator()

    def run_demo_1_basic_qa_with_context(self):
        """
        演示1：基础问答 + 跨轮次上下文

        展示：
          - 实体跨轮次追踪
          - 对话记忆维持
          - RAG 检索生成
        """
        print("\n" + "=" * 60)
        print("  演示1: 基础问答 + 跨轮次上下文追踪")
        print("=" * 60)

        conversation_id = None
        messages = [
            "我的订单#20240001发货了吗？",        # 第1轮
            "那它大概什么时候能到？",              # 第2轮 — "它"指代订单#20240001
            "好的，那如果到了我不满意能退货吗？",   # 第3轮 — 继承订单上下文
        ]

        for i, msg in enumerate(messages):
            print(f"\n{'─' * 40}")
            print(f"👤 用户(第{i+1}轮): {msg}")
            response = self.orchestrator.process(msg, conversation_id=conversation_id)
            conversation_id = response.conversation_id

            print(f"🤖 客服: {response.content[:150]}...")
            print(f"   [置信度: {response.confidence:.2f}, 意图: {response.intent}]")
            if response.tracked_entities:
                print(f"   [跟踪实体: {response.tracked_entities}]")

    def run_demo_2_multi_intent(self):
        """
        演示2：多意图处理

        展示：
          - 一条消息包含多个意图的识别
          - 多意图信息的逐个处理
        """
        print("\n" + "=" * 60)
        print("  演示2: 多意图识别与处理")
        print("=" * 60)

        multi_intent_messages = [
            "我要退货，顺便问一下你们会员怎么升级？",
            "密码忘了，而且物流查不到，退款什么时候到账？",
        ]

        for msg in multi_intent_messages:
            print(f"\n{'─' * 40}")
            print(f"👤 用户: {msg}")

            # 直接使用意图识别器查看多意图
            result = self.orchestrator.intent_recognizer.recognize(msg)
            print(f"🔍 识别到 {len(result.intents)} 个意图:")
            for intent in result.intents:
                print(f"   • {intent.intent.value} (置信度: {intent.confidence:.2f})")
            print(f"   多意图: {result.is_multi_intent}")
            if result.processing_priority:
                print(f"   处理优先级: {result.processing_priority}")

    def run_demo_3_structured_flow(self):
        """
        演示3：退换货结构化流程

        展示：
          - FSM 状态转移
          - 槽位逐步填充
          - 流程取消处理
        """
        print("\n" + "=" * 60)
        print("  演示3: 退换货结构化流程 (FSM + Slot Filling)")
        print("=" * 60)

        conversation_id = None

        flow_messages = [
            "我要退货",
            "",           # 系统提示提供订单号
            "#20240001",  # 用户提供订单号
            "质量问题",    # 选择退货原因
            "有照片，衣服明显色差",
            "确认",       # 确认提交
        ]

        # 启动流程
        msg = flow_messages[0]
        print(f"\n👤 用户: {msg}")
        response = self.orchestrator.process(msg)
        conversation_id = response.conversation_id
        print(f"🤖 客服: {response.content}")
        if response.flow_step:
            print(f"   [流程步骤: {response.flow_step.get('status')}]")

        # 后续槽位填充
        for i, msg in enumerate(flow_messages[2:5], 1):
            print(f"\n👤 用户: {msg}")
            response = self.orchestrator.process(msg, conversation_id=conversation_id)
            print(f"🤖 客服: {response.content}")
            if response.flow_step:
                print(f"   [进度: {response.flow_step.get('status')}]")

        # 确认提交
        print(f"\n👤 用户: 确认")
        response = self.orchestrator.process("确认", conversation_id=conversation_id)
        print(f"🤖 客服: {response.content}")
        if response.flow_step:
            print(f"   [流程结果: {response.flow_step.get('status')}]")

    def run_demo_4_transfer_flow(self):
        """
        演示4：转人工流程

        展示：
          - 转接触发
          - 上下文打包透传
          - 排队/分配状态
        """
        print("\n" + "=" * 60)
        print("  演示4: 人工转接流程")
        print("=" * 60)

        conversation_id = None

        # 先建立一些对话上下文
        setup_messages = [
            "我的订单#20240001收到的东西是坏的",
            "我拍了照片，确实有明显损坏",
        ]
        for msg in setup_messages:
            response = self.orchestrator.process(msg, conversation_id=conversation_id)
            conversation_id = response.conversation_id

        # 触发转人工
        print(f"\n👤 用户: 帮我转人工吧，我不想和机器人说了")
        response = self.orchestrator.process(
            "帮我转人工吧，我不想和机器人说了",
            conversation_id=conversation_id,
        )
        print(f"🤖 客服: {response.content}")
        if response.transfer_info:
            print(f"\n📋 转接详情:")
            for k, v in response.transfer_info.items():
                print(f"   {k}: {v}")


# ============================================================================
# 全局单例
# ============================================================================
_orchestrator: Optional[DialogueOrchestrator] = None
_orch_lock = threading.Lock()


def get_orchestrator() -> DialogueOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orch_lock:
            if _orchestrator is None:
                _orchestrator = DialogueOrchestrator()
    return _orchestrator


# ============================================================================
# 主入口 — 运行所有演示
# ============================================================================
if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    demo = MultiTurnDemo()

    print("\n" + "█" * 60)
    print("  DeepService 对话管理 — 多轮对话完整演示")
    print("  整合 Session / Context / Intent / FSM / Router / Transfer")
    print("█" * 60)

    # 运行演示（根据需要注释/取消注释）
    try:
        demo.run_demo_1_basic_qa_with_context()
    except Exception as e:
        logger.error(f"演示1失败: {e}")

    try:
        demo.run_demo_2_multi_intent()
    except Exception as e:
        logger.error(f"演示2失败: {e}")

    try:
        demo.run_demo_3_structured_flow()
    except Exception as e:
        logger.error(f"演示3失败: {e}")

    try:
        demo.run_demo_4_transfer_flow()
    except Exception as e:
        logger.error(f"演示4失败: {e}")

    print("\n" + "█" * 60)
    print("  演示完成 ✓")
    print("█" * 60)
