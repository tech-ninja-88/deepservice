"""
Dialogue Orchestrator — integrates all dialogue modules into a single processing pipeline.
Pipeline: Session -> TransferDetect -> Intent -> Context -> StateTracker -> Router -> Defense -> Response.
"""

import json
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Generator, Tuple
from pathlib import Path

from loguru import logger

from config import get_config

# Import all dialogue management modules
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


# ### Standard response format
@dataclass
class DialogueResponse:
    """
    Standardized response format — all processing paths return this.
    """
    # Core fields
    conversation_id: str
    content: str
    response_type: str                      # knowledge_based / uncertain / greeting / transfer / flow_step
    confidence: float = 1.0

    # Intent info
    intent: str = ""
    is_multi_intent: bool = False
    intents_detail: List[Dict] = field(default_factory=list)

    # Route info
    route_action: str = ""

    # Structured flow info
    flow_step: Optional[Dict] = None

    # Transfer info
    transfer_info: Optional[Dict] = None

    # Sources
    sources: List[Dict] = field(default_factory=list)

    # Metadata
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


# ### Core orchestrator
class DialogueOrchestrator:
    """
    Full dialogue processing engine.
    Usage: orchestrator.process("你好，我想退货", conversation_id="...")
    """

    def __init__(self):
        self.config = get_config()

        # Initialize all subsystems
        self.session_mgr = get_session_manager()
        self.context_mgr = get_context_manager()
        self.intent_recognizer = get_intent_recognizer()
        self.state_tracker = get_dialogue_state_tracker()
        self.router = get_router()
        self.transfer_service = get_human_transfer_service()

        # Optional RAG dependencies (lazy import to avoid circular deps)
        self._retrieval_service = None
        self._rag_generator = None
        self._defense_system = None

        logger.info("[DialogueOrchestrator] Orchestrator initialized")

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
        Process a single user message through the full pipeline. Core entry point.
        """
        start_time = time.time()
        logger.info(f"[Orchestrator] ====== Processing message ======")
        logger.info(f"[Orchestrator] Input: '{user_message[:80]}...'")

        # Step 1: Session management
        session, is_new = self.session_mgr.get_or_create_session(
            session_id=conversation_id,
            user_id=user_id,
            channel=channel,
            locale=locale,
        )
        conversation_id = session.id
        logger.info(
            f"[Step 1] Session: {conversation_id[:12]}... "
            f"(new={is_new}, turn={session.turn_count})"
        )

        # Save user message
        user_msg = Message(role=MessageRole.USER, content=user_message)
        self.session_mgr.append_message(conversation_id, user_msg)

        # Step 2: Human transfer detection
        intent = ""
        confidence = 1.0

        # Quick intent recognition for transfer check
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
            logger.info(f"[Step 2] Transfer triggered: {transfer_trigger.value}")
            return self._handle_transfer(
                conversation_id, user_id, transfer_trigger,
                quick_result, start_time,
            )

        # Step 3: Intent recognition (full)
        logger.info("[Step 3] Intent recognition...")
        recognition = quick_result  # reuse quick result
        primary_intent = recognition.primary_intent
        top_intent_name = primary_intent.intent.value if primary_intent else "unknown"

        logger.info(
            f"[Step 3] Intent: {top_intent_name} "
            f"(confidence: {primary_intent.confidence if primary_intent else 0:.2f})"
        )
        if recognition.is_multi_intent:
            logger.info(
                f"[Step 3] Multi-intent: {[i.intent.value for i in recognition.intents]}"
            )

        # Step 4: Build conversation memory
        logger.info("[Step 4] Building memory...")
        memory = self.context_mgr.build_memory(conversation_id, self.session_mgr)

        # Check for cross-turn entities relevant to current query
        relevant_entities = self.context_mgr.get_relevant_entities_for_query(
            memory, user_message
        )

        if relevant_entities:
            logger.info(
                f"[Step 4] Cross-turn entities: "
                f"{[(e.type.value, e.value) for e in relevant_entities]}"
            )

        # Step 5: Check if in structured flow
        state = self.state_tracker.fsm.get_current_state(conversation_id)
        logger.info(f"[Step 5] Current state: {state.value}")

        if self.state_tracker.fsm.is_in_structured_flow(conversation_id):
            # In flow — continue filling slots
            return self._handle_flow_step(
                conversation_id, user_message, recognition, memory, start_time
            )

        # Step 6: Route decision
        logger.info("[Step 6] Route decision...")
        route_decisions = self.router.route(recognition, conversation_id)

        if not route_decisions:
            return self._build_fallback_response(
                conversation_id, "Unable to determine how to handle your request.", recognition, start_time
            )

        primary_decision = route_decisions[0]
        logger.info(f"[Step 6] Route: {primary_decision.action.value}")

        # Step 7: Execute route
        logger.info("[Step 7] Executing handler...")
        response = self._execute_route(
            primary_decision,
            user_message=user_message,
            conversation_id=conversation_id,
            recognition=recognition,
            memory=memory,
            start_time=start_time,
        )

        # Step 8: Save assistant message
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

        # Step 9: Update user profile
        self.context_mgr.update_profile_from_turn(
            user_id=user_id,
            intent=top_intent_name,
            sentiment=0.5,  # sentiment analysis not yet wired
            session_id=conversation_id,
        )

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[Orchestrator] ====== Processing complete ({elapsed:.0f}ms) ======")

        return response

    def process_stream(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: str = "anonymous",
    ) -> Generator[str, None, None]:
        """Streaming response for SSE transport.

        Processes the full message synchronously then yields in chunks
        for progressive UI updates. True token-level streaming requires
        deeper integration with RAGGenerator.generate_stream().
        """
        full_response = self.process(user_message, conversation_id, user_id)

        content = full_response.content
        chunk_size = 10

        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield json.dumps({"type": "token", "content": chunk}, ensure_ascii=False)

        yield json.dumps({"type": "metadata", "data": full_response.to_dict()}, ensure_ascii=False)
        yield json.dumps({"type": "done"}, ensure_ascii=False)

    # ### Route handler branches

    def _execute_route(
        self,
        decision: RouteDecision,
        **context,
    ) -> DialogueResponse:
        """Execute the handler matching the route decision"""
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
        RAG retrieval + generation with hallucination guard boundary check and output validation.
        """
        # 1. Retrieve
        search_result = self.retrieval_service.search(
            query=user_message,
            strategy="rrf",
            enable_rerank=True,
        )

        # 2. Knowledge boundary pre-check
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

        # 3. Generate (inject context memory)
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

        # 4. Output validation
        validation = self.defense_system.post_generation_validate(
            rag_response.content, search_result
        )
        final_confidence = self.defense_system.score_confidence(
            search_result, rag_response.content, validation
        )

        # 5. If hallucination risk is high, switch to fallback
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
        """RAG + order lookup (reserved for DB integration)"""
        # Order database integration point
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
        Start a structured flow.
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
        Handle user input during a structured flow.
        """
        result = self.state_tracker.process_user_input(conversation_id, user_message)

        response_type = "flow_step"
        if result["status"] == "flow_completed":
            response_type = "flow_completed"
        elif result["status"] == "flow_cancelled":
            response_type = "knowledge_based"  # back to free chat
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
        """Ask for clarification"""
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
        """Direct rule-based reply"""
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
        """Direct LLM response (no RAG retrieval)"""
        try:
            from config import get_llm_client
            client = get_llm_client()
            if client is None:
                return self._build_fallback_response(
                    conversation_id,
                    "抱歉，LLM 服务未配置，请联系管理员设置 DEEPSEEK_API_KEY。",
                    None, start_time,
                    error="llm_unavailable",
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
        Execute human transfer.
        """
        # Build context
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

        # Initiate transfer
        transfer = self.transfer_service.initiate_transfer(
            session_id=conversation_id,
            trigger=trigger,
            user_id=user_id,
            conversation_summary=summary,
            recent_messages=recent_messages,
            tracked_entities=entities,
        )

        # Build user response
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
        """Build a fallback response"""
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


_orchestrator: Optional[DialogueOrchestrator] = None


def get_orchestrator() -> DialogueOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DialogueOrchestrator()
    return _orchestrator


# ### Quick smoke test
if __name__ == "__main__":
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    orch = DialogueOrchestrator()
    resp = orch.process("你好，请问可以退货吗？")
    print(f"Intent: {resp.intent} | Confidence: {resp.confidence:.2f}")
    print(f"Response: {resp.content[:200]}")
    print("Dialogue orchestrator self-check complete.")
