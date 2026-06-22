"""
Router: maps intents to processing strategies, handles multi-intent orchestration,
and applies fallback chains when confidence is low.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

from loguru import logger

from intent_recognizer import (
    IntentCategory, IntentResult, RecognitionResult,
    HybridIntentRecognizer, get_intent_recognizer,
)
from dialogue_state import (
    DialogueStateTracker, FlowType, DialogueState,
    get_dialogue_state_tracker,
)


# [[[ Route action & decision ]]]
class RouteAction(str, Enum):
    """Routing actions"""
    RAG_RETRIEVAL = "rag_retrieval"           # RAG retrieval + generation
    RAG_ORDER_LOOKUP = "rag_order_lookup"     # RAG + order data lookup
    RULE_RESPONSE = "rule_response"           # fixed script response
    STRUCTURED_FLOW = "structured_flow"       # start structured flow (returns etc.)
    TRANSFER_HUMAN = "transfer_human"         # transfer to human agent
    CLARIFY = "clarify"                       # ask for clarification
    LLM_DIRECT = "llm_direct"                 # direct LLM answer (no RAG)
    MULTI_STEP = "multi_step"                 # multi-step processing


@dataclass
class RouteDecision:
    """A single routing decision"""
    intent: IntentCategory
    action: RouteAction
    priority: int = 0                         # lower = higher priority
    confidence_threshold: float = 0.50
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "action": self.action.value,
            "priority": self.priority,
            "confidence_threshold": self.confidence_threshold,
            "metadata": self.metadata,
        }


# [[[ Intent routing table ]]]
class IntentRouter:
    """
    Maps intents to processing strategies via a routing table.
    """

    ROUTE_TABLE: Dict[IntentCategory, RouteDecision] = {
        # --- RAG retrieval ---
        IntentCategory.PRODUCT_INQUIRY: RouteDecision(
            intent=IntentCategory.PRODUCT_INQUIRY,
            action=RouteAction.RAG_RETRIEVAL,
            priority=1,
            confidence_threshold=0.50,
            metadata={"source": "knowledge_base"},
        ),
        IntentCategory.SHIPPING_INQUIRY: RouteDecision(
            intent=IntentCategory.SHIPPING_INQUIRY,
            action=RouteAction.RAG_RETRIEVAL,
            priority=1,
            confidence_threshold=0.50,
        ),
        IntentCategory.MEMBERSHIP_INQUIRY: RouteDecision(
            intent=IntentCategory.MEMBERSHIP_INQUIRY,
            action=RouteAction.RAG_RETRIEVAL,
            priority=1,
            confidence_threshold=0.50,
        ),
        IntentCategory.USAGE_GUIDE: RouteDecision(
            intent=IntentCategory.USAGE_GUIDE,
            action=RouteAction.RAG_RETRIEVAL,
            priority=1,
            confidence_threshold=0.50,
        ),
        IntentCategory.TECHNICAL_ISSUE: RouteDecision(
            intent=IntentCategory.TECHNICAL_ISSUE,
            action=RouteAction.RAG_RETRIEVAL,
            priority=1,
            confidence_threshold=0.50,
            metadata={"suggest_transfer_on_low_confidence": True},
        ),

        # --- RAG + order lookup ---
        IntentCategory.ORDER_STATUS: RouteDecision(
            intent=IntentCategory.ORDER_STATUS,
            action=RouteAction.RAG_ORDER_LOOKUP,
            priority=1,
            confidence_threshold=0.50,
            metadata={"requires_entity": "order_id"},
        ),
        IntentCategory.REFUND_CONSULT: RouteDecision(
            intent=IntentCategory.REFUND_CONSULT,
            action=RouteAction.RAG_ORDER_LOOKUP,
            priority=1,
            confidence_threshold=0.50,
            metadata={"requires_entity": "order_id"},
        ),
        IntentCategory.PAYMENT_ISSUE: RouteDecision(
            intent=IntentCategory.PAYMENT_ISSUE,
            action=RouteAction.RAG_ORDER_LOOKUP,
            priority=1,
            confidence_threshold=0.50,
        ),

        # --- Structured flows ---
        IntentCategory.RETURN_EXCHANGE: RouteDecision(
            intent=IntentCategory.RETURN_EXCHANGE,
            action=RouteAction.STRUCTURED_FLOW,
            priority=1,
            confidence_threshold=0.60,
            metadata={"flow_type": FlowType.RETURN_EXCHANGE},
        ),

        # --- Transfer to human ---
        IntentCategory.TRANSFER_HUMAN: RouteDecision(
            intent=IntentCategory.TRANSFER_HUMAN,
            action=RouteAction.TRANSFER_HUMAN,
            priority=0,                       # highest priority
            confidence_threshold=0.70,
            metadata={"trigger": "user_requested"},
        ),
        IntentCategory.COMPLAINT: RouteDecision(
            intent=IntentCategory.COMPLAINT,
            action=RouteAction.TRANSFER_HUMAN,
            priority=0,
            confidence_threshold=0.60,
            metadata={"trigger": "complaint_detected"},
        ),

        # --- Rule-based responses ---
        IntentCategory.GREETING: RouteDecision(
            intent=IntentCategory.GREETING,
            action=RouteAction.RULE_RESPONSE,
            priority=0,
            confidence_threshold=0.80,
        ),

        # --- Clarification ---
        IntentCategory.CLARIFICATION: RouteDecision(
            intent=IntentCategory.CLARIFICATION,
            action=RouteAction.CLARIFY,
            priority=1,
            confidence_threshold=0.30,
        ),

        # --- Fallback ---
        IntentCategory.UNKNOWN: RouteDecision(
            intent=IntentCategory.UNKNOWN,
            action=RouteAction.CLARIFY,
            priority=99,
            confidence_threshold=0.10,
        ),
        IntentCategory.CHITCHAT: RouteDecision(
            intent=IntentCategory.CHITCHAT,
            action=RouteAction.LLM_DIRECT,
            priority=99,
            confidence_threshold=0.30,
        ),
        IntentCategory.FEEDBACK: RouteDecision(
            intent=IntentCategory.FEEDBACK,
            action=RouteAction.LLM_DIRECT,
            priority=2,
            confidence_threshold=0.40,
        ),
        IntentCategory.ACCOUNT_ISSUE: RouteDecision(
            intent=IntentCategory.ACCOUNT_ISSUE,
            action=RouteAction.RAG_RETRIEVAL,
            priority=1,
            confidence_threshold=0.50,
            metadata={"suggest_transfer": True},  # account issues typically need human review
        ),
    }

    # Rule-based response templates
    RULE_RESPONSES: Dict[IntentCategory, str] = {
        IntentCategory.GREETING: (
            "您好！我是 DeepService 智能客服助手 👋\n\n"
            "我可以帮您解答以下问题：\n"
            "  📦 订单查询与物流追踪\n"
            "  🔄 退换货政策与流程\n"
            "  💰 退款咨询\n"
            "  👤 账号与会员问题\n"
            "  🛠️ 技术问题\n\n"
            "请直接描述您遇到的问题，我会尽力为您解答！"
        ),
    }

    def __init__(self):
        self.recognizer = get_intent_recognizer()
        self.state_tracker = get_dialogue_state_tracker()
        logger.info("[IntentRouter] Router initialized")

    def route(
        self,
        recognition_result: RecognitionResult,
        session_id: str = "",
    ) -> List[RouteDecision]:
        """
        Produce routing decisions for recognition results, sorted by priority.
        Returns multiple decisions for multi-intent results.
        """
        decisions = []

        for intent_result in recognition_result.intents:
            decision = self._decide_for_intent(intent_result, session_id)

            if decision:
                decisions.append(decision)

        # Sort by priority
        decisions.sort(key=lambda d: d.priority)

        logger.info(
            f"[IntentRouter] Route decisions: "
            f"{[(d.intent.value, d.action.value) for d in decisions]}"
        )

        return decisions

    def route_single(self, recognition_result: RecognitionResult) -> RouteAction:
        """
        Single-intent routing — returns the top-priority action.
        """
        decisions = self.route(recognition_result)
        if decisions:
            return decisions[0].action
        return RouteAction.CLARIFY  # ultimate fallback

    def _decide_for_intent(
        self,
        intent_result: IntentResult,
        session_id: str,
    ) -> Optional[RouteDecision]:
        """Decide routing for a single intent"""
        intent = intent_result.intent
        confidence = intent_result.confidence

        # Look up in route table
        base_decision = self.ROUTE_TABLE.get(intent)
        if base_decision is None:
            base_decision = self.ROUTE_TABLE[IntentCategory.UNKNOWN]

        decision = RouteDecision(
            intent=intent,
            action=base_decision.action,
            priority=base_decision.priority,
            confidence_threshold=base_decision.confidence_threshold,
            metadata=dict(base_decision.metadata),
        )

        # Low confidence -> apply fallback
        if confidence < decision.confidence_threshold:
            decision = self._apply_fallback_strategy(decision, confidence, session_id)

        return decision

    def _apply_fallback_strategy(
        self,
        decision: RouteDecision,
        confidence: float,
        session_id: str,
    ) -> RouteDecision:
        """
        Fallback chain: RAG_RETRIEVAL -> CLARIFY (twice) -> TRANSFER_HUMAN.
        """
        clarify_count = self._get_clarify_count(session_id)

        if clarify_count < 2:
            # First/second attempt: ask for clarification
            logger.info(
                f"[IntentRouter] Low confidence ({confidence:.2f} < {decision.confidence_threshold})"
                f", downgrading to clarify (attempt {clarify_count + 1})"
            )
            return RouteDecision(
                intent=decision.intent,
                action=RouteAction.CLARIFY,
                priority=decision.priority + 1,
                confidence_threshold=0.1,
                metadata={
                    "original_action": decision.action.value,
                    "original_confidence": confidence,
                    "clarify_attempt": clarify_count + 1,
                },
            )
        else:
            # Still unclear after two attempts -> transfer to human
            logger.info(
                f"[IntentRouter] Still unclear after {clarify_count} clarify attempts, transferring to human"
            )
            return RouteDecision(
                intent=decision.intent,
                action=RouteAction.TRANSFER_HUMAN,
                priority=0,  # highest priority
                confidence_threshold=0.0,
                metadata={
                    "original_action": decision.action.value,
                    "trigger": "low_confidence_after_clarify",
                    "clarify_attempts": clarify_count,
                },
            )

    def _get_clarify_count(self, session_id: str) -> int:
        """Get clarification attempt count for a session"""
        # Simplified: pull from state tracker
        # TODO: pull from Redis when integrating session store
        try:
            status = self.state_tracker.get_status(session_id)
            return status.get("clarify_count", 0)
        except Exception:
            return 0

    def get_routing_summary(
        self,
        recognition_result: RecognitionResult,
    ) -> Dict:
        """Return routing summary for logging/debugging"""
        decisions = self.route(recognition_result)
        return {
            "is_multi_intent": recognition_result.is_multi_intent,
            "route_count": len(decisions),
            "routes": [d.to_dict() for d in decisions],
            "primary_action": decisions[0].action.value if decisions else "clarify",
        }


# [[[ Route executor ]]]
class RouteExecutor:
    """
    Executes routing decisions by dispatching to registered handlers.
    """

    def __init__(self):
        self.router = IntentRouter()
        self.state_tracker = get_dialogue_state_tracker()

        # Handler registry
        self._handlers: Dict[RouteAction, Callable] = {}

        logger.info("[RouteExecutor] Initialized")

    def register_handler(self, action: RouteAction, handler: Callable):
        """Register a custom handler for an action"""
        self._handlers[action] = handler
        logger.info(f"[RouteExecutor] Handler registered: {action.value}")

    def execute(
        self,
        recognition_result: RecognitionResult,
        session_id: str,
        user_message: str,
        **context,
    ) -> Dict:
        """
        Execute routing decisions. context may include retrieval_service, generator, etc.
        """
        decisions = self.router.route(recognition_result, session_id)

        if not decisions:
            return {
                "status": "no_route",
                "message": "无法确定如何处理您的请求，请尝试换个方式描述。",
            }

        # Process in priority order
        results = []
        for decision in decisions:
            handler = self._handlers.get(decision.action)
            if handler:
                try:
                    result = handler(
                        recognition_result=recognition_result,
                        decision=decision,
                        session_id=session_id,
                        user_message=user_message,
                        **context,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"[RouteExecutor] Handler {decision.action.value} failed: {e}")
                    results.append({
                        "status": "error",
                        "action": decision.action.value,
                        "error": str(e),
                    })
            else:
                logger.warning(f"[RouteExecutor] Unregistered handler: {decision.action.value}")
                results.append({
                    "status": "unhandled",
                    "action": decision.action.value,
                    "message": f"Handler {decision.action.value} not registered",
                })

        # Aggregate results
        return self._aggregate_results(results, decisions[0])

    def _aggregate_results(
        self,
        results: List[Dict],
        primary_decision: RouteDecision,
    ) -> Dict:
        """Merge results from multiple handlers"""
        if not results:
            return {"status": "error", "message": "无处理结果"}

        if len(results) == 1:
            return results[0]

        # Concatenate multi-intent results
        messages = [r.get("message", "") for r in results if r.get("message")]
        return {
            "status": "multi_intent_processed",
            "primary_action": primary_decision.action.value,
            "message": "\n\n---\n\n".join(messages),
            "sub_results": results,
        }


# [[[ Built-in handlers ]]]
def build_default_handlers() -> Dict[RouteAction, Callable]:
    """
    Build the default handler map for RouteExecutor registration.
    """
    handlers = {}

    def handle_rule_response(**kwargs) -> Dict:
        decision = kwargs.get("decision")
        if decision and decision.intent in IntentRouter.RULE_RESPONSES:
            return {
                "status": "rule_response",
                "message": IntentRouter.RULE_RESPONSES[decision.intent],
            }
        return {"status": "rule_response", "message": "您好！请问有什么可以帮您的？"}

    def handle_clarify(**kwargs) -> Dict:
        user_msg = kwargs.get("user_message", "")
        recognition = kwargs.get("recognition_result")
        top_intent = recognition.get_top_intent(min_confidence=0.3) if recognition else None
        intent_name = top_intent.intent.value if top_intent else "您的问题"

        return {
            "status": "clarify",
            "message": (
                f"抱歉，我没有完全理解您关于'{intent_name}'的问题。\n"
                f"能否请您再详细描述一下？"
            ),
        }

    def handle_transfer_human(**kwargs) -> Dict:
        decision = kwargs.get("decision")
        trigger = decision.metadata.get("trigger", "unknown") if decision else "unknown"
        return {
            "status": "transfer_human",
            "message": "正在为您转接人工客服，请稍候...",
            "trigger": trigger,
        }

    def handle_llm_direct(**kwargs) -> Dict:
        return {"status": "llm_direct", "message": ""}

    handlers[RouteAction.RULE_RESPONSE] = handle_rule_response
    handlers[RouteAction.CLARIFY] = handle_clarify
    handlers[RouteAction.TRANSFER_HUMAN] = handle_transfer_human
    handlers[RouteAction.LLM_DIRECT] = handle_llm_direct

    return handlers


_router: Optional[IntentRouter] = None
_executor: Optional[RouteExecutor] = None


def get_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


def get_route_executor() -> RouteExecutor:
    global _executor
    if _executor is None:
        _executor = RouteExecutor()
        for action, handler in build_default_handlers().items():
            _executor.register_handler(action, handler)
    return _executor


# [[[ Self-check ]]]
if __name__ == "__main__":
    logger.info("Router self-check")
    recognizer = get_intent_recognizer()
    router = get_router()

    test_queries = [
        "我的订单#12345发货了吗？",
        "我要退货",
        "你好",
        "你们太差了我要投诉",
        "asdfghjkl",
    ]

    for query in test_queries:
        result = recognizer.recognize(query)
        decisions = router.route(result)
        logger.info(f"  '{query}' -> intent={[i.intent.value for i in result.intents]}, route={[(d.action.value, d.priority) for d in decisions]}")

    # Route summary
    result = recognizer.recognize("我要退货，还要查订单")
    summary = router.get_routing_summary(result)
    logger.info(f"  Multi-intent summary: {json.dumps(summary, ensure_ascii=False)}")

    logger.info("Router self-check complete.")
