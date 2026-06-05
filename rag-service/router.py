"""
=============================================================================
DeepService 对话管理 — 路由决策模块 (Router)
=============================================================================
职责：
  1. 意图 → 处理策略的路由映射
  2. 多意图时的任务编排（并行/串行处理）
  3. 置信度不足时的兜底策略
  4. 处理结果聚合

企业级设计原则：
  - 每个意图有明确的路由目标（RAG检索 / 规则应答 / 转人工 / 结构化流程）
  - 意图路由器是对话系统的"交通指挥"
  - 低置信度意图自动降级或追问
  - 支持处理策略链（如：先检索 → 置信度不够 → 追问 → 仍不够 → 转人工）
=============================================================================
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


# ============================================================================
# 路由策略定义
# ============================================================================
class RouteAction(str, Enum):
    """路由动作"""
    RAG_RETRIEVAL = "rag_retrieval"           # RAG 检索生成
    RAG_ORDER_LOOKUP = "rag_order_lookup"     # RAG + 订单数据查询
    RULE_RESPONSE = "rule_response"           # 规则直接回复（固定话术）
    STRUCTURED_FLOW = "structured_flow"       # 启动结构化流程（退换货等）
    TRANSFER_HUMAN = "transfer_human"         # 转人工
    CLARIFY = "clarify"                       # 追问澄清
    LLM_DIRECT = "llm_direct"                 # 直接 LLM 回答（无 RAG）
    MULTI_STEP = "multi_step"                 # 多步骤处理


@dataclass
class RouteDecision:
    """路由决策"""
    intent: IntentCategory
    action: RouteAction
    priority: int = 0                         # 处理优先级（数字越小越优先）
    confidence_threshold: float = 0.50        # 触发此路由的最低置信度
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "action": self.action.value,
            "priority": self.priority,
            "confidence_threshold": self.confidence_threshold,
            "metadata": self.metadata,
        }


# ============================================================================
# 意图路由表
# ============================================================================
class IntentRouter:
    """
    意图路由器

    核心路由表：意图 → 处理策略的映射
    """

    # 路由表定义
    ROUTE_TABLE: Dict[IntentCategory, RouteDecision] = {
        # ──── RAG 检索类 ────
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

        # ──── RAG + 数据查询类 ────
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

        # ──── 结构化流程类 ────
        IntentCategory.RETURN_EXCHANGE: RouteDecision(
            intent=IntentCategory.RETURN_EXCHANGE,
            action=RouteAction.STRUCTURED_FLOW,
            priority=1,
            confidence_threshold=0.60,
            metadata={"flow_type": FlowType.RETURN_EXCHANGE},
        ),

        # ──── 转人工类 ────
        IntentCategory.TRANSFER_HUMAN: RouteDecision(
            intent=IntentCategory.TRANSFER_HUMAN,
            action=RouteAction.TRANSFER_HUMAN,
            priority=0,                       # 最高优先级
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

        # ──── 规则应答类 ────
        IntentCategory.GREETING: RouteDecision(
            intent=IntentCategory.GREETING,
            action=RouteAction.RULE_RESPONSE,
            priority=0,
            confidence_threshold=0.80,
        ),

        # ──── 追问澄清类 ────
        IntentCategory.CLARIFICATION: RouteDecision(
            intent=IntentCategory.CLARIFICATION,
            action=RouteAction.CLARIFY,
            priority=1,
            confidence_threshold=0.30,
        ),

        # ──── 兜底类 ────
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
            metadata={"suggest_transfer": True},  # 账号问题最终需要人工
        ),
    }

    # 规则应答模板
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
        logger.info("[IntentRouter] 路由器初始化完成")

    def route(
        self,
        recognition_result: RecognitionResult,
        session_id: str = "",
    ) -> List[RouteDecision]:
        """
        对识别结果进行路由决策

        返回按优先级排列的路由决策列表。
        多意图时返回多个决策。
        """
        decisions = []

        for intent_result in recognition_result.intents:
            decision = self._decide_for_intent(intent_result, session_id)

            if decision:
                decisions.append(decision)

        # 按优先级排序
        decisions.sort(key=lambda d: d.priority)

        logger.info(
            f"[IntentRouter] 路由决策: "
            f"{[(d.intent.value, d.action.value) for d in decisions]}"
        )

        return decisions

    def route_single(self, recognition_result: RecognitionResult) -> RouteAction:
        """
        单意图路由 — 返回最高优先级的动作

        用于大多数单意图场景。
        """
        decisions = self.route(recognition_result)
        if decisions:
            return decisions[0].action
        return RouteAction.CLARIFY  # 兜底

    def _decide_for_intent(
        self,
        intent_result: IntentResult,
        session_id: str,
    ) -> Optional[RouteDecision]:
        """为单个意图做路由决策"""
        intent = intent_result.intent
        confidence = intent_result.confidence

        # 查找路由表
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

        # 置信度不足 → 降级策略
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
        置信度不足时的降级策略

        降级链:
          RAG_RETRIEVAL → CLARIFY（追问一次）
          CLARIFY（再次失败） → TRANSFER_HUMAN
        """
        # 检查该会话的追问次数
        clarify_count = self._get_clarify_count(session_id)

        if clarify_count < 2:
            # 第一次/第二次：追问澄清
            logger.info(
                f"[IntentRouter] 置信度不足 ({confidence:.2f} < {decision.confidence_threshold})"
                f"，降级为追问澄清（第{clarify_count + 1}次）"
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
            # 两次追问仍不足 → 转人工
            logger.info(
                f"[IntentRouter] 已追问{clarify_count}次仍不明确，降级为转人工"
            )
            return RouteDecision(
                intent=decision.intent,
                action=RouteAction.TRANSFER_HUMAN,
                priority=0,  # 最高优先级
                confidence_threshold=0.0,
                metadata={
                    "original_action": decision.action.value,
                    "trigger": "low_confidence_after_clarify",
                    "clarify_attempts": clarify_count,
                },
            )

    def _get_clarify_count(self, session_id: str) -> int:
        """获取会话的追问次数"""
        # 简化实现：从状态跟踪器获取
        # 生产环境从 Redis 获取
        try:
            status = self.state_tracker.get_status(session_id)
            return status.get("clarify_count", 0)
        except Exception:
            return 0

    def get_routing_summary(
        self,
        recognition_result: RecognitionResult,
    ) -> Dict:
        """
        获取路由摘要 — 用于日志和调试
        """
        decisions = self.route(recognition_result)
        return {
            "is_multi_intent": recognition_result.is_multi_intent,
            "route_count": len(decisions),
            "routes": [d.to_dict() for d in decisions],
            "primary_action": decisions[0].action.value if decisions else "clarify",
        }


# ============================================================================
# 路由执行器 — 实际调用对应的处理模块
# ============================================================================
class RouteExecutor:
    """
    路由执行器

    根据路由决策，实际调用对应的处理器。
    """

    def __init__(self):
        self.router = IntentRouter()
        self.state_tracker = get_dialogue_state_tracker()

        # 处理函数注册表
        self._handlers: Dict[RouteAction, Callable] = {}

        logger.info("[RouteExecutor] 初始化完成")

    def register_handler(self, action: RouteAction, handler: Callable):
        """注册自定义处理器"""
        self._handlers[action] = handler
        logger.info(f"[RouteExecutor] 注册处理器: {action.value}")

    def execute(
        self,
        recognition_result: RecognitionResult,
        session_id: str,
        user_message: str,
        **context,
    ) -> Dict:
        """
        执行路由决策

        返回处理结果。
        context 可传入：retrieval_service, generator, defense_system 等依赖
        """
        decisions = self.router.route(recognition_result, session_id)

        if not decisions:
            return {
                "status": "no_route",
                "message": "无法确定如何处理您的请求，请尝试换个方式描述。",
            }

        # 按优先级处理
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
                    logger.error(f"[RouteExecutor] 处理器 {decision.action.value} 执行失败: {e}")
                    results.append({
                        "status": "error",
                        "action": decision.action.value,
                        "error": str(e),
                    })
            else:
                logger.warning(f"[RouteExecutor] 未注册的处理器: {decision.action.value}")
                results.append({
                    "status": "unhandled",
                    "action": decision.action.value,
                    "message": f"处理器 {decision.action.value} 未注册",
                })

        # 聚合结果
        return self._aggregate_results(results, decisions[0])

    def _aggregate_results(
        self,
        results: List[Dict],
        primary_decision: RouteDecision,
    ) -> Dict:
        """聚合多个处理器的结果"""
        if not results:
            return {"status": "error", "message": "无处理结果"}

        if len(results) == 1:
            return results[0]

        # 多意图处理结果拼接
        messages = [r.get("message", "") for r in results if r.get("message")]
        return {
            "status": "multi_intent_processed",
            "primary_action": primary_decision.action.value,
            "message": "\n\n---\n\n".join(messages),
            "sub_results": results,
        }


# ============================================================================
# 内置处理器实现
# ============================================================================
def build_default_handlers() -> Dict[RouteAction, Callable]:
    """
    构建默认处理器字典

    返回可直接注册到 RouteExecutor 的处理器映射。
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


# ============================================================================
# 全局单例
# ============================================================================
import threading

_router: Optional[IntentRouter] = None
_executor: Optional[RouteExecutor] = None
_lock = threading.Lock()


def get_router() -> IntentRouter:
    global _router
    if _router is None:
        with _lock:
            if _router is None:
                _router = IntentRouter()
    return _router


def get_route_executor() -> RouteExecutor:
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                _executor = RouteExecutor()
                # 注册默认处理器
                for action, handler in build_default_handlers().items():
                    _executor.register_handler(action, handler)
    return _executor


# ============================================================================
# 独立测试
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepService Router — 独立测试")
    logger.info("=" * 60)

    recognizer = get_intent_recognizer()
    router = get_router()

    test_queries = [
        "我的订单#12345发货了吗？",
        "我要退货",
        "你好",
        "你们太差了我要投诉",
        "asdfghjkl",  # 无意义
    ]

    for query in test_queries:
        result = recognizer.recognize(query)
        decisions = router.route(result)
        logger.info(f"\n输入: '{query}'")
        logger.info(f"  意图: {[i.intent.value for i in result.intents]}")
        logger.info(f"  路由: {[(d.action.value, d.priority) for d in decisions]}")

    # 测试路由摘要
    result = recognizer.recognize("我要退货，还要查订单")
    summary = router.get_routing_summary(result)
    logger.info(f"\n路由摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    logger.info("=" * 60)
    logger.info("路由测试完成 ✓")
