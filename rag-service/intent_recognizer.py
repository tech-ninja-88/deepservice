"""Hybrid intent recognition: rule-based fast-path + LLM fallback with multi-intent decomposition and entity extraction."""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

from loguru import logger

from config import get_config


# *** Intent definitions
class IntentCategory(str, Enum):
    """Predefined intent categories mapped to routing strategies."""

    # Business inquiries
    PRODUCT_INQUIRY = "product_inquiry"         # product info lookup
    ORDER_STATUS = "order_status"               # order status lookup
    RETURN_EXCHANGE = "return_exchange"         # return/exchange inquiry
    REFUND_CONSULT = "refund_consult"           # refund inquiry
    PAYMENT_ISSUE = "payment_issue"             # payment issue
    SHIPPING_INQUIRY = "shipping_inquiry"       # shipping/delivery lookup
    ACCOUNT_ISSUE = "account_issue"             # account issues
    MEMBERSHIP_INQUIRY = "membership_inquiry"   # membership benefits

    # Technical support
    TECHNICAL_ISSUE = "technical_issue"         # technical bugs/errors
    USAGE_GUIDE = "usage_guide"                 # how-to / usage guide

    # Service requests
    COMPLAINT = "complaint"                     # complaint
    FEEDBACK = "feedback"                       # feedback / suggestion
    TRANSFER_HUMAN = "transfer_human"           # explicit human handoff request

    # Dialogue management
    GREETING = "greeting"                       # greeting / salutation
    CLARIFICATION = "clarification"             # clarification follow-up
    CHITCHAT = "chitchat"                       # casual chit-chat
    UNKNOWN = "unknown"                         # unable to classify

    @classmethod
    def get_routing_intents(cls) -> List[str]:
        """Return intents that require routing (non-generic dialogue)."""
        return [
            cls.PRODUCT_INQUIRY, cls.ORDER_STATUS, cls.RETURN_EXCHANGE,
            cls.REFUND_CONSULT, cls.PAYMENT_ISSUE, cls.SHIPPING_INQUIRY,
            cls.ACCOUNT_ISSUE, cls.MEMBERSHIP_INQUIRY, cls.TECHNICAL_ISSUE,
            cls.USAGE_GUIDE, cls.COMPLAINT, cls.FEEDBACK,
        ]


@dataclass
class IntentResult:
    """Single intent result with optional sub-intents for multi-intent scenarios."""
    intent: IntentCategory
    confidence: float
    reason: str = ""
    sub_intents: List["IntentResult"] = field(default_factory=list)  # sub-intents (multi-intent)

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "sub_intents": [s.to_dict() for s in self.sub_intents] if self.sub_intents else [],
        }


@dataclass
class ExtractedEntity:
    """An entity extracted from the user message (order_id, product, date, etc.)."""
    name: str                           # entity name
    value: str                          # entity value
    entity_type: str                    # entity type (order_id, product, date, etc.)
    confidence: float = 1.0
    start_pos: int = 0                  # start position in original text
    end_pos: int = 0                    # end position
    normalized_value: str = ""          # normalized value

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.entity_type,
            "confidence": round(self.confidence, 4),
            "normalized": self.normalized_value or self.value,
        }


@dataclass
class RecognitionResult:
    """Full recognition result: intents + entities + metadata."""
    original_text: str
    intents: List[IntentResult]                     # sorted by confidence descending
    entities: List[ExtractedEntity] = field(default_factory=list)
    is_multi_intent: bool = False                   # multi-intent flag
    primary_intent: Optional[IntentResult] = None    # primary intent (first in multi-intent)
    processing_priority: List[str] = field(default_factory=list)  # processing order
    recognition_time_ms: float = 0.0
    method: str = "rule"                            # "rule" | "llm" | "hybrid"

    def get_top_intent(self, min_confidence: float = 0.3) -> Optional[IntentResult]:
        """Return the highest-confidence intent above the threshold."""
        if self.intents and self.intents[0].confidence >= min_confidence:
            return self.intents[0]
        return None

    def has_intent(self, intent: IntentCategory) -> bool:
        """Check whether a given intent is present."""
        return any(i.intent == intent for i in self.intents)

    def to_dict(self) -> Dict:
        return {
            "original_text": self.original_text,
            "intents": [i.to_dict() for i in self.intents],
            "entities": [e.to_dict() for e in self.entities],
            "is_multi_intent": self.is_multi_intent,
            "primary_intent": self.primary_intent.to_dict() if self.primary_intent else None,
            "processing_priority": self.processing_priority,
            "recognition_time_ms": round(self.recognition_time_ms, 2),
            "method": self.method,
        }


# *** 1. Fast rule-based intent matching (~60% of queries)
class RuleBasedIntentMatcher:
    """
    Rule-based matcher for high-frequency, deterministic scenarios.

    Zero latency (no API call), 100% deterministic, and highly interpretable.
    """

    # Intent rules (sorted by priority)
    INTENT_RULES: List[Tuple[List[str], IntentCategory, float, str]] = [
        # (keywords, intent, confidence, reason)
        (["你好", "嗨", "hello", "hi", "在吗", "您好", "早上好", "下午好", "晚上好"],
         IntentCategory.GREETING, 0.98, "greeting trigger"),

        (["人工", "转人工", "客服人员", "人工客服", "真人", "转接"],
         IntentCategory.TRANSFER_HUMAN, 0.95, "explicit human handoff request"),

        (["投诉", "举报", "垃圾", "太差", "很差", "坑人", "骗子", "差评", "态度"],
         IntentCategory.COMPLAINT, 0.92, "complaint keywords"),

        (["退货", "退换", "换货", "退回去", "退掉"],
         IntentCategory.RETURN_EXCHANGE, 0.90, "return/exchange keywords"),

        (["退款", "退钱", "返款", "退费", "打款"],
         IntentCategory.REFUND_CONSULT, 0.90, "refund keywords"),

        (["订单", "发货", "送达", "物流", "快递", "运单", "tracking", "到哪"],
         IntentCategory.ORDER_STATUS, 0.88, "order & shipping status"),

        (["支付", "付款", "扣款", "刷卡", "银行卡", "支付宝", "微信支付"],
         IntentCategory.PAYMENT_ISSUE, 0.88, "payment related"),

        (["密码", "登录", "注册", "绑定", "手机号", "验证码", "账号", "注销"],
         IntentCategory.ACCOUNT_ISSUE, 0.88, "account related"),

        (["会员", "vip", "积分", "折扣", "等级", "权益", "升级"],
         IntentCategory.MEMBERSHIP_INQUIRY, 0.87, "membership related"),

        (["怎么", "如何", "怎样", "为什么", "是什么", "什么原因"],
         IntentCategory.USAGE_GUIDE, 0.80, "how-to inquiry (weak signal)"),

        (["报错", "错误", "bug", "闪退", "卡顿", "打不开", "连不上", "安装"],
         IntentCategory.TECHNICAL_ISSUE, 0.87, "technical error keywords"),
    ]

    # Multi-intent connector words
    MULTI_INTENT_CONNECTORS = [
        "另外", "还有", "同时", "顺便", "此外", "以及",
        "然后", "接着", "再", "也", "并且", "和",
        "第一个", "第二个", "一是", "二是", "一方面", "另一方面",
    ]

    def match(self, text: str) -> Optional[RecognitionResult]:
        """
        Fast rule-based matching.

        Returns None if rules cannot match, meaning fallback to LLM is needed.
        """
        text_lower = text.lower().strip()
        matched_intents = []

        for keywords, intent, confidence, reason in self.INTENT_RULES:
            for kw in keywords:
                if kw in text_lower:
                    matched_intents.append(IntentResult(
                        intent=intent,
                        confidence=confidence,
                        reason=reason,
                    ))
                    break  # one match per rule group is enough

        if not matched_intents:
            return None  # rules could not match

        # Deduplicate (same intent may hit multiple rules)
        seen = set()
        unique_intents = []
        for ir in matched_intents:
            if ir.intent.value not in seen:
                seen.add(ir.intent.value)
                unique_intents.append(ir)

        # Multi-intent detection
        is_multi = len(unique_intents) > 1
        if not is_multi:
            # Check for multi-intent connectors (may indicate multi-intent missed by rules)
            if any(conn in text_lower for conn in self.MULTI_INTENT_CONNECTORS):
                is_multi = True

        # Build result
        unique_intents.sort(key=lambda x: x.confidence, reverse=True)

        # Entity extraction (shared module)
        from entity_extractor import extract_entities
        entities = extract_entities(text)

        return RecognitionResult(
            original_text=text,
            intents=unique_intents,
            entities=entities,
            is_multi_intent=is_multi,
            primary_intent=unique_intents[0] if unique_intents else None,
            processing_priority=[i.intent.value for i in unique_intents],
            method="rule",
        )



# *** 2. LLM intent recognition (~35% complex scenarios)
class LLMIntentRecognizer:
    """
    DeepSeek-based LLM intent recognition for ambiguous/complex inputs.

    Used when rules fail; handles multi-intent decomposition and semantic understanding.
    """

    # Few-shot examples (critical: anchor the output format)
    FEW_SHOT_EXAMPLES = """
示例1:
用户输入: "我的订单#20240001发货了吗？"
输出:
```json
{
  "intents": [
    {"intent": "order_status", "confidence": 0.95, "reason": "用户询问特定订单的发货状态"}
  ],
  "is_multi_intent": false,
  "entities": [
    {"name": "订单号", "value": "#20240001", "type": "order_id"}
  ]
}
```

示例2:
用户输入: "我要退货，顺便问一下你们会员怎么升级？"
输出:
```json
{
  "intents": [
    {"intent": "return_exchange", "confidence": 0.92, "reason": "用户明确表达退货意图"},
    {"intent": "membership_inquiry", "confidence": 0.88, "reason": "用户询问会员升级规则"}
  ],
  "is_multi_intent": true,
  "processing_priority": ["return_exchange", "membership_inquiry"],
  "entities": []
}
```

示例3:
用户输入: "我的密码忘了，而且物流信息也查不到，还有退款什么时候到账？"
输出:
```json
{
  "intents": [
    {"intent": "account_issue", "confidence": 0.90, "reason": "忘记密码属于账号问题"},
    {"intent": "order_status", "confidence": 0.85, "reason": "查询物流信息"},
    {"intent": "refund_consult", "confidence": 0.88, "reason": "询问退款到账时间"}
  ],
  "is_multi_intent": true,
  "processing_priority": ["account_issue", "order_status", "refund_consult"],
  "entities": []
}
```"""

    def __init__(self):
        self.config = get_config()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from config import get_llm_client
            self._client = get_llm_client()
        return self._client

    def recognize(self, text: str) -> RecognitionResult:
        """Run LLM-based intent recognition and parse the JSON response."""
        start_time = time.time()

        if self.client is None:
            return RecognitionResult(
                original_text=text,
                intents=[IntentResult(intent=IntentCategory.UNKNOWN, confidence=0.1, reason="llm_unavailable")],
                method="llm",
                recognition_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,  # low temperature for consistency
                max_tokens=500,
            )

            result_text = response.choices[0].message.content or "{}"
            parsed = self._parse_response(result_text, text)

        except Exception as e:
            logger.error(f"[LLMIntentRecognizer] recognition failed: {e}")
            parsed = RecognitionResult(
                original_text=text,
                intents=[IntentResult(intent=IntentCategory.UNKNOWN, confidence=0.1, reason=f"LLM error: {e}")],
                method="llm",
            )

        parsed.recognition_time_ms = (time.time() - start_time) * 1000
        return parsed

    def _build_system_prompt(self) -> str:
        """Build the system prompt for intent classification."""
        intent_list = "\n".join(
            f"  - {i.value}: {self._get_intent_description(i)}"
            for i in IntentCategory
        )

        return f"""你是一个专业的用户意图识别专家，负责分析电商客服场景下的用户消息。

你的任务是：
1. 识别用户消息中的**所有意图**（一个消息可能包含多个意图）
2. 对每个意图给出置信度（0-1之间）
3. 判断是否为多意图消息
4. 抽取消息中的关键实体

意图类别定义：
{intent_list}

输出要求：
- 必须以 JSON 格式输出
- 多意图时，按用户表达的顺序排列
- 为多意图指定 processing_priority（处理的先后顺序）
- 置信度基于语义明确度，模糊表达给低分

{self.FEW_SHOT_EXAMPLES}

现在请分析用户消息。记住：只输出 JSON，不要添加额外解释。"""

    def _get_intent_description(self, intent: IntentCategory) -> str:
        """Return the Chinese description for a given intent (used in LLM prompt)."""
        descriptions = {
            IntentCategory.PRODUCT_INQUIRY: "询问产品信息、规格、价格、库存",
            IntentCategory.ORDER_STATUS: "查询订单状态、物流追踪、发货进度",
            IntentCategory.RETURN_EXCHANGE: "退换货申请、退换货流程、退换条件",
            IntentCategory.REFUND_CONSULT: "退款进度、退款金额、退款方式",
            IntentCategory.PAYMENT_ISSUE: "支付失败、支付方式、扣款异常",
            IntentCategory.SHIPPING_INQUIRY: "物流详情、配送范围、运费咨询",
            IntentCategory.ACCOUNT_ISSUE: "密码、登录、绑定、注册、注销",
            IntentCategory.MEMBERSHIP_INQUIRY: "会员等级、积分、权益、优惠",
            IntentCategory.TECHNICAL_ISSUE: "APP故障、网页报错、功能异常",
            IntentCategory.USAGE_GUIDE: "使用方法、操作步骤、功能说明",
            IntentCategory.COMPLAINT: "投诉、不满表达、要求处理",
            IntentCategory.FEEDBACK: "意见建议、产品反馈",
            IntentCategory.TRANSFER_HUMAN: "明确要求转人工客服",
            IntentCategory.GREETING: "问候、寒暄、开场白",
            IntentCategory.CLARIFICATION: "对前文的澄清追问",
            IntentCategory.CHITCHAT: "与业务无关的闲聊",
            IntentCategory.UNKNOWN: "无法归类",
        }
        return descriptions.get(intent, "未定义")

    def _parse_response(self, raw: str, original_text: str) -> RecognitionResult:
        """Parse the JSON response returned by the LLM."""
        try:
            # Extract JSON block
            json_match = re.search(r"\{[\s\S]*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in output")

            data = json.loads(json_match.group())

            # Parse intents
            intents = []
            for item in data.get("intents", []):
                intent_name = item.get("intent", "unknown")
                try:
                    intent = IntentCategory(intent_name)
                except ValueError:
                    intent = IntentCategory.UNKNOWN
                intents.append(IntentResult(
                    intent=intent,
                    confidence=float(item.get("confidence", 0.5)),
                    reason=item.get("reason", ""),
                ))

            if not intents:
                intents.append(IntentResult(
                    intent=IntentCategory.UNKNOWN,
                    confidence=0.3,
                    reason="LLM returned no valid intent",
                ))

            # Parse entities
            entities = []
            for item in data.get("entities", []):
                entities.append(ExtractedEntity(
                    name=item.get("name", ""),
                    value=item.get("value", ""),
                    entity_type=item.get("type", "unknown"),
                    confidence=float(item.get("confidence", 0.8)),
                ))

            # Multi-intent check
            is_multi = data.get("is_multi_intent", len(intents) > 1)

            return RecognitionResult(
                original_text=original_text,
                intents=intents,
                entities=entities,
                is_multi_intent=is_multi,
                primary_intent=intents[0] if intents else None,
                processing_priority=data.get("processing_priority", [i.intent.value for i in intents]),
                method="llm",
            )

        except Exception as e:
            logger.warning(f"[LLMIntentRecognizer] JSON parse failed: {e}, raw={raw[:200]}")
            return RecognitionResult(
                original_text=original_text,
                intents=[IntentResult(intent=IntentCategory.UNKNOWN, confidence=0.1, reason=f"parse_error: {e}")],
                method="llm",
            )


# *** 3. Hybrid intent recognizer (rule + LLM dual-layer strategy)
class HybridIntentRecognizer:
    """
    Hybrid recognizer: rule-first fast path, LLM fallback, and cross-turn intent tracking.

    Strategy: Step 1 rule match (60%) -> Step 2 LLM fallback (35%) -> Step 3 UNKNOWN (5%).
    """

    def __init__(self):
        self.rule_matcher = RuleBasedIntentMatcher()
        self.llm_recognizer = LLMIntentRecognizer()
        self._intent_history: Dict[str, List[IntentResult]] = {}  # session_id -> intent history
        logger.info("[HybridIntentRecognizer] initialized: rule-first -> LLM fallback")

    def recognize(
        self,
        text: str,
        session_id: Optional[str] = None,
        force_llm: bool = False,
    ) -> RecognitionResult:
        """Recognize user intent. Optionally tracks intent across turns via session_id."""
        start_time = time.time()

        # Step 1: rule matching (unless LLM forced)
        if not force_llm:
            rule_result = self.rule_matcher.match(text)
            if rule_result is not None and self._is_rule_result_reliable(rule_result):
                logger.debug(
                    f"[HybridIntentRecognizer] rule hit: "
                    f"{[i.intent.value for i in rule_result.intents]}"
                )

                # Cross-turn enhancement: merge historical intent info
                if session_id:
                    rule_result = self._enhance_with_history(rule_result, session_id)

                rule_result.recognition_time_ms = (time.time() - start_time) * 1000
                self._update_history(session_id, rule_result)
                return rule_result

        # Step 2: LLM recognition
        logger.debug("[HybridIntentRecognizer] rule miss, falling back to LLM")
        llm_result = self.llm_recognizer.recognize(text)

        # Merge rule-extracted entities (rule entities have higher accuracy)
        rule_entities = self.rule_matcher._extract_entities_rule(text)
        existing_names = {e.name for e in llm_result.entities}
        for entity in rule_entities:
            if entity.name not in existing_names:
                llm_result.entities.append(entity)

        # Step 3: cross-turn enhancement
        if session_id:
            llm_result = self._enhance_with_history(llm_result, session_id)

        llm_result.recognition_time_ms = (time.time() - start_time) * 1000
        llm_result.method = "hybrid"
        self._update_history(session_id, llm_result)
        return llm_result

    def recognize_multi_intents_decompose(
        self,
        text: str,
        session_id: Optional[str] = None,
    ) -> List[Tuple[IntentResult, str]]:
        """Decompose a multi-intent message into (intent, sub-query) pairs for independent routing."""
        result = self.recognize(text, session_id)

        if not result.is_multi_intent or len(result.intents) <= 1:
            return [(result.primary_intent or IntentResult(
                intent=IntentCategory.UNKNOWN, confidence=0.1
            ), text)]

        return self._decompose_query(text, result.intents)

    def _decompose_query(
        self,
        text: str,
        intents: List[IntentResult],
    ) -> List[Tuple[IntentResult, str]]:
        """
        Decompose a multi-intent query into independent sub-queries.

        Uses heuristic connector-word splitting; falls back to full-text-per-intent.
        """
        # Try splitting on connector words
        connectors = ["另外", "还有", "同时", "顺便", "此外", "然后", "接着"]
        for conn in connectors:
            if conn in text:
                parts = text.split(conn)
                if len(parts) == len(intents):
                    return list(zip(intents, [p.strip() for p in parts]))

        # Cannot split: each intent gets the full query
        return [(intent, text) for intent in intents]

    def detect_intent_shift(
        self,
        session_id: str,
        current_intent: IntentResult,
    ) -> bool:
        """Return True if the user's intent has shifted from the previous turn."""
        history = self._intent_history.get(session_id, [])
        if len(history) < 2:
            return False

        prev_intent = history[-2]  # turn before last (latest is just-appended)
        return prev_intent.intent != current_intent.intent

    def _is_rule_result_reliable(self, result: RecognitionResult) -> bool:
        """Check whether the rule-based result is reliable enough to return."""
        if not result.intents:
            return False
        # Primary intent confidence is high enough
        if result.intents[0].confidence >= 0.85:
            return True
        # Backed by extracted entities
        if result.entities:
            return True
        return False

    def _enhance_with_history(
        self,
        result: RecognitionResult,
        session_id: str,
    ) -> RecognitionResult:
        """Enrich the recognition result using session intent history."""
        history = self._intent_history.get(session_id, [])

        if not history:
            return result

        # Check if this is a follow-up to the previous answer
        clarification_signals = ["怎么样", "结果呢", "然后呢", "具体", "详细", "能再"]
        if any(sig in result.original_text for sig in clarification_signals):
            # Follow-up scenario: inherit the last intent
            last_intent = history[-1]
            if last_intent.intent != IntentCategory.UNKNOWN:
                result.intents.insert(0, IntentResult(
                    intent=last_intent.intent,
                    confidence=0.60,  # inherited follow-up intent gets lower confidence
                    reason=f"follow-up, inheriting previous intent: {last_intent.intent.value}",
                ))

        return result

    def _update_history(self, session_id: Optional[str], result: RecognitionResult):
        """Append current intents to the session history (keep last 20)."""
        if not session_id:
            return
        if session_id not in self._intent_history:
            self._intent_history[session_id] = []
        for intent in result.intents:
            self._intent_history[session_id].append(intent)
        # Keep only the most recent 20 entries
        if len(self._intent_history[session_id]) > 20:
            self._intent_history[session_id] = self._intent_history[session_id][-20:]

    def cleanup_stale_sessions(self, active_session_ids: set):
        """Remove intent history for sessions no longer active."""
        stale = [sid for sid in list(self._intent_history.keys())
                 if sid not in active_session_ids]
        for sid in stale:
            del self._intent_history[sid]
        if stale:
            logger.debug(f"[HybridIntentRecognizer] Cleaned {len(stale)} stale intent histories")


_recognizer: Optional[HybridIntentRecognizer] = None


def get_intent_recognizer() -> HybridIntentRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = HybridIntentRecognizer()
    return _recognizer


# *** Self-check
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Intent Recognizer — self-check")
    logger.info("=" * 60)

    recognizer = HybridIntentRecognizer()

    test_cases = [
        # (input, expected characteristics)
        ("你好", "greeting"),
        ("我的订单#20240001发货了吗？", "single intent + entity extraction"),
        ("我要退货，顺便问一下会员怎么升级？", "multi-intent"),
        ("密码忘了，物流也查不到，退款什么时候到账？", "three intents"),
        ("刚才那个订单怎么样了？", "follow-up (needs history)"),
        ("你们这个App简直太垃圾了！", "complaint"),
        ("帮我转人工", "human handoff"),
    ]

    for text, desc in test_cases:
        result = recognizer.recognize(text)
        intent_str = ", ".join(
            f"{i.intent.value}({i.confidence:.2f})"
            for i in result.intents
        )
        logger.info(f"\n[test] {desc}")
        logger.info(f"  input: {text}")
        logger.info(f"  intent: {intent_str}")
        logger.info(f"  multi-intent: {result.is_multi_intent}")
        logger.info(f"  entities: {[e.value for e in result.entities]}")
        logger.info(f"  method: {result.method}")
        logger.info(f"  time: {result.recognition_time_ms:.1f}ms")

    logger.info("=" * 60)
    logger.info("Intent recognition self-check complete ✓")
