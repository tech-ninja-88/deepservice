"""
=============================================================================
DeepService 对话管理 — 意图识别模块 (Intent Recognizer)
=============================================================================
职责：
  1. 意图分类 — 将用户消息映射到预定义的业务意图
  2. 多意图识别与拆分 — 检测一句话包含的多个意图并拆解为子任务
  3. 实体抽取 — 从消息中提取结构化信息（订单号、产品、日期等）
  4. 意图置信度评估 — 给每个意图分配置信度

企业级设计原则：
  - "快速规则 + LLM兜底" 双层策略：规则覆盖60%高频场景，LLM处理35%复杂场景
  - 多意图拆分是实际业务的核心需求（占比高达37%）
  - 意图分类结果直接影响后续路由和处理策略
  - 跨轮次意图追踪：检测用户意图是否发生了变化

Prompt 工程要点：
  - 显式定义意图类型和边界
  - Few-shot 示例锚定输出格式
  - 要求结构化 JSON 输出
  - 低 temperature 确保一致性

参考：
  [reference:4] — 三段式架构：意图识别-对话管理-回复生成
  [reference:5] — 多意图问题占比37%，多轮对话占比超60%
=============================================================================
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

from loguru import logger

from config import get_config


# ============================================================================
# 意图定义
# ============================================================================
class IntentCategory(str, Enum):
    """
    预定义意图类别

    设计原则：
      - 每个意图对应明确的路由策略
      - 意图粒度适中（太细导致分类不准，太粗导致路由无效）
      - 保留 UNKNOWN 作为兜底
    """
    # 业务咨询
    PRODUCT_INQUIRY = "product_inquiry"         # 产品信息查询
    ORDER_STATUS = "order_status"               # 订单状态查询
    RETURN_EXCHANGE = "return_exchange"         # 退换货咨询
    REFUND_CONSULT = "refund_consult"           # 退款咨询
    PAYMENT_ISSUE = "payment_issue"             # 支付问题
    SHIPPING_INQUIRY = "shipping_inquiry"       # 物流查询
    ACCOUNT_ISSUE = "account_issue"             # 账号问题
    MEMBERSHIP_INQUIRY = "membership_inquiry"   # 会员权益咨询

    # 技术支持
    TECHNICAL_ISSUE = "technical_issue"         # 技术故障
    USAGE_GUIDE = "usage_guide"                 # 使用指南

    # 服务请求
    COMPLAINT = "complaint"                     # 投诉
    FEEDBACK = "feedback"                       # 反馈建议
    TRANSFER_HUMAN = "transfer_human"           # 要求转人工

    # 对话管理
    GREETING = "greeting"                       # 问候寒暄
    CLARIFICATION = "clarification"             # 澄清追问
    CHITCHAT = "chitchat"                       # 闲聊
    UNKNOWN = "unknown"                         # 无法识别

    @classmethod
    def get_routing_intents(cls) -> List[str]:
        """需要路由处理的意图（非通用对话）"""
        return [
            cls.PRODUCT_INQUIRY, cls.ORDER_STATUS, cls.RETURN_EXCHANGE,
            cls.REFUND_CONSULT, cls.PAYMENT_ISSUE, cls.SHIPPING_INQUIRY,
            cls.ACCOUNT_ISSUE, cls.MEMBERSHIP_INQUIRY, cls.TECHNICAL_ISSUE,
            cls.USAGE_GUIDE, cls.COMPLAINT, cls.FEEDBACK,
        ]


@dataclass
class IntentResult:
    """单个意图识别结果"""
    intent: IntentCategory
    confidence: float
    reason: str = ""
    sub_intents: List["IntentResult"] = field(default_factory=list)  # 子意图（多意图场景）

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "sub_intents": [s.to_dict() for s in self.sub_intents] if self.sub_intents else [],
        }


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    name: str                           # 实体名称
    value: str                          # 实体值
    entity_type: str                    # 实体类别（order_id, product, date, etc.）
    confidence: float = 1.0
    start_pos: int = 0                  # 在原文本中的起始位置
    end_pos: int = 0                    # 结束位置
    normalized_value: str = ""          # 标准化后的值

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
    """
    完整识别结果

    包含意图 + 实体 + 元信息
    """
    original_text: str
    intents: List[IntentResult]                     # 按置信度降序排列
    entities: List[ExtractedEntity] = field(default_factory=list)
    is_multi_intent: bool = False                   # 是否多意图
    primary_intent: Optional[IntentResult] = None    # 主意图（多意图时第一个）
    processing_priority: List[str] = field(default_factory=list)  # 处理优先级
    recognition_time_ms: float = 0.0
    method: str = "rule"                            # "rule" | "llm" | "hybrid"

    def get_top_intent(self, min_confidence: float = 0.3) -> Optional[IntentResult]:
        """获取最高置信度意图"""
        if self.intents and self.intents[0].confidence >= min_confidence:
            return self.intents[0]
        return None

    def has_intent(self, intent: IntentCategory) -> bool:
        """检查是否包含指定意图"""
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


# ============================================================================
# 1. 基于规则的快速意图匹配（60% 场景）
# ============================================================================
class RuleBasedIntentMatcher:
    """
    规则匹配器 — 高频、确定性场景的快速通道

    优势：
      - 零延迟（无 API 调用）
      - 100% 确定（不依赖模型状态）
      - 可解释性强
    """

    # 意图规则定义（按优先级排序）
    INTENT_RULES: List[Tuple[List[str], IntentCategory, float, str]] = [
        # (关键词列表, 意图, 置信度, 说明)
        (["你好", "嗨", "hello", "hi", "在吗", "您好", "早上好", "下午好", "晚上好"],
         IntentCategory.GREETING, 0.98, "问候触发"),

        (["人工", "转人工", "客服人员", "人工客服", "真人", "转接"],
         IntentCategory.TRANSFER_HUMAN, 0.95, "用户明确要求转人工"),

        (["投诉", "举报", "垃圾", "太差", "很差", "坑人", "骗子", "差评", "态度"],
         IntentCategory.COMPLAINT, 0.92, "投诉关键词"),

        (["退货", "退换", "换货", "退回去", "退掉"],
         IntentCategory.RETURN_EXCHANGE, 0.90, "退换货关键词"),

        (["退款", "退钱", "返款", "退费", "打款"],
         IntentCategory.REFUND_CONSULT, 0.90, "退款关键词"),

        (["订单", "发货", "送达", "物流", "快递", "运单", "tracking", "到哪"],
         IntentCategory.ORDER_STATUS, 0.88, "订单和物流状态"),

        (["支付", "付款", "扣款", "刷卡", "银行卡", "支付宝", "微信支付"],
         IntentCategory.PAYMENT_ISSUE, 0.88, "支付相关问题"),

        (["密码", "登录", "注册", "绑定", "手机号", "验证码", "账号", "注销"],
         IntentCategory.ACCOUNT_ISSUE, 0.88, "账号相关问题"),

        (["会员", "vip", "积分", "折扣", "等级", "权益", "升级"],
         IntentCategory.MEMBERSHIP_INQUIRY, 0.87, "会员权益相关"),

        (["怎么", "如何", "怎样", "为什么", "是什么", "什么原因"],
         IntentCategory.USAGE_GUIDE, 0.80, "使用方法询问（弱信号）"),

        (["报错", "错误", "bug", "闪退", "卡顿", "打不开", "连不上", "安装"],
         IntentCategory.TECHNICAL_ISSUE, 0.87, "技术故障关键词"),
    ]

    # 多意图连接词
    MULTI_INTENT_CONNECTORS = [
        "另外", "还有", "同时", "顺便", "此外", "以及",
        "然后", "接着", "再", "也", "并且", "和",
        "第一个", "第二个", "一是", "二是", "一方面", "另一方面",
    ]

    def match(self, text: str) -> Optional[RecognitionResult]:
        """
        快速规则匹配

        返回 None 表示规则无法匹配，需降级到 LLM。
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
                    break  # 一组规则命中一次即可

        if not matched_intents:
            return None  # 规则无法匹配

        # 去重（同一意图可能被多条规则命中）
        seen = set()
        unique_intents = []
        for ir in matched_intents:
            if ir.intent.value not in seen:
                seen.add(ir.intent.value)
                unique_intents.append(ir)

        # 多意图检测
        is_multi = len(unique_intents) > 1
        if not is_multi:
            # 检查是否含有多意图连接词（可能规则没检测到的多意图）
            if any(conn in text_lower for conn in self.MULTI_INTENT_CONNECTORS):
                is_multi = True

        # 构建结果
        unique_intents.sort(key=lambda x: x.confidence, reverse=True)

        # 实体抽取（规则版）
        entities = self._extract_entities_rule(text)

        return RecognitionResult(
            original_text=text,
            intents=unique_intents,
            entities=entities,
            is_multi_intent=is_multi,
            primary_intent=unique_intents[0] if unique_intents else None,
            processing_priority=[i.intent.value for i in unique_intents],
            method="rule",
        )

    def _extract_entities_rule(self, text: str) -> List[ExtractedEntity]:
        """基于规则的快速实体抽取"""
        entities = []

        # 订单号
        for match in re.finditer(r"(?:订单|#|No\.)\s*(\d{6,20})", text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                name="订单号", value=match.group(0),
                entity_type="order_id", confidence=0.90,
                start_pos=match.start(), end_pos=match.end(),
            ))

        # 手机号
        for match in re.finditer(r"1[3-9]\d{9}", text):
            entities.append(ExtractedEntity(
                name="手机号", value=match.group(0),
                entity_type="phone_number", confidence=0.95,
                start_pos=match.start(), end_pos=match.end(),
            ))

        # 快递单号
        for match in re.finditer(r"(?:SF|YT|YD|DB|ZTO|STO)\d{8,15}", text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                name="快递单号", value=match.group(0),
                entity_type="tracking_number", confidence=0.90,
                start_pos=match.start(), end_pos=match.end(),
            ))

        # 日期
        for match in re.finditer(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
            entities.append(ExtractedEntity(
                name="日期", value=match.group(0),
                entity_type="date", confidence=0.90,
                start_pos=match.start(), end_pos=match.end(),
                normalized_value=match.group(0).replace("/", "-"),
            ))

        # 金额
        for match in re.finditer(r"(\d+\.?\d*)\s*(元|块|美元|USD|CNY|¥)", text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                name="金额", value=match.group(0),
                entity_type="amount", confidence=0.85,
                start_pos=match.start(), end_pos=match.end(),
                normalized_value=match.group(1),
            ))

        return entities


# ============================================================================
# 2. LLM 意图识别（35% 复杂场景）
# ============================================================================
class LLMIntentRecognizer:
    """
    基于 DeepSeek 的 LLM 意图识别

    使用场景：
      - 规则无法匹配的模糊/复杂输入
      - 多意图细粒度拆分
      - 需要语义理解的意图分类
    """

    # Few-shot 示例（关键：锚定输出格式）
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
            from openai import OpenAI
            from config import get_api_key
            api_key = get_api_key()
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY 未设置")
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.llm.base_url,
            )
        return self._client

    def recognize(self, text: str) -> RecognitionResult:
        """
        LLM 意图识别
        """
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,  # 低温度确保一致性
                max_tokens=500,
            )

            result_text = response.choices[0].message.content or "{}"
            parsed = self._parse_response(result_text, text)

        except Exception as e:
            logger.error(f"[LLMIntentRecognizer] 识别失败: {e}")
            parsed = RecognitionResult(
                original_text=text,
                intents=[IntentResult(intent=IntentCategory.UNKNOWN, confidence=0.1, reason=f"LLM error: {e}")],
                method="llm",
            )

        parsed.recognition_time_ms = (time.time() - start_time) * 1000
        return parsed

    def _build_system_prompt(self) -> str:
        """构建意图识别的 System Prompt"""
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
        """获取意图描述"""
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
        """解析 LLM 返回的 JSON"""
        try:
            # 提取 JSON 块
            json_match = re.search(r"\{[\s\S]*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("未找到 JSON 输出")

            data = json.loads(json_match.group())

            # 解析意图
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
                    reason="LLM 未返回有效意图",
                ))

            # 解析实体
            entities = []
            for item in data.get("entities", []):
                entities.append(ExtractedEntity(
                    name=item.get("name", ""),
                    value=item.get("value", ""),
                    entity_type=item.get("type", "unknown"),
                    confidence=float(item.get("confidence", 0.8)),
                ))

            # 多意图判断
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
            logger.warning(f"[LLMIntentRecognizer] JSON 解析失败: {e}, raw={raw[:200]}")
            return RecognitionResult(
                original_text=original_text,
                intents=[IntentResult(intent=IntentCategory.UNKNOWN, confidence=0.1, reason=f"parse_error: {e}")],
                method="llm",
            )


# ============================================================================
# 3. 混合意图识别器（规则 + LLM 双层策略）
# ============================================================================
class HybridIntentRecognizer:
    """
    混合意图识别器 — 生产环境主入口

    策略：
      Step 1: 规则快速匹配 → 高置信度命中直接返回（60% 场景）
      Step 2: 规则无匹配 → 降级 LLM 识别（35% 场景）
      Step 3: LLM 也失败 → UNKNOWN（5% 场景）

    跨轮次意图追踪：
      - 检测用户意图是否与上一轮一致（延续）
      - 检测用户意图是否转向（intent shift）
    """

    def __init__(self):
        self.rule_matcher = RuleBasedIntentMatcher()
        self.llm_recognizer = LLMIntentRecognizer()
        self._intent_history: Dict[str, List[IntentResult]] = {}  # session_id → 意图历史
        logger.info("[HybridIntentRecognizer] 初始化: 规则优先 → LLM 兜底")

    def recognize(
        self,
        text: str,
        session_id: Optional[str] = None,
        force_llm: bool = False,
    ) -> RecognitionResult:
        """
        识别用户意图

        参数:
          text: 用户输入文本
          session_id: 会话 ID（用于跨轮次意图追踪）
          force_llm: 强制使用 LLM（跳过规则）

        返回:
          RecognitionResult
        """
        start_time = time.time()

        # Step 1: 规则匹配（除非强制 LLM）
        if not force_llm:
            rule_result = self.rule_matcher.match(text)
            if rule_result is not None and self._is_rule_result_reliable(rule_result):
                logger.debug(
                    f"[HybridIntentRecognizer] 规则命中: "
                    f"{[i.intent.value for i in rule_result.intents]}"
                )

                # 跨轮次增强：融合历史意图信息
                if session_id:
                    rule_result = self._enhance_with_history(rule_result, session_id)

                rule_result.recognition_time_ms = (time.time() - start_time) * 1000
                self._update_history(session_id, rule_result)
                return rule_result

        # Step 2: LLM 识别
        logger.debug("[HybridIntentRecognizer] 规则未命中，降级 LLM 识别")
        llm_result = self.llm_recognizer.recognize(text)

        # 融合规则抽取的实体（规则实体准确度高）
        rule_entities = self.rule_matcher._extract_entities_rule(text)
        existing_names = {e.name for e in llm_result.entities}
        for entity in rule_entities:
            if entity.name not in existing_names:
                llm_result.entities.append(entity)

        # Step 3: 跨轮次增强
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
        """
        多意图拆分：将一个消息拆分为多个子任务

        每个子任务 = (意图, 去除了其他意图后的子查询)

        用途：后续对每个子任务独立路由处理
        """
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
        将多意图查询分解为独立的子查询

        启发式规则：
          - 检查多意图连接词位置进行切分
          - 无法切分时为每个意图保留完整查询
        """
        # 尝试按连接词切分
        connectors = ["另外", "还有", "同时", "顺便", "此外", "然后", "接着"]
        for conn in connectors:
            if conn in text:
                parts = text.split(conn)
                if len(parts) == len(intents):
                    return list(zip(intents, [p.strip() for p in parts]))

        # 无法切分：每个意图对应完整查询
        return [(intent, text) for intent in intents]

    def detect_intent_shift(
        self,
        session_id: str,
        current_intent: IntentResult,
    ) -> bool:
        """
        检测意图转移

        返回 True 表示用户意图发生了变化（需要调整对话策略）。
        """
        history = self._intent_history.get(session_id, [])
        if len(history) < 2:
            return False

        prev_intent = history[-2]  # 上上一轮（history 最新项是刚追加的）
        return prev_intent.intent != current_intent.intent

    def _is_rule_result_reliable(self, result: RecognitionResult) -> bool:
        """判断规则结果是否可靠"""
        if not result.intents:
            return False
        # 主意图置信度足够高
        if result.intents[0].confidence >= 0.85:
            return True
        # 有实体支撑
        if result.entities:
            return True
        return False

    def _enhance_with_history(
        self,
        result: RecognitionResult,
        session_id: str,
    ) -> RecognitionResult:
        """利用历史意图增强识别结果"""
        history = self._intent_history.get(session_id, [])

        if not history:
            return result

        # 检查是否为对上轮回答的追问
        clarification_signals = ["怎么样", "结果呢", "然后呢", "具体", "详细", "能再"]
        if any(sig in result.original_text for sig in clarification_signals):
            # 追问场景：保持上轮意图
            last_intent = history[-1]
            if last_intent.intent != IntentCategory.UNKNOWN:
                result.intents.insert(0, IntentResult(
                    intent=last_intent.intent,
                    confidence=0.60,  # 追问继承的意图置信度降低
                    reason=f"追问，继承上轮意图: {last_intent.intent.value}",
                ))

        return result

    def _update_history(self, session_id: Optional[str], result: RecognitionResult):
        """更新意图历史"""
        if not session_id:
            return
        if session_id not in self._intent_history:
            self._intent_history[session_id] = []
        for intent in result.intents:
            self._intent_history[session_id].append(intent)
        # 只保留最近 20 条
        if len(self._intent_history[session_id]) > 20:
            self._intent_history[session_id] = self._intent_history[session_id][-20:]


# ============================================================================
# 全局单例
# ============================================================================
import threading

_recognizer: Optional[HybridIntentRecognizer] = None
_recognizer_lock = threading.Lock()


def get_intent_recognizer() -> HybridIntentRecognizer:
    global _recognizer
    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                _recognizer = HybridIntentRecognizer()
    return _recognizer


# ============================================================================
# 独立测试
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepService Intent Recognizer — 独立测试")
    logger.info("=" * 60)

    recognizer = HybridIntentRecognizer()

    test_cases = [
        # (输入, 期望特征)
        ("你好", "问候"),
        ("我的订单#20240001发货了吗？", "单意图 + 实体抽取"),
        ("我要退货，顺便问一下会员怎么升级？", "多意图"),
        ("密码忘了，物流也查不到，退款什么时候到账？", "三意图"),
        ("刚才那个订单怎么样了？", "追问（需历史）"),
        ("你们这个App简直太垃圾了！", "投诉"),
        ("帮我转人工", "转人工"),
    ]

    for text, desc in test_cases:
        result = recognizer.recognize(text)
        intent_str = ", ".join(
            f"{i.intent.value}({i.confidence:.2f})"
            for i in result.intents
        )
        logger.info(f"\n[测试] {desc}")
        logger.info(f"  输入: {text}")
        logger.info(f"  意图: {intent_str}")
        logger.info(f"  多意图: {result.is_multi_intent}")
        logger.info(f"  实体: {[e.value for e in result.entities]}")
        logger.info(f"  方法: {result.method}")
        logger.info(f"  耗时: {result.recognition_time_ms:.1f}ms")

    logger.info("=" * 60)
    logger.info("意图识别测试完成 ✓")
