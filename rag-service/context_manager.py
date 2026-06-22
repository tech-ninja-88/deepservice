"""
DeepService 对话管理 — 上下文管理 (Context Manager)

职责：
  1. 短期记忆 — 滑动窗口保留最近 N 轮完整对话
  2. 长期记忆 — 对话摘要 + 用户画像（偏好、历史标签）
  3. 跨轮次上下文融合 — 将多轮上下文注入 LLM Prompt
  4. 实体追踪 — 跨轮次追踪用户提及的关键实体

Memory model: sliding window (N=10) for recent turns + compressed summaries for older turns.
User profiles are built incrementally across turns. Entities carry across rounds automatically.
"""

import time
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

from loguru import logger

from config import get_config
from session_manager import (
    SessionManager, Message, MessageRole, get_session_manager,
)


# ============================================================================
# 数据结构
# ============================================================================
class EntityType(str, Enum):
    """实体类型"""
    ORDER_ID = "order_id"           # 订单号
    PRODUCT_NAME = "product_name"   # 产品名称
    PHONE_NUMBER = "phone_number"   # 手机号
    EMAIL = "email"                 # 邮箱
    DATE = "date"                   # 日期
    AMOUNT = "amount"               # 金额
    TRACKING_NUMBER = "tracking"    # 物流单号
    ADDRESS = "address"             # 地址
    USERNAME = "username"           # 用户名
    REASON = "reason"               # 原因/意图


@dataclass
class Entity:
    """提取的实体"""
    type: EntityType
    value: str
    confidence: float = 1.0
    first_mentioned_turn: int = 0       # 首次出现的轮次
    last_mentioned_turn: int = 0        # 最后出现的轮次
    metadata: Dict = field(default_factory=dict)


@dataclass
class UserProfile:
    """
    用户画像 — 渐进式构建

    画像维度：
      - 基础属性（设备、语言、渠道）
      - 行为偏好（咨询时段、话题偏好）
      - 情感特征（耐心程度、满意度趋势）
      - 历史标签（高频问题类型、VIP等级）
    """
    user_id: str = "anonymous"

    # 基础属性
    preferred_language: str = "zh-CN"
    device_type: str = "unknown"
    channel: str = "web"

    # 行为统计
    total_sessions: int = 0
    total_messages: int = 0
    avg_session_duration: float = 0.0

    # 话题偏好（意图分布）
    intent_distribution: Dict[str, int] = field(default_factory=dict)

    # 情感特征
    sentiment_scores: List[float] = field(default_factory=list)  # 最近情感分数
    avg_sentiment: float = 0.0
    complaint_count: int = 0            # 投诉次数

    # 历史标签
    frequent_topics: List[str] = field(default_factory=list)     # 高频话题
    vip_level: str = ""
    preferred_contact: str = ""         # 偏好联系方式
    notes: str = ""                     # 备注

    # 时间戳
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update_intent(self, intent: str):
        """更新意图分布"""
        self.intent_distribution[intent] = self.intent_distribution.get(intent, 0) + 1
        # 更新高频话题
        sorted_topics = sorted(
            self.intent_distribution.items(), key=lambda x: x[1], reverse=True
        )
        self.frequent_topics = [t for t, _ in sorted_topics[:5]]

    def update_sentiment(self, sentiment: float):
        """更新情感分数"""
        self.sentiment_scores.append(sentiment)
        if len(self.sentiment_scores) > 20:
            self.sentiment_scores = self.sentiment_scores[-20:]
        self.avg_sentiment = sum(self.sentiment_scores) / len(self.sentiment_scores)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_context_string(self) -> str:
        """生成用于注入 Prompt 的用户画像文本"""
        parts = []
        if self.frequent_topics:
            parts.append(f"常见问题类型: {', '.join(self.frequent_topics[:3])}")
        if self.avg_sentiment < 0.3:
            parts.append("用户近期情绪偏低，请注意安抚")
        if self.vip_level:
            parts.append(f"VIP等级: {self.vip_level}")
        if self.complaint_count > 2:
            parts.append(f"历史投诉次数: {self.complaint_count}，请特别注意服务质量")
        return "\n".join(parts) if parts else ""


@dataclass
class ConversationMemory:
    """
    对话完整记忆（短期 + 长期）

    这是每次 LLM 调用时注入上下文的数据载体。
    """
    # 短期记忆
    recent_messages: List[Message] = field(default_factory=list)     # 最近 N 轮（完整保留）

    # 长期记忆
    summary: str = ""                                                 # 早期对话摘要
    tracked_entities: Dict[str, Entity] = field(default_factory=dict) # 跨轮次实体
    user_profile_text: str = ""                                       # 用户画像文本

    # 元数据
    total_turns: int = 0
    memory_token_estimate: int = 0

    def to_prompt_context(self, max_tokens: int = 3000) -> str:
        """
        将记忆转化为可注入 Prompt 的文本

        按优先级排列（重要信息在前）：
          1. 用户画像
          2. 跨轮次实体
          3. 长期摘要
          4. 短期最近消息
        """
        sections = []

        # 1. 用户画像（高优先级）
        if self.user_profile_text:
            sections.append(f"<用户画像>\n{self.user_profile_text}\n</用户画像>")

        # 2. 跨轮次实体
        if self.tracked_entities:
            entity_lines = []
            for ent in self.tracked_entities.values():
                entity_lines.append(
                    f"  - {ent.type.value}: {ent.value}"
                    f" (首次提及: 第{ent.first_mentioned_turn}轮)"
                )
            sections.append(
                f"<对话中提及的关键信息>\n" + "\n".join(entity_lines) + "\n</关键信息>"
            )

        # 3. 长期摘要
        if self.summary:
            sections.append(f"<历史对话摘要>\n{self.summary}\n</历史对话摘要>")

        # 4. 短期最近消息
        if self.recent_messages:
            recent_text = "\n".join(
                f"[{msg.role.value}]: {msg.content}"
                for msg in self.recent_messages
            )
            sections.append(f"<最近对话>\n{recent_text}\n</最近对话>")

        context = "\n\n".join(sections)

        # Token 估算（中文约 1.5 字符/token，英文约 4 字符/token）
        self.memory_token_estimate = len(context) // 2
        return context


# ============================================================================
# 上下文管理器
# ============================================================================
class ContextManager:
    """
    上下文管理器 — 多轮对话记忆中枢

    核心职责：
      1. 管理滑动窗口（短期记忆）
      2. 触发摘要生成（长期记忆）
      3. 维护用户画像
      4. 追踪跨轮次实体
      5. 构建注入 Prompt 的完整上下文
    """

    def __init__(self, window_size: int = 10, summary_trigger_rounds: int = 8):
        """
        参数:
          window_size: 滑动窗口中保留的完整轮数
          summary_trigger_rounds: 触发摘要生成的轮数阈值
        """
        self.window_size = window_size
        self.summary_trigger_rounds = summary_trigger_rounds
        self.config = get_config()

        # 用户画像存储（user_id → UserProfile）
        self._profiles: Dict[str, UserProfile] = {}

        logger.info(
            f"[ContextManager] 初始化: window={window_size}, "
            f"summary_trigger={summary_trigger_rounds}"
        )

    def build_memory(
        self,
        session_id: str,
        session_manager: Optional[SessionManager] = None,
    ) -> ConversationMemory:
        """
        构建对话记忆（核心方法）

        流程：
          1. 从 SessionManager 获取全部消息
          2. 分离短期窗口内 + 窗口外
          3. 对窗口外消息生成/获取摘要
          4. 追踪实体
          5. 获取用户画像
          6. 返回 ConversationMemory
        """
        sm = session_manager or get_session_manager()
        messages = sm.get_messages(session_id, limit=100)
        turns = self._messages_to_turns(messages)

        total_turns = len(turns)
        memory = ConversationMemory(total_turns=total_turns)

        if not turns:
            return memory

        # 分离短期和长期
        if total_turns <= self.window_size:
            # 全部在窗口内
            memory.recent_messages = messages[-self.window_size * 2:]  # 每条消息独立
        else:
            # 最近 window_size 轮保留完整消息
            recent_turns = turns[-self.window_size:]
            memory.recent_messages = []
            for user_msg, asst_msg in recent_turns:
                memory.recent_messages.append(user_msg)
                if asst_msg:
                    memory.recent_messages.append(asst_msg)

            # 早期轮次生成摘要
            early_turns = turns[:-self.window_size]
            memory.summary = self._get_or_generate_summary(
                session_id, early_turns, sm
            )

        # 追踪实体
        memory.tracked_entities = self._track_entities(turns)

        # 获取用户画像
        session = sm.get_session(session_id)
        if session:
            profile = self.get_user_profile(session.user_id)
            memory.user_profile_text = profile.to_context_string()

        return memory

    def _messages_to_turns(
        self,
        messages: List[Message],
    ) -> List[Tuple[Message, Optional[Message]]]:
        """将消息列表转换为配对的轮次列表"""
        turns = []
        current_user = None

        for msg in messages:
            if msg.role == MessageRole.USER:
                if current_user:
                    turns.append((current_user, None))  # 上一条用户消息无回复
                current_user = msg
            elif msg.role == MessageRole.ASSISTANT and current_user:
                turns.append((current_user, msg))
                current_user = None
            elif msg.role == MessageRole.ASSISTANT:
                turns.append((Message(role=MessageRole.USER, content=""), msg))

        if current_user:
            turns.append((current_user, None))

        return turns

    def _get_or_generate_summary(
        self,
        session_id: str,
        early_turns: List[Tuple[Message, Optional[Message]]],
        sm: SessionManager,
    ) -> str:
        """
        获取或生成摘要

        摘要策略：
          - 优先使用缓存的摘要（Redis/内存中）
          - 无缓存时调用 LLM 生成
          - 摘要长度控制在 200 字以内
        """
        # TODO(redis): 使用 turns_hash 做 Redis 缓存键，避免重复调用 LLM 生成摘要。
        # 当前为简化实现每次都重新生成；生产环境应：
        #   1. 计算 turns_hash
        #   2. 查 Redis cache_key → 命中返回缓存
        #   3. 未命中 → 调用 LLM → 写入 Redis（TTL=会话超时）

        return self._generate_summary(early_turns)

    def _generate_summary(
        self,
        turns: List[Tuple[Message, Optional[Message]]],
    ) -> str:
        """
        使用 LLM 生成对话摘要

        摘要模板要点：
          - 用户的主要需求和问题
          - 已经或部分解决的结论
          - 尚未解决的问题
          - 关键实体信息
        """
        if not turns:
            return ""

        # 构建对话文本
        dialogue_text = "\n".join(
            f"用户: {user_msg.content}\n"
            f"客服: {asst_msg.content if asst_msg else '(等待回复)'}"
            for user_msg, asst_msg in turns
        )

        # Use DeepSeek to generate summary
        try:
            from config import get_llm_client
            client = get_llm_client()
            if client is None:
                return "\n".join(
                    f"User asked: {user_msg.content[:50]}..."
                    for user_msg, _ in turns[-3:]
                )

            prompt = f"""请用2-3句话总结以下客服对话的关键信息。包括：
1. 用户的主要问题或需求
2. 已经解决的问题
3. 未解决的问题
4. 关键信息（如订单号、产品名等）

对话内容：
{dialogue_text}

请直接输出摘要（100字以内）："""

            response = client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的对话摘要生成器。用简洁的中文总结对话。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            summary = response.choices[0].message.content or ""
            logger.debug(f"[ContextManager] 摘要已生成 ({len(summary)} 字符)")
            return summary.strip()

        except Exception as e:
            logger.error(f"[ContextManager] 摘要生成失败: {e}")
            # 降级：直接截取前几条消息
            fallback = "\n".join(
                f"用户曾问: {user_msg.content[:50]}..."
                for user_msg, _ in turns[-3:]
            )
            return fallback

    def _track_entities(
        self,
        turns: List[Tuple[Message, Optional[Message]]],
    ) -> Dict[str, Entity]:
        """
        跨轮次实体追踪

        从所有消息中提取实体，并追踪其在各轮次的分布。
        """
        entities: Dict[str, Dict] = {}  # entity_type:value → Entity

        for turn_idx, (user_msg, asst_msg) in enumerate(turns):
            # 从用户消息中提取实体
            for entity in self._extract_entities(user_msg.content):
                key = f"{entity.type.value}:{entity.value}"
                if key in entities:
                    entities[key]["last_mentioned_turn"] = turn_idx + 1
                    entities[key]["count"] += 1
                else:
                    entities[key] = {
                        "type": entity.type,
                        "value": entity.value,
                        "confidence": entity.confidence,
                        "first_mentioned_turn": turn_idx + 1,
                        "last_mentioned_turn": turn_idx + 1,
                        "count": 1,
                    }

            # 从助手消息中补充
            if asst_msg:
                for entity in self._extract_entities(asst_msg.content):
                    key = f"{entity.type.value}:{entity.value}"
                    if key not in entities:
                        entities[key] = {
                            "type": entity.type,
                            "value": entity.value,
                            "confidence": entity.confidence * 0.8,  # 助手消息置信度略低
                            "first_mentioned_turn": turn_idx + 1,
                            "last_mentioned_turn": turn_idx + 1,
                            "count": 1,
                        }

        # 转换为 Entity 对象
        return {
            key: Entity(
                type=data["type"],
                value=data["value"],
                confidence=data["confidence"],
                first_mentioned_turn=data["first_mentioned_turn"],
                last_mentioned_turn=data["last_mentioned_turn"],
                metadata={"mention_count": data["count"]},
            )
            for key, data in entities.items()
        }

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using shared regex patterns."""
        from entity_extractor import extract_entities

        type_map = {
            "order_id": EntityType.ORDER_ID,
            "phone_number": EntityType.PHONE_NUMBER,
            "tracking_number": EntityType.TRACKING_NUMBER,
            "date": EntityType.DATE,
            "amount": EntityType.AMOUNT,
            "email": EntityType.EMAIL,
        }

        entities = []
        for ext in extract_entities(text):
            etype = type_map.get(ext.entity_type)
            if etype:
                entities.append(Entity(
                    type=etype, value=ext.value,
                    confidence=ext.confidence,
                ))
        return entities

    def _compute_turns_hash(self, turns: List[Tuple[Message, Optional[Message]]]) -> str:
        """计算轮次哈希（用于摘要缓存判断）"""
        content = "|".join(
            user_msg.content + (asst_msg.content if asst_msg else "")
            for user_msg, asst_msg in turns
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    # ──── User Profile 管理 ────

    def get_user_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户画像"""
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    def update_profile_from_turn(
        self,
        user_id: str,
        intent: str,
        sentiment: float,
        session_id: str = "",
    ):
        """根据一轮对话更新用户画像"""
        profile = self.get_user_profile(user_id)
        profile.update_intent(intent)
        profile.update_sentiment(sentiment)
        profile.total_messages += 1
        profile.last_seen = time.time()
        profile.updated_at = time.time()

        if intent == "complaint":
            profile.complaint_count += 1

    def inject_context_to_prompt(
        self,
        base_prompt: str,
        memory: ConversationMemory,
    ) -> str:
        """
        将对话记忆注入到 Prompt 模板中

        这是多轮对话的核心——把记忆上下文融合进 LLM 调用。
        """
        context_text = memory.to_prompt_context()
        if not context_text:
            return base_prompt

        # 在 system prompt 中追加对话记忆
        return f"{base_prompt}\n\n<对话记忆>\n{context_text}\n</对话记忆>"

    def get_relevant_entities_for_query(
        self,
        memory: ConversationMemory,
        current_query: str,
    ) -> List[Entity]:
        """
        获取与当前查询相关的跨轮次实体

        用途：如果用户说"那个订单"，从上文中找到最近提到的订单号。
        """
        relevant = []
        for entity in memory.tracked_entities.values():
            # 检查实体类型或值是否在当前查询中被提及
            if (entity.type.value in current_query.lower() or
                    entity.value.lower() in current_query.lower()):
                relevant.append(entity)
                continue

            # 如果是代词引用（那个/这个），返回最近提及的实体
            if any(word in current_query for word in ["那个", "这个", "它", "该"]):
                if entity.last_mentioned_turn >= memory.total_turns - 2:
                    relevant.append(entity)

        return sorted(relevant, key=lambda e: e.last_mentioned_turn, reverse=True)

    def cleanup_stale_profiles(self, max_age_seconds: int = 604800):
        """Remove user profiles not updated in the last max_age_seconds (default 7 days)."""
        import time
        now = time.time()
        stale = [uid for uid, p in self._profiles.items()
                 if now - p.updated_at > max_age_seconds]
        for uid in stale:
            del self._profiles[uid]
        if stale:
            logger.info(f"[ContextManager] Cleaned {len(stale)} stale profiles")


# Thread-safe lazy init — ContextManager caches user profiles
import threading

_context_manager: Optional[ContextManager] = None
_cm_lock = threading.Lock()


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        with _cm_lock:
            if _context_manager is None:
                _context_manager = ContextManager()
    return _context_manager


# ============================================================================
# 独立测试
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepService Context Manager — 独立测试")
    logger.info("=" * 60)

    # 准备测试数据
    sm = SessionManager()
    session = sm.create_session(user_id="test_user_ctx")
    test_messages = [
        ("user", "我的订单#20240001什么时候发货？"),
        ("assistant", "让我帮您查询订单#20240001。该订单预计明天发货，物流单号将更新到您的账户。"),
        ("user", "好的，那退换货政策是什么？"),
        ("assistant", "自签收之日起7天内可申请退货。质量问题退换货运费由商家承担。"),
        ("user", "那刚才那个订单能改地址吗？"),  # "刚才那个订单" → 指代#20240001
    ]
    for role, content in test_messages:
        sm.append_message(session.id, Message(
            role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
            content=content,
        ))

    cm = ContextManager(window_size=5)

    # 测试1：构建记忆
    memory = cm.build_memory(session.id, sm)
    logger.info(f"[测试1] 对话轮数: {memory.total_turns}")
    logger.info(f"[测试1] 跟踪实体数: {len(memory.tracked_entities)}")
    for key, ent in memory.tracked_entities.items():
        logger.info(f"  实体: {key} — 首次出现在第{ent.first_mentioned_turn}轮")

    # 测试2：获取与"刚才那个订单"相关的实体
    relevant = cm.get_relevant_entities_for_query(memory, "刚才那个订单能改地址吗？")
    logger.info(f"[测试2] 相关实体数: {len(relevant)}")
    for ent in relevant:
        logger.info(f"  {ent.type.value}: {ent.value}")

    # 测试3：上下文生成
    context_text = memory.to_prompt_context()
    logger.info(f"[测试3] 上下文长度: {len(context_text)} 字符, 估算 Token: {memory.memory_token_estimate}")
    logger.info(f"[测试3] 上下文预览:\n{context_text[:300]}...")

    # 测试4：用户画像
    profile = cm.get_user_profile("test_user_ctx")
    cm.update_profile_from_turn("test_user_ctx", "order_status", 0.6, session.id)
    cm.update_profile_from_turn("test_user_ctx", "return_exchange", 0.5, session.id)
    logger.info(f"[测试4] 用户画像: 高频话题={profile.frequent_topics}, 平均情感={profile.avg_sentiment:.2f}")

    logger.info("=" * 60)
    logger.info("上下文管理测试完成 ✓")
