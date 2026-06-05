"""
=============================================================================
DeepService 对话管理 — 人工转接模块 (Human Transfer)
=============================================================================
职责：
  1. 转人工触发条件判断（多维度触发机制）
  2. 对话上下文的完整打包和透传
  3. WebSocket 实时消息通道
  4. 人工坐席接管与归还

企业级设计原则：
  - 六种触发条件覆盖所有转人工场景
  - 上下文透传确保人工不需要重复询问
  - 转接过程对用户透明，无需刷新页面
  - 支持排队、超时、拒绝等异常情况

触发条件：
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 用户主动要求（回复"人工"、"转人工"等）                     │
  │ 2. 关键词触发（投诉、紧急、生命危险等敏感词）                  │
  │ 3. 置信度低（连续2轮低置信度+追问无效）                       │
  │ 4. 负面情感累积（连续3轮负面情感）                            │
  │ 5. 特定意图强制转接（投诉、账号恢复等）                       │
  │ 6. 结构化流程失败（槽位填充3次失败）                          │
  └─────────────────────────────────────────────────────────────┘

WebSocket 通道设计：
  用户端 ←→ WebSocket ←→ 坐席端
    │                        │
    │  send(message)         │  receive(message)
    │  on_message(callback)  │  send(message)
    │                        │
    └──────── WebSocket Server ─────────┘
              消息路由 + 状态同步
=============================================================================
"""

import json
import time
import uuid
import queue
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum

from loguru import logger

from config import get_config


# ============================================================================
# 枚举与常量
# ============================================================================
class TransferTrigger(str, Enum):
    """转接触发原因"""
    USER_REQUESTED = "user_requested"             # 用户主动要求
    KEYWORD_DETECTED = "keyword_detected"         # 敏感关键词
    LOW_CONFIDENCE = "low_confidence"             # 置信度不足
    NEGATIVE_SENTIMENT = "negative_sentiment"     # 负面情感累积
    INTENT_FORCED = "intent_forced"               # 意图强制转接
    FLOW_FAILED = "flow_failed"                   # 流程失败
    ADMIN_TRANSFER = "admin_transfer"             # 管理员主动接入


class TransferStatus(str, Enum):
    """转接状态"""
    PENDING = "pending"               # 等待坐席接听
    QUEUED = "queued"                 # 排队中
    CONNECTED = "connected"           # 已接通
    AGENT_DISCONNECTED = "disconnected"  # 坐席断连
    COMPLETED = "completed"           # 已完成
    TIMEOUT = "timeout"               # 超时
    REJECTED = "rejected"             # 被拒绝
    ERROR = "error"                   # 异常


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class TransferContext:
    """
    转接上下文 — 完整打包给人工坐席的信息

    设计原则：人工坐席看到这个不需要再问任何基本信息。
    """
    # 基本信息
    transfer_id: str = field(default_factory=lambda: f"transfer_{uuid.uuid4().hex[:10]}")
    session_id: str = ""
    user_id: str = "anonymous"
    trigger: TransferTrigger = TransferTrigger.USER_REQUESTED

    # 对话摘要（给坐席的"预习材料"）
    conversation_summary: str = ""
    recent_messages: List[Dict] = field(default_factory=list)  # 最近 5 轮
    current_intent: str = ""
    tracked_entities: Dict[str, str] = field(default_factory=dict)

    # 用户信息
    user_profile_text: str = ""
    user_sentiment_avg: float = 0.5
    complaint_history: int = 0

    # 处理信息
    priority: int = 0                   # 优先级（0=紧急, 1=普通, 2=低）
    created_at: float = field(default_factory=time.time)
    status: TransferStatus = TransferStatus.PENDING
    assigned_agent: str = ""            # 分配的坐席 ID
    agent_name: str = ""

    # 元数据
    channel: str = "web"
    user_agent: str = ""
    ip_address: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "transfer_id": self.transfer_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "trigger": self.trigger.value,
            "conversation_summary": self.conversation_summary,
            "recent_messages": self.recent_messages,
            "current_intent": self.current_intent,
            "tracked_entities": self.tracked_entities,
            "user_profile_text": self.user_profile_text,
            "user_sentiment_avg": self.user_sentiment_avg,
            "complaint_history": self.complaint_history,
            "priority": self.priority,
            "created_at": self.created_at,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "agent_name": self.agent_name,
            "notes": self.notes,
        }

    def to_agent_display(self) -> str:
        """生成给坐席看的格式化显示"""
        lines = [
            "=" * 40,
            f"转接请求 #{self.transfer_id[:10]}",
            f"触发原因: {self.trigger.value}",
            f"优先级: {'🔴 紧急' if self.priority == 0 else '🟡 普通' if self.priority == 1 else '🟢 低'}",
            "",
            "【对话摘要】",
            self.conversation_summary or "（无）",
            "",
            "【最近对话】",
        ]

        for msg in self.recent_messages[-6:]:
            role_icon = "👤" if msg.get("role") in ("user", "用户") else "🤖"
            lines.append(f"  {role_icon} {msg.get('content', '')[:100]}")

        if self.tracked_entities:
            lines.append("")
            lines.append("【关键信息】")
            for k, v in self.tracked_entities.items():
                lines.append(f"  • {k}: {v}")

        lines.append("")
        lines.append(f"用户情感: {self.user_sentiment_avg:.2f} (0=负面, 1=正面)")
        lines.append(f"历史投诉: {self.complaint_history} 次")
        lines.append("=" * 40)

        return "\n".join(lines)


@dataclass
class AgentInfo:
    """坐席信息"""
    agent_id: str
    name: str = ""
    skills: List[str] = field(default_factory=list)  # 技能标签：售前、售后、技术
    status: str = "offline"                           # online / busy / offline
    current_sessions: int = 0
    max_sessions: int = 5


# ============================================================================
# WebSocket 消息（与具体框架解耦）
# ============================================================================
@dataclass
class WSMessage:
    """WebSocket 消息结构（框架无关）"""
    type: str                           # message / system / status / typing / close
    sender: str                         # "user" | "agent" | "system"
    content: str = ""
    transfer_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "WSMessage":
        data = json.loads(json_str)
        return cls(**data)


# ============================================================================
# 触发条件检测器
# ============================================================================
class TransferTriggerDetector:
    """
    转人工触发条件检测器

    六维度检测，任一命中即触发转接。
    """

    # 用户主动转接触发词
    USER_REQUEST_KEYWORDS = [
        "人工", "转人工", "客服人员", "人工客服",
        "真人", "转接", "人工坐席", "找人工",
        "有人吗", "不是机器人",
    ]

    # 敏感/紧急关键词（直接触发转接）
    URGENT_KEYWORDS = [
        "生命危险", "人身安全", "报警", "诈骗",
        "法院", "律师函", "消协", "315",
        "媒体曝光", "曝光你们", "记者",
    ]

    # 负面情感词
    NEGATIVE_SENTIMENT_WORDS = [
        "气死", "垃圾", "骗子", "太差", "坑人",
        "失望", "投诉", "举报", "差评", "退款",
        "再也不用", "再也不买",
    ]

    def __init__(self):
        # 会话级别计数器
        self._negative_count: Dict[str, int] = {}           # session_id → 负面计数
        self._low_confidence_count: Dict[str, int] = {}     # session_id → 低置信度计数
        self._clarify_count: Dict[str, int] = {}            # session_id → 追问次数

    def detect(
        self,
        session_id: str,
        user_message: str,
        intent: str = "",
        confidence: float = 1.0,
        sentiment: float = 0.5,
    ) -> Optional[TransferTrigger]:
        """
        多维度检测是否需要转人工

        返回触发的 TransferTrigger，或 None（不需要转接）。
        """
        msg_lower = user_message.lower()

        # 条件1: 用户主动要求
        if any(kw in msg_lower for kw in self.USER_REQUEST_KEYWORDS):
            logger.info(f"[TransferTrigger] 用户主动请求转人工: {session_id[:12]}")
            return TransferTrigger.USER_REQUESTED

        # 条件2: 紧急/敏感关键词
        if any(kw in msg_lower for kw in self.URGENT_KEYWORDS):
            logger.warning(f"[TransferTrigger] 敏感关键词触发: {session_id[:12]}")
            return TransferTrigger.KEYWORD_DETECTED

        # 条件3: 特定意图强制转接
        if intent in ("complaint", "transfer_human"):
            logger.info(f"[TransferTrigger] 意图强制转接: intent={intent}")
            return TransferTrigger.INTENT_FORCED

        # 条件4: 连续低置信度
        if confidence < 0.4:
            self._low_confidence_count[session_id] = (
                self._low_confidence_count.get(session_id, 0) + 1
            )
            if self._low_confidence_count.get(session_id, 0) >= 3:
                logger.warning(
                    f"[TransferTrigger] 连续低置信度触发: "
                    f"count={self._low_confidence_count[session_id]}"
                )
                return TransferTrigger.LOW_CONFIDENCE
        else:
            self._low_confidence_count[session_id] = 0  # 重置

        # 条件5: 连续负面情感
        if sentiment < 0.3:
            self._negative_count[session_id] = (
                self._negative_count.get(session_id, 0) + 1
            )
            if self._negative_count.get(session_id, 0) >= 3:
                logger.warning(
                    f"[TransferTrigger] 负面情感累积触发: "
                    f"count={self._negative_count[session_id]}"
                )
                return TransferTrigger.NEGATIVE_SENTIMENT
        else:
            self._negative_count[session_id] = max(
                0, self._negative_count.get(session_id, 0) - 1
            )  # 逐渐减少

        return None

    def reset_counts(self, session_id: str):
        """重置会话计数器（转接完成后）"""
        self._negative_count.pop(session_id, None)
        self._low_confidence_count.pop(session_id, None)
        self._clarify_count.pop(session_id, None)


# ============================================================================
# 坐席管理器
# ============================================================================
class AgentManager:
    """
    坐席管理器

    管理在线坐席、分配任务、排队。
    """

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._transfer_queue: List[TransferContext] = []
        self._active_transfers: Dict[str, TransferContext] = {}  # transfer_id → context
        self._lock = threading.Lock()
        logger.info("[AgentManager] 坐席管理器初始化")

    def register_agent(self, agent: AgentInfo):
        """注册坐席"""
        with self._lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"[AgentManager] 坐席注册: {agent.agent_id} ({agent.name})")

    def unregister_agent(self, agent_id: str):
        """注销坐席"""
        with self._lock:
            self._agents.pop(agent_id, None)
            logger.info(f"[AgentManager] 坐席注销: {agent_id}")

    def assign_transfer(self, transfer: TransferContext) -> Optional[str]:
        """
        分配转接请求给坐席

        分配策略：
          1. 按技能匹配
          2. 按负载均衡（当前处理数最少的优先）
          3. 无可用坐席时加入队列

        返回坐席 ID 或 None（已排队）
        """
        with self._lock:
            # 查找空闲且技能匹配的坐席
            available = [
                a for a in self._agents.values()
                if a.status == "online"
                and a.current_sessions < a.max_sessions
            ]

            if not available:
                # 排队
                transfer.status = TransferStatus.QUEUED
                self._transfer_queue.append(transfer)
                logger.info(
                    f"[AgentManager] 无可用坐席，进入排队 "
                    f"(队列长度: {len(self._transfer_queue)})"
                )
                return None

            # 按技能匹配排序
            available.sort(key=lambda a: (
                self._skill_match_score(a, transfer),
                -a.current_sessions,  # 负载少的优先
            ))

            # 分配给最匹配的坐席
            agent = available[0]
            agent.current_sessions += 1
            transfer.status = TransferStatus.PENDING
            transfer.assigned_agent = agent.agent_id
            transfer.agent_name = agent.name

            self._active_transfers[transfer.transfer_id] = transfer
            logger.info(
                f"[AgentManager] 转接 {transfer.transfer_id[:10]} "
                f"分配给坐席 {agent.agent_id}"
            )
            return agent.agent_id

    def complete_transfer(self, transfer_id: str):
        """完成转接"""
        with self._lock:
            transfer = self._active_transfers.pop(transfer_id, None)
            if transfer:
                transfer.status = TransferStatus.COMPLETED
                # 释放坐席
                if transfer.assigned_agent in self._agents:
                    self._agents[transfer.assigned_agent].current_sessions -= 1

    def reject_transfer(self, transfer_id: str, reason: str = ""):
        """坐席拒绝转接 — 重新分配"""
        with self._lock:
            transfer = self._active_transfers.pop(transfer_id, None)
            if transfer:
                transfer.status = TransferStatus.REJECTED
                transfer.notes = f"拒绝原因: {reason}"
                # 重新排队
                transfer.assigned_agent = ""
                self._transfer_queue.insert(0, transfer)

    def get_queue_length(self) -> int:
        return len(self._transfer_queue)

    def get_online_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == "online")

    def _skill_match_score(self, agent: AgentInfo, transfer: TransferContext) -> int:
        """计算坐席技能匹配得分"""
        score = 0
        tags_lower = [t.lower() for t in transfer.tags]
        for skill in agent.skills:
            if skill.lower() in tags_lower:
                score += 1
        return -score  # 负数让 .sort() 时高分在前


# ============================================================================
# WebSocket 消息通道（框架无关抽象）
# ============================================================================
class WebSocketChannel:
    """
    WebSocket 消息通道

    框架无关的设计（可以挂接到 FastAPI / Flask-SocketIO / Django Channels）。

    通道模式：
      - system → user:  系统通知用户（排队、转接成功）
      - user → agent:   用户消息透传给坐席
      - agent → user:   坐席消息透传给用户
      - system → agent: 系统通知坐席（新转接、用户状态）
    """

    def __init__(self):
        # 连接注册表
        self._user_connections: Dict[str, Any] = {}      # session_id → ws connection
        self._agent_connections: Dict[str, Any] = {}     # agent_id → ws connection
        self._connection_lock = threading.Lock()

        # 消息处理回调
        self._on_user_message: Optional[Callable] = None
        self._on_agent_message: Optional[Callable] = None

        logger.info("[WebSocketChannel] 消息通道初始化")

    def register_user(self, session_id: str, connection: Any):
        """注册用户 WebSocket 连接"""
        with self._connection_lock:
            self._user_connections[session_id] = connection
            logger.debug(f"[WebSocketChannel] 用户上线: {session_id[:12]}")

    def register_agent(self, agent_id: str, connection: Any):
        """注册坐席 WebSocket 连接"""
        with self._connection_lock:
            self._agent_connections[agent_id] = connection
            logger.debug(f"[WebSocketChannel] 坐席上线: {agent_id}")

    def unregister_user(self, session_id: str):
        """注销用户连接"""
        with self._connection_lock:
            self._user_connections.pop(session_id, None)

    def unregister_agent(self, agent_id: str):
        """注销坐席连接"""
        with self._connection_lock:
            self._agent_connections.pop(agent_id, None)

    def send_to_user(self, session_id: str, message: WSMessage) -> bool:
        """发送消息给用户"""
        conn = self._user_connections.get(session_id)
        if conn:
            try:
                self._ws_send(conn, message.to_json())
                return True
            except Exception as e:
                logger.error(f"[WebSocketChannel] 发送用户消息失败: {e}")
                return False

        # 离线消息处理（生产环境：存储到 Redis，用户上线后推送）
        logger.debug(f"[WebSocketChannel] 用户 {session_id[:12]} 离线，消息暂存")
        return False

    def send_to_agent(self, agent_id: str, message: WSMessage) -> bool:
        """发送消息给坐席"""
        conn = self._agent_connections.get(agent_id)
        if conn:
            try:
                self._ws_send(conn, message.to_json())
                return True
            except Exception as e:
                logger.error(f"[WebSocketChannel] 发送坐席消息失败: {e}")
                return False
        return False

    def broadcast_to_agents(self, message: WSMessage):
        """广播给所有在线坐席"""
        for agent_id, conn in list(self._agent_connections.items()):
            try:
                self._ws_send(conn, message.to_json())
            except Exception:
                pass

    def set_on_user_message(self, callback: Callable):
        """设置用户消息处理回调"""
        self._on_user_message = callback

    def set_on_agent_message(self, callback: Callable):
        """设置坐席消息处理回调"""
        self._on_agent_message = callback

    def handle_user_message(self, session_id: str, raw_message: str):
        """处理来自用户的消息"""
        try:
            msg = WSMessage.from_json(raw_message)
            if self._on_user_message:
                self._on_user_message(session_id, msg)
        except Exception as e:
            logger.error(f"[WebSocketChannel] 处理用户消息失败: {e}")

    def handle_agent_message(self, agent_id: str, raw_message: str):
        """处理来自坐席的消息"""
        try:
            msg = WSMessage.from_json(raw_message)
            if self._on_agent_message:
                self._on_agent_message(agent_id, msg)
        except Exception as e:
            logger.error(f"[WebSocketChannel] 处理坐席消息失败: {e}")

    def _ws_send(self, connection: Any, message: str):
        """
        发送 WebSocket 消息

        子类可重写此方法适配不同框架。
        """
        # FastAPI WebSocket
        if hasattr(connection, "send_text"):
            # 使用 asyncio
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import asyncio
                    asyncio.create_task(connection.send_text(message))
                else:
                    loop.run_until_complete(connection.send_text(message))
            except RuntimeError:
                pass
        # 其他框架的通用回退
        elif hasattr(connection, "send"):
            connection.send(message)


# ============================================================================
# 人工转接服务（统一门面）
# ============================================================================
class HumanTransferService:
    """
    人工转接服务 — 对外统一接口

    整合了转接触发检测、坐席分配、会话透传、WebSocket消息通道。
    """

    def __init__(self):
        self.detector = TransferTriggerDetector()
        self.agent_manager = AgentManager()
        self.ws = WebSocketChannel()
        self.config = get_config()

        # 转接历史
        self._transfer_history: Dict[str, TransferContext] = {}

        logger.info("[HumanTransferService] 初始化完成")

    def check_trigger(
        self,
        session_id: str,
        user_message: str,
        intent: str = "",
        confidence: float = 1.0,
        sentiment: float = 0.5,
    ) -> Optional[TransferTrigger]:
        """检查是否需要触发转人工"""
        return self.detector.detect(
            session_id, user_message, intent, confidence, sentiment
        )

    def initiate_transfer(
        self,
        session_id: str,
        trigger: TransferTrigger,
        user_id: str = "anonymous",
        conversation_summary: str = "",
        recent_messages: List[Dict] = None,
        tracked_entities: Dict[str, str] = None,
        user_sentiment_avg: float = 0.5,
        complaint_history: int = 0,
        **kwargs,
    ) -> TransferContext:
        """
        发起转人工

        1. 打包完整上下文
        2. 分配坐席
        3. WebSocket 通知
        """
        # 构建转接上下文
        transfer = TransferContext(
            session_id=session_id,
            user_id=user_id,
            trigger=trigger,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages or [],
            tracked_entities=tracked_entities or {},
            user_sentiment_avg=user_sentiment_avg,
            complaint_history=complaint_history,
            priority=0 if trigger in (TransferTrigger.KEYWORD_DETECTED, TransferTrigger.NEGATIVE_SENTIMENT)
            else 1,
            **kwargs,
        )

        # 分配坐席
        agent_id = self.agent_manager.assign_transfer(transfer)

        # 保存记录
        self._transfer_history[transfer.transfer_id] = transfer

        # 通知用户
        if agent_id:
            # 已分配坐席 — 通知用户
            self.ws.send_to_user(session_id, WSMessage(
                type="system",
                sender="system",
                content=f"已为您接通人工客服，坐席 {transfer.agent_name} 将为您服务。",
                transfer_id=transfer.transfer_id,
                metadata={"agent_id": agent_id, "agent_name": transfer.agent_name},
            ))

            # 通知坐席
            self.ws.send_to_agent(agent_id, WSMessage(
                type="system",
                sender="system",
                content="新的客户转接请求",
                transfer_id=transfer.transfer_id,
                metadata={
                    "transfer_context": transfer.to_dict(),
                    "display": transfer.to_agent_display(),
                },
            ))
        else:
            # 排队中 — 通知用户等待
            queue_pos = self.agent_manager.get_queue_length()
            self.ws.send_to_user(session_id, WSMessage(
                type="system",
                sender="system",
                content=(
                    f"所有坐席当前正忙，您排在第 {queue_pos} 位。"
                    f"预计等待 {queue_pos * 2} 分钟，请稍候..."
                ),
                transfer_id=transfer.transfer_id,
                metadata={"queue_position": queue_pos},
            ))

        logger.info(
            f"[HumanTransferService] 转接已发起: {transfer.transfer_id[:10]} "
            f"(trigger={trigger.value}, agent={agent_id or 'queued'})"
        )

        return transfer

    def send_user_message_to_agent(
        self,
        session_id: str,
        transfer_id: str,
        content: str,
    ):
        """将用户消息转发给坐席"""
        transfer = self._transfer_history.get(transfer_id)
        if not transfer or not transfer.assigned_agent:
            return

        self.ws.send_to_agent(transfer.assigned_agent, WSMessage(
            type="message",
            sender="user",
            content=content,
            transfer_id=transfer_id,
            metadata={"session_id": session_id},
        ))

    def send_agent_message_to_user(
        self,
        agent_id: str,
        transfer_id: str,
        content: str,
    ):
        """将坐席消息转发给用户"""
        transfer = self._transfer_history.get(transfer_id)
        if not transfer:
            return

        self.ws.send_to_user(transfer.session_id, WSMessage(
            type="message",
            sender="agent",
            content=content,
            transfer_id=transfer_id,
            metadata={"agent_id": agent_id, "agent_name": transfer.agent_name},
        ))

    def end_transfer(
        self,
        transfer_id: str,
        reason: str = "completed",
    ):
        """结束转接"""
        transfer = self._transfer_history.get(transfer_id)
        if not transfer:
            return

        # 通知双方
        self.ws.send_to_user(transfer.session_id, WSMessage(
            type="system",
            sender="system",
            content="人工服务已结束。如需继续帮助，请随时联系我。",
            transfer_id=transfer_id,
            metadata={"reason": reason},
        ))

        if transfer.assigned_agent:
            self.ws.send_to_agent(transfer.assigned_agent, WSMessage(
                type="system",
                sender="system",
                content=f"会话 {transfer.session_id[:12]} 转接已结束",
                transfer_id=transfer_id,
                metadata={"reason": reason},
            ))

        # 清理
        self.agent_manager.complete_transfer(transfer_id)
        self.detector.reset_counts(transfer.session_id)
        logger.info(f"[HumanTransferService] 转接结束: {transfer_id[:10]} ({reason})")


# ============================================================================
# 全局单例
# ============================================================================
_transfer_service: Optional[HumanTransferService] = None
_transfer_lock = threading.Lock()


def get_human_transfer_service() -> HumanTransferService:
    global _transfer_service
    if _transfer_service is None:
        with _transfer_lock:
            if _transfer_service is None:
                _transfer_service = HumanTransferService()
    return _transfer_service


# ============================================================================
# 独立测试
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepService Human Transfer — 独立测试")
    logger.info("=" * 60)

    service = HumanTransferService()

    # 测试1：触发检测
    logger.info("\n[测试1] 转接触发检测")
    test_cases = [
        ("帮我转人工", "order_status", 0.9),
        ("你们太垃圾了", "complaint", 0.9),
        ("这个产品怎么用？", "product_inquiry", 0.3),
    ]
    for msg, intent, conf in test_cases:
        trigger = service.check_trigger("test_session", msg, intent, conf)
        logger.info(f"  '{msg}' → {trigger.value if trigger else '不触发'}")

    # 测试2：发起转接
    logger.info("\n[测试2] 发起转接")
    transfer = service.initiate_transfer(
        session_id="test_session",
        trigger=TransferTrigger.USER_REQUESTED,
        user_id="user_001",
        conversation_summary="用户咨询退货流程，已告知7天退货政策",
        recent_messages=[
            {"role": "user", "content": "我要退货"},
            {"role": "assistant", "content": "请问您的订单号是？"},
            {"role": "user", "content": "#20240001"},
            {"role": "assistant", "content": "已查到这个订单，请问退货原因？"},
            {"role": "user", "content": "商品有质量问题，帮我转人工"},
        ],
        tracked_entities={"order_id": "#20240001"},
    )
    logger.info(f"  转接ID: {transfer.transfer_id}")
    logger.info(f"  状态: {transfer.status.value}")

    # 测试3：上下文打包
    logger.info("\n[测试3] 转接上下文（坐席视角）")
    display = transfer.to_agent_display()
    logger.info(f"\n{display}")

    logger.info("=" * 60)
    logger.info("人工转接测试完成 ✓")
