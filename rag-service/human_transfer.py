"""
Human transfer service: multi-trigger detection, context packaging, agent assignment, WebSocket relay.
Six trigger conditions cover user requests, keywords, low confidence, negative sentiment, forced intents,
and flow failures. Context is packed so agents don't re-ask basic info.
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


# ... Enums & constants
class TransferTrigger(str, Enum):
    """Transfer trigger reasons"""
    USER_REQUESTED = "user_requested"             # user explicitly asked
    KEYWORD_DETECTED = "keyword_detected"         # sensitive keyword hit
    LOW_CONFIDENCE = "low_confidence"             # confidence too low
    NEGATIVE_SENTIMENT = "negative_sentiment"     # negative sentiment accumulated
    INTENT_FORCED = "intent_forced"               # intent forced transfer
    FLOW_FAILED = "flow_failed"                   # structured flow failed
    ADMIN_TRANSFER = "admin_transfer"             # admin initiated


class TransferStatus(str, Enum):
    """Transfer lifecycle statuses"""
    PENDING = "pending"               # waiting for agent to accept
    QUEUED = "queued"                 # in queue
    CONNECTED = "connected"           # agent connected
    AGENT_DISCONNECTED = "disconnected"  # agent disconnected
    COMPLETED = "completed"           # completed
    TIMEOUT = "timeout"               # timed out
    REJECTED = "rejected"             # rejected
    ERROR = "error"                   # error state


# ... Data structures
@dataclass
class TransferContext:
    """
    Packed context handed to the human agent — they should not need to re-ask anything.
    """
    # Basic info
    transfer_id: str = field(default_factory=lambda: f"transfer_{uuid.uuid4().hex[:10]}")
    session_id: str = ""
    user_id: str = "anonymous"
    trigger: TransferTrigger = TransferTrigger.USER_REQUESTED

    # Conversation summary (preview for the agent)
    conversation_summary: str = ""
    recent_messages: List[Dict] = field(default_factory=list)  # last 5 turns
    current_intent: str = ""
    tracked_entities: Dict[str, str] = field(default_factory=dict)

    # User info
    user_profile_text: str = ""
    user_sentiment_avg: float = 0.5
    complaint_history: int = 0

    # Processing info
    priority: int = 0                   # 0=urgent, 1=normal, 2=low
    created_at: float = field(default_factory=time.time)
    status: TransferStatus = TransferStatus.PENDING
    assigned_agent: str = ""
    agent_name: str = ""

    # Metadata
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
        """Format context as a human-readable agent display"""
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
    """Human agent profile"""
    agent_id: str
    name: str = ""
    skills: List[str] = field(default_factory=list)  # pre-sales, after-sales, technical
    status: str = "offline"                           # online / busy / offline
    current_sessions: int = 0
    max_sessions: int = 5


# ... WebSocket message (framework-agnostic)
@dataclass
class WSMessage:
    """Framework-agnostic WebSocket message"""
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


# ... Trigger condition detector
class TransferTriggerDetector:
    """
    Multi-dimensional trigger detector — any one hit initiates a transfer.
    """

    # User-requested keywords
    USER_REQUEST_KEYWORDS = [
        "人工", "转人工", "客服人员", "人工客服",
        "真人", "转接", "人工坐席", "找人工",
        "有人吗", "不是机器人",
    ]

    # Sensitive/urgent keywords (immediate transfer)
    URGENT_KEYWORDS = [
        "生命危险", "人身安全", "报警", "诈骗",
        "法院", "律师函", "消协", "315",
        "媒体曝光", "曝光你们", "记者",
    ]

    # Negative sentiment words
    NEGATIVE_SENTIMENT_WORDS = [
        "气死", "垃圾", "骗子", "太差", "坑人",
        "失望", "投诉", "举报", "差评", "退款",
        "再也不用", "再也不买",
    ]

    def __init__(self):
        # Per-session counters
        self._negative_count: Dict[str, int] = {}           # session_id -> negative count
        self._low_confidence_count: Dict[str, int] = {}     # session_id -> low-confidence count
        self._clarify_count: Dict[str, int] = {}            # session_id -> clarify count

    def detect(
        self,
        session_id: str,
        user_message: str,
        intent: str = "",
        confidence: float = 1.0,
        sentiment: float = 0.5,
    ) -> Optional[TransferTrigger]:
        """
        Check all triggers; returns the matching TransferTrigger or None.
        """
        msg_lower = user_message.lower()

        # Condition 1: user explicitly requests human
        if any(kw in msg_lower for kw in self.USER_REQUEST_KEYWORDS):
            logger.info(f"[TransferTrigger] User requested transfer: {session_id[:12]}")
            return TransferTrigger.USER_REQUESTED

        # Condition 2: sensitive/urgent keywords
        if any(kw in msg_lower for kw in self.URGENT_KEYWORDS):
            logger.warning(f"[TransferTrigger] Sensitive keyword triggered: {session_id[:12]}")
            return TransferTrigger.KEYWORD_DETECTED

        # Condition 3: intent forces transfer
        if intent in ("complaint", "transfer_human"):
            logger.info(f"[TransferTrigger] Intent-forced transfer: intent={intent}")
            return TransferTrigger.INTENT_FORCED

        # Condition 4: consecutive low confidence
        if confidence < 0.4:
            self._low_confidence_count[session_id] = (
                self._low_confidence_count.get(session_id, 0) + 1
            )
            if self._low_confidence_count.get(session_id, 0) >= 3:
                logger.warning(
                    f"[TransferTrigger] Low confidence triggered: "
                    f"count={self._low_confidence_count[session_id]}"
                )
                return TransferTrigger.LOW_CONFIDENCE
        else:
            self._low_confidence_count[session_id] = 0  # reset

        # Condition 5: consecutive negative sentiment
        if sentiment < 0.3:
            self._negative_count[session_id] = (
                self._negative_count.get(session_id, 0) + 1
            )
            if self._negative_count.get(session_id, 0) >= 3:
                logger.warning(
                    f"[TransferTrigger] Negative sentiment triggered: "
                    f"count={self._negative_count[session_id]}"
                )
                return TransferTrigger.NEGATIVE_SENTIMENT
        else:
            self._negative_count[session_id] = max(
                0, self._negative_count.get(session_id, 0) - 1
            )  # gradually decrease

        return None

    def reset_counts(self, session_id: str):
        """Reset per-session counters after transfer"""
        self._negative_count.pop(session_id, None)
        self._low_confidence_count.pop(session_id, None)
        self._clarify_count.pop(session_id, None)


# ... Agent manager
class AgentManager:
    """
    Manages online agents, task assignment, and queuing.
    """

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._transfer_queue: List[TransferContext] = []
        self._active_transfers: Dict[str, TransferContext] = {}  # transfer_id -> context
        self._lock = threading.Lock()
        logger.info("[AgentManager] Agent manager initialized")

    def register_agent(self, agent: AgentInfo):
        """Register an agent"""
        with self._lock:
            self._agents[agent.agent_id] = agent
            logger.info(f"[AgentManager] 坐席注册: {agent.agent_id} ({agent.name})")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            self._agents.pop(agent_id, None)
            logger.info(f"[AgentManager] Agent unregistered: {agent_id}")

    def assign_transfer(self, transfer: TransferContext) -> Optional[str]:
        """
        Assign transfer to an agent by skill match + load balancing.
        Returns agent_id or None if queued.
        """
        with self._lock:
            # Find available matching agents
            available = [
                a for a in self._agents.values()
                if a.status == "online"
                and a.current_sessions < a.max_sessions
            ]

            if not available:
                # Queue
                transfer.status = TransferStatus.QUEUED
                self._transfer_queue.append(transfer)
                logger.info(
                    f"[AgentManager] No available agents, queued "
                    f"(queue length: {len(self._transfer_queue)})"
                )
                return None

            # Sort by skill match
            available.sort(key=lambda a: (
                self._skill_match_score(a, transfer),
                -a.current_sessions,  # fewer current sessions = higher priority
            ))

            # Assign to best match
            agent = available[0]
            agent.current_sessions += 1
            transfer.status = TransferStatus.PENDING
            transfer.assigned_agent = agent.agent_id
            transfer.agent_name = agent.name

            self._active_transfers[transfer.transfer_id] = transfer
            logger.info(
                f"[AgentManager] Transfer {transfer.transfer_id[:10]} "
                f"assigned to agent {agent.agent_id}"
            )
            return agent.agent_id

    def complete_transfer(self, transfer_id: str):
        """Complete a transfer"""
        with self._lock:
            transfer = self._active_transfers.pop(transfer_id, None)
            if transfer:
                transfer.status = TransferStatus.COMPLETED
                # Release agent
                if transfer.assigned_agent in self._agents:
                    self._agents[transfer.assigned_agent].current_sessions -= 1

    def reject_transfer(self, transfer_id: str, reason: str = ""):
        """Agent rejected transfer — reassign"""
        with self._lock:
            transfer = self._active_transfers.pop(transfer_id, None)
            if transfer:
                transfer.status = TransferStatus.REJECTED
                transfer.notes = f"Rejected: {reason}"
                # Re-queue at front
                transfer.assigned_agent = ""
                self._transfer_queue.insert(0, transfer)

    def get_queue_length(self) -> int:
        return len(self._transfer_queue)

    def get_online_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == "online")

    def _skill_match_score(self, agent: AgentInfo, transfer: TransferContext) -> int:
        """Score skill match between agent and transfer (negative so .sort() places best first)"""
        score = 0
        tags_lower = [t.lower() for t in transfer.tags]
        for skill in agent.skills:
            if skill.lower() in tags_lower:
                score += 1
        return -score


# ... WebSocket message channel (framework-agnostic)
class WebSocketChannel:
    """
    Framework-agnostic WebSocket message channel.
    Routes: system<->user, user<->agent, system<->agent (queue, handoff, status).
    """

    def __init__(self):
        # Connection registries
        self._user_connections: Dict[str, Any] = {}      # session_id -> ws connection
        self._agent_connections: Dict[str, Any] = {}     # agent_id -> ws connection
        self._connection_lock = threading.Lock()

        # Message callbacks
        self._on_user_message: Optional[Callable] = None
        self._on_agent_message: Optional[Callable] = None

        logger.info("[WebSocketChannel] Message channel initialized")

    def register_user(self, session_id: str, connection: Any):
        """Register user WebSocket connection"""
        with self._connection_lock:
            self._user_connections[session_id] = connection
            logger.debug(f"[WebSocketChannel] 用户上线: {session_id[:12]}")

    def register_agent(self, agent_id: str, connection: Any):
        """Register agent WebSocket connection"""
        with self._connection_lock:
            self._agent_connections[agent_id] = connection
            logger.debug(f"[WebSocketChannel] Agent online: {agent_id}")

    def unregister_user(self, session_id: str):
        """Unregister user connection"""
        with self._connection_lock:
            self._user_connections.pop(session_id, None)

    def unregister_agent(self, agent_id: str):
        """Unregister agent connection"""
        with self._connection_lock:
            self._agent_connections.pop(agent_id, None)

    def send_to_user(self, session_id: str, message: WSMessage) -> bool:
        """Send message to user"""
        conn = self._user_connections.get(session_id)
        if conn:
            try:
                self._ws_send(conn, message.to_json())
                return True
            except Exception as e:
                logger.error(f"[WebSocketChannel] Send to user failed: {e}")
                return False

        # Offline message (production: store in Redis, push on reconnect)
        logger.debug(f"[WebSocketChannel] User {session_id[:12]} offline, message held")
        return False

    def send_to_agent(self, agent_id: str, message: WSMessage) -> bool:
        """Send message to agent"""
        conn = self._agent_connections.get(agent_id)
        if conn:
            try:
                self._ws_send(conn, message.to_json())
                return True
            except Exception as e:
                logger.error(f"[WebSocketChannel] Send to agent failed: {e}")
                return False
        return False

    def broadcast_to_agents(self, message: WSMessage):
        """Broadcast to all online agents"""
        for agent_id, conn in list(self._agent_connections.items()):
            try:
                self._ws_send(conn, message.to_json())
            except Exception:
                pass

    def set_on_user_message(self, callback: Callable):
        """Set callback for user messages"""
        self._on_user_message = callback

    def set_on_agent_message(self, callback: Callable):
        """Set callback for agent messages"""
        self._on_agent_message = callback

    def handle_user_message(self, session_id: str, raw_message: str):
        """Handle incoming user message"""
        try:
            msg = WSMessage.from_json(raw_message)
            if self._on_user_message:
                self._on_user_message(session_id, msg)
        except Exception as e:
            logger.error(f"[WebSocketChannel] Handle user message failed: {e}")

    def handle_agent_message(self, agent_id: str, raw_message: str):
        """Handle incoming agent message"""
        try:
            msg = WSMessage.from_json(raw_message)
            if self._on_agent_message:
                self._on_agent_message(agent_id, msg)
        except Exception as e:
            logger.error(f"[WebSocketChannel] Handle agent message failed: {e}")

    def _ws_send(self, connection: Any, message: str):
        """
        Send WebSocket message. Override to adapt to different frameworks.
        """
        # FastAPI WebSocket
        if hasattr(connection, "send_text"):
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(connection.send_text(message))
            except RuntimeError:
                pass  # sync context — WS send unavailable
        # Generic fallback
        elif hasattr(connection, "send"):
            connection.send(message)


# ... Human transfer service (public facade)
class HumanTransferService:
    """
    Public API integrating trigger detection, agent assignment, session relay, and WebSocket.
    """

    def __init__(self):
        self.detector = TransferTriggerDetector()
        self.agent_manager = AgentManager()
        self.ws = WebSocketChannel()
        self.config = get_config()

        # Transfer history
        self._transfer_history: Dict[str, TransferContext] = {}

        logger.info("[HumanTransferService] Initialized")

    def check_trigger(
        self,
        session_id: str,
        user_message: str,
        intent: str = "",
        confidence: float = 1.0,
        sentiment: float = 0.5,
    ) -> Optional[TransferTrigger]:
        """Check if transfer should be triggered"""
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
        Initiate transfer: pack context, assign agent, notify via WebSocket.
        """
        # Build transfer context
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

        # Assign agent
        agent_id = self.agent_manager.assign_transfer(transfer)

        # Save record
        self._transfer_history[transfer.transfer_id] = transfer

        # Notify user
        if agent_id:
            # Agent assigned — notify user
            self.ws.send_to_user(session_id, WSMessage(
                type="system",
                sender="system",
                content=f"已为您接通人工客服，坐席 {transfer.agent_name} 将为您服务。",
                transfer_id=transfer.transfer_id,
                metadata={"agent_id": agent_id, "agent_name": transfer.agent_name},
            ))

            # Notify agent
            self.ws.send_to_agent(agent_id, WSMessage(
                type="system",
                sender="system",
                content="New customer transfer request",
                transfer_id=transfer.transfer_id,
                metadata={
                    "transfer_context": transfer.to_dict(),
                    "display": transfer.to_agent_display(),
                },
            ))
        else:
            # Queued — notify user to wait
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
            f"[HumanTransferService] Transfer initiated: {transfer.transfer_id[:10]} "
            f"(trigger={trigger.value}, agent={agent_id or 'queued'})"
        )

        return transfer

    def send_user_message_to_agent(
        self,
        session_id: str,
        transfer_id: str,
        content: str,
    ):
        """Relay user message to agent"""
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
        """Relay agent message to user"""
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
        """End a transfer session"""
        transfer = self._transfer_history.get(transfer_id)
        if not transfer:
            return

        # Notify both parties
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
                content=f"Session {transfer.session_id[:12]} transfer ended",
                transfer_id=transfer_id,
                metadata={"reason": reason},
            ))

        # Cleanup
        self.agent_manager.complete_transfer(transfer_id)
        self.detector.reset_counts(transfer.session_id)
        logger.info(f"[HumanTransferService] Transfer ended: {transfer_id[:10]} ({reason})")


# Thread-safe singleton — WebSocket callbacks can race during handoff
import threading

_transfer_service: Optional[HumanTransferService] = None
_transfer_lock = threading.Lock()


def get_human_transfer_service() -> HumanTransferService:
    global _transfer_service
    if _transfer_service is None:
        with _transfer_lock:
            if _transfer_service is None:
                _transfer_service = HumanTransferService()
    return _transfer_service


# ... Self-check
if __name__ == "__main__":
    logger.info("Human Transfer self-check")
    service = HumanTransferService()

    # 1. Trigger detection
    logger.info("[1] Trigger detection")
    test_cases = [
        ("帮我转人工", "order_status", 0.9),
        ("你们太垃圾了", "complaint", 0.9),
        ("这个产品怎么用？", "product_inquiry", 0.3),
    ]
    for msg, intent, conf in test_cases:
        trigger = service.check_trigger("test_session", msg, intent, conf)
        logger.info(f"  '{msg}' -> {trigger.value if trigger else 'no trigger'}")

    # 测试2：发起转接
    logger.info("[2] Initiate transfer")
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
    logger.info(f"  Transfer ID: {transfer.transfer_id}, status: {transfer.status.value}")

    # 3. Context display
    logger.info("[3] Agent display")
    logger.info(f"\n{transfer.to_agent_display()}")

    logger.info("Human transfer self-check complete.")
