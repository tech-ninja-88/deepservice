"""
=============================================================================
DeepService 对话管理 — 会话管理模块 (Session Manager)
=============================================================================
职责：
  1. 会话生命周期管理（创建 → 活跃 → 转人工 → 关闭）
  2. Redis 存储会话状态（生产） / 内存存储（开发）
  3. 会话 TTL 过期策略（自动清理僵尸会话）
  4. 会话元数据追踪（用户标识、渠道、设备信息）

企业级设计原则：
  - 会话是对话管理的原子单元，所有消息和状态依附于会话
  - Redis 作为主要存储保证服务无状态可水平扩展
  - 会话超时机制防止资源泄露
  - 优雅的降级：Redis 不可用时自动切换到内存存储

存储结构设计（Redis）：
  session:{id}:info       → Hash   — 会话元信息（状态、创建时间、用户ID等）
  session:{id}:messages   → List   — 消息历史（JSON序列化）
  session:{id}:state      → Hash   — 对话状态机当前状态
  session:{id}:slots      → Hash   — 槽位填充数据
  session:{id}:ttl        → 1800s  — 整体会话 TTL
=============================================================================
"""

import json
import time
import uuid
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger

from config import get_config


# ============================================================================
# 枚举定义
# ============================================================================
class SessionStatus(str, Enum):
    """会话状态枚举"""
    ACTIVE = "active"               # 活跃中（用户正在对话）
    IDLE = "idle"                   # 空闲（超过5分钟无活动但未关闭）
    WAITING_TRANSFER = "waiting"    # 等待转接人工
    IN_HUMAN_SERVICE = "human"      # 人工坐席服务中
    CLOSED = "closed"               # 已关闭
    EXPIRED = "expired"             # 已过期（系统自动关闭）
    ERROR = "error"                 # 异常状态


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    HUMAN_AGENT = "human_agent"


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class Message:
    """单条消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    # 元数据可包含：intent, confidence, tokens, sources, feedback 等

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """会话对象"""
    id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    user_id: str = "anonymous"
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    ttl: int = 1800                           # 默认 30 分钟过期

    # 元信息
    channel: str = "web"                      # 渠道：web / mobile / api / wechat
    locale: str = "zh-CN"                     # 语言
    user_agent: str = ""                      # 设备信息
    ip_address: str = ""                      # 客户端 IP

    # 统计
    message_count: int = 0                    # 消息总数
    turn_count: int = 0                       # 对话轮数（一问一答算一轮）

    # 最后一条消息时间（用于空闲检测）
    last_user_message_at: float = 0.0
    last_assistant_message_at: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "ttl": self.ttl,
            "channel": self.channel,
            "locale": self.locale,
            "user_agent": self.user_agent,
            "message_count": self.message_count,
            "turn_count": self.turn_count,
            "last_user_message_at": self.last_user_message_at,
            "last_assistant_message_at": self.last_assistant_message_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", "anonymous"),
            status=SessionStatus(data.get("status", "active")),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            closed_at=data.get("closed_at"),
            ttl=data.get("ttl", 1800),
            channel=data.get("channel", "web"),
            locale=data.get("locale", "zh-CN"),
            user_agent=data.get("user_agent", ""),
            message_count=data.get("message_count", 0),
            turn_count=data.get("turn_count", 0),
            last_user_message_at=data.get("last_user_message_at", 0.0),
            last_assistant_message_at=data.get("last_assistant_message_at", 0.0),
        )

    @property
    def is_expired(self) -> bool:
        """判断会话是否过期"""
        return (time.time() - self.updated_at) > self.ttl

    @property
    def idle_seconds(self) -> float:
        """空闲时长（秒）"""
        return time.time() - self.last_user_message_at


# ============================================================================
# 存储后端抽象
# ============================================================================
class SessionStoreBackend(ABC):
    """会话存储后端抽象基类"""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Dict]:
        ...

    @abstractmethod
    def save_session(self, session: Session) -> bool:
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        ...

    @abstractmethod
    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        ...

    @abstractmethod
    def append_message(self, session_id: str, message: Message) -> bool:
        ...

    @abstractmethod
    def refresh_ttl(self, session_id: str) -> bool:
        ...

    @abstractmethod
    def list_active_sessions(self, limit: int = 100) -> List[str]:
        ...

    @abstractmethod
    def get_stats(self) -> Dict:
        ...


class RedisSessionStore(SessionStoreBackend):
    """
    Redis 会话存储

    存储结构：
      session:{id}:info     → Hash   — 会话元信息
      session:{id}:messages → List   — 消息列表（JSON）
      session:index         → Sorted Set — 活跃会话索引（按更新时间排序）
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or "redis://localhost:6379"
        self._redis = None
        self._connect()

    def _connect(self):
        """连接 Redis"""
        try:
            import redis
            self._redis = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self._redis.ping()
            logger.info(f"[RedisSessionStore] Redis 连接成功: {self.redis_url}")
        except Exception as e:
            logger.error(f"[RedisSessionStore] Redis 连接失败: {e}")
            raise ConnectionError(f"无法连接 Redis: {e}")

    def _key_info(self, session_id: str) -> str:
        return f"session:{session_id}:info"

    def _key_messages(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    def _key_index(self) -> str:
        return "session:index"

    def get_session(self, session_id: str) -> Optional[Dict]:
        try:
            data = self._redis.hgetall(self._key_info(session_id))
            if not data:
                return None
            # 类型转换
            for int_field in ["message_count", "turn_count", "ttl"]:
                if int_field in data:
                    data[int_field] = int(data[int_field])
            for float_field in ["created_at", "updated_at", "closed_at",
                                "last_user_message_at", "last_assistant_message_at"]:
                if float_field in data and data[float_field]:
                    data[float_field] = float(data[float_field])
            return data
        except Exception as e:
            logger.error(f"[RedisSessionStore] 获取会话失败 {session_id}: {e}")
            return None

    def save_session(self, session: Session) -> bool:
        try:
            session.updated_at = time.time()
            data = session.to_dict()
            # 转换所有值为字符串（Redis Hash 要求）
            str_data = {k: str(v) if v is not None else "" for k, v in data.items()}

            pipe = self._redis.pipeline()
            pipe.hset(self._key_info(session.id), mapping=str_data)
            pipe.expire(self._key_info(session.id), session.ttl)
            pipe.expire(self._key_messages(session.id), session.ttl)
            # 更新活跃索引
            pipe.zadd(self._key_index(), {session.id: time.time()})
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"[RedisSessionStore] 保存会话失败 {session.id}: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        try:
            pipe = self._redis.pipeline()
            pipe.delete(self._key_info(session_id))
            pipe.delete(self._key_messages(session_id))
            pipe.zrem(self._key_index(), session_id)
            pipe.execute()
            logger.info(f"[RedisSessionStore] 会话已删除: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[RedisSessionStore] 删除会话失败 {session_id}: {e}")
            return False

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        try:
            raw_messages = self._redis.lrange(
                self._key_messages(session_id), -limit, -1
            )
            return [json.loads(m) for m in raw_messages]
        except Exception as e:
            logger.error(f"[RedisSessionStore] 获取消息失败 {session_id}: {e}")
            return []

    def append_message(self, session_id: str, message: Message) -> bool:
        try:
            pipe = self._redis.pipeline()
            pipe.rpush(self._key_messages(session_id), json.dumps(
                message.to_dict(), ensure_ascii=False
            ))
            # 保持消息列表不超过 500 条（防止内存溢出）
            pipe.ltrim(self._key_messages(session_id), -500, -1)
            pipe.expire(self._key_messages(session_id), 3600)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"[RedisSessionStore] 追加消息失败 {session_id}: {e}")
            return False

    def refresh_ttl(self, session_id: str) -> bool:
        try:
            pipe = self._redis.pipeline()
            ttl = 1800
            pipe.expire(self._key_info(session_id), ttl)
            pipe.expire(self._key_messages(session_id), ttl)
            pipe.execute()
            return True
        except Exception:
            return False

    def list_active_sessions(self, limit: int = 100) -> List[str]:
        try:
            # 获取最近更新的会话
            return self._redis.zrevrange(
                self._key_index(), 0, limit - 1
            )
        except Exception:
            return []

    def get_stats(self) -> Dict:
        try:
            total = self._redis.zcard(self._key_index())
            return {
                "total_active_sessions": total,
                "backend": "redis",
                "url": self.redis_url,
            }
        except Exception:
            return {"backend": "redis", "error": "stats_unavailable"}


class MemorySessionStore(SessionStoreBackend):
    """
    内存会话存储（开发环境 / Redis 降级）

    使用线程安全的字典存储，服务重启后数据丢失。
    生产环境必须使用 Redis。
    """

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._messages: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()
        logger.warning("[MemorySessionStore] 使用内存存储，服务重启后数据将丢失！")

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            data = self._sessions.get(session_id)
            if data:
                # 检查过期
                ttl = data.get("ttl", 1800)
                if time.time() - data.get("updated_at", 0) > ttl:
                    self.delete_session(session_id)
                    return None
                return dict(data)
            return None

    def save_session(self, session: Session) -> bool:
        with self._lock:
            session.updated_at = time.time()
            self._sessions[session.id] = session.to_dict()
            return True

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._messages.pop(session_id, None)
            return True

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        with self._lock:
            msgs = self._messages.get(session_id, [])
            return msgs[-limit:]

    def append_message(self, session_id: str, message: Message) -> bool:
        with self._lock:
            if session_id not in self._messages:
                self._messages[session_id] = []
            self._messages[session_id].append(message.to_dict())
            # 保持最近 500 条
            if len(self._messages[session_id]) > 500:
                self._messages[session_id] = self._messages[session_id][-500:]
            return True

    def refresh_ttl(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["updated_at"] = time.time()
            return True

    def list_active_sessions(self, limit: int = 100) -> List[str]:
        with self._lock:
            active = []
            for sid, data in self._sessions.items():
                if data.get("status") != SessionStatus.CLOSED.value:
                    active.append(sid)
            return sorted(active, key=lambda s: self._sessions[s].get("updated_at", 0), reverse=True)[:limit]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_active_sessions": len(self._sessions),
                "total_messages": sum(len(msgs) for msgs in self._messages.values()),
                "backend": "memory",
            }


# ============================================================================
# 会话管理器
# ============================================================================
class SessionManager:
    """
    会话管理器 — 对外唯一接口

    特性：
      - 自动选择存储后端（Redis 优先，失败降级到内存）
      - 会话 TTL 管理（默认 30 分钟自动过期）
      - 空闲会话自动清理
      - 线程安全
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.config = get_config()
        self._store = self._init_store(redis_url)
        self._cleanup_thread = None

        # 统计
        self._sessions_created = 0
        self._sessions_closed = 0

        logger.info(f"[SessionManager] 初始化完成，后端: {self._store.get_stats().get('backend', 'unknown')}")

    def _init_store(self, redis_url: Optional[str] = None) -> SessionStoreBackend:
        """初始化存储后端（Redis 优先，降级到内存）"""
        # 如果明确提供了 Redis URL，直接使用
        if redis_url:
            try:
                return RedisSessionStore(redis_url)
            except Exception:
                logger.warning("Redis 不可用，降级到内存存储")

        # 尝试环境变量中的 Redis URL
        import os
        env_redis = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")
        if env_redis:
            try:
                return RedisSessionStore(env_redis)
            except Exception:
                logger.warning("Redis 不可用，降级到内存存储")

        return MemorySessionStore()

    # ──── Session CRUD ────

    def create_session(
        self,
        user_id: str = "anonymous",
        channel: str = "web",
        locale: str = "zh-CN",
        user_agent: str = "",
        ip_address: str = "",
        ttl: int = 1800,
    ) -> Session:
        """
        创建新会话

        参数:
          user_id: 用户标识（可匿名）
          channel: 渠道来源
          ttl: 过期时间（秒），默认 30 分钟

        返回:
          Session 对象
        """
        session = Session(
            user_id=user_id,
            channel=channel,
            locale=locale,
            user_agent=user_agent,
            ip_address=ip_address,
            ttl=ttl,
        )
        success = self._store.save_session(session)
        if success:
            self._sessions_created += 1
            logger.info(
                f"[SessionManager] 会话创建: {session.id} "
                f"(user={user_id}, channel={channel}, ttl={ttl}s)"
            )
        else:
            logger.error(f"[SessionManager] 会话创建失败: {session.id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话

        如果会话已过期，自动标记为 EXPIRED 并返回 None。
        """
        data = self._store.get_session(session_id)
        if not data:
            return None
        session = Session.from_dict(data)
        if session.is_expired:
            logger.info(f"[SessionManager] 会话已过期: {session_id}")
            self.close_session(session_id, reason="expired")
            return None
        return session

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: str = "anonymous",
        **kwargs,
    ) -> Tuple[Session, bool]:
        """
        获取或创建会话

        返回: (Session, is_new) — is_new 表示是否新创建
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session, False

        session = self.create_session(user_id=user_id, **kwargs)
        return session, True

    def update_session(self, session: Session) -> bool:
        """更新会话信息"""
        return self._store.save_session(session)

    def update_status(self, session_id: str, status: SessionStatus) -> bool:
        """更新会话状态"""
        session = self.get_session(session_id)
        if not session:
            return False
        session.status = status
        if status == SessionStatus.CLOSED:
            session.closed_at = time.time()
        return self._store.save_session(session)

    def close_session(self, session_id: str, reason: str = "user_requested") -> bool:
        """
        关闭会话

        原因选项: user_requested, expired, transferred, error
        """
        session = self.get_session(session_id)
        if not session:
            return False

        # 如果转人工了，不直接关闭
        if session.status == SessionStatus.IN_HUMAN_SERVICE and reason != "transferred":
            logger.info(f"[SessionManager] 会话 {session_id} 在人工服务中，标记为转为已关闭")
            session.status = SessionStatus.CLOSED
            session.closed_at = time.time()
        else:
            session.status = SessionStatus.CLOSED if reason != "expired" else SessionStatus.EXPIRED
            session.closed_at = time.time()

        success = self._store.save_session(session)
        if success:
            self._sessions_closed += 1
            logger.info(f"[SessionManager] 会话关闭: {session_id} (原因: {reason})")
        return success

    def delete_session(self, session_id: str) -> bool:
        """物理删除会话"""
        return self._store.delete_session(session_id)

    def refresh_session(self, session_id: str) -> bool:
        """续期会话 TTL"""
        return self._store.refresh_ttl(session_id)

    # ──── Message 管理 ────

    def append_message(self, session_id: str, message: Message) -> bool:
        """追加一条消息到会话"""
        success = self._store.append_message(session_id, message)

        if success:
            # 更新会话统计
            session = self.get_session(session_id)
            if session:
                session.message_count += 1
                if message.role == MessageRole.USER:
                    session.last_user_message_at = time.time()
                    session.turn_count += 1
                elif message.role == MessageRole.ASSISTANT:
                    session.last_assistant_message_at = time.time()
                self._store.save_session(session)

            # 续期
            self._store.refresh_ttl(session_id)

        return success

    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Message]:
        """
        获取会话消息

        参数:
          limit: 返回条数
          offset: 偏移量（0 = 最新）
        """
        raw = self._store.get_messages(session_id, limit + offset)
        messages = [Message.from_dict(m) for m in raw]
        if offset > 0:
            messages = messages[:-offset] if len(messages) > offset else []
        return messages[-limit:] if limit > 0 else messages

    def get_recent_messages(
        self,
        session_id: str,
        n: int = 10,
    ) -> List[Message]:
        """获取最近 N 条消息（常用）"""
        return self.get_messages(session_id, limit=n, offset=0)

    def get_last_n_turns(
        self,
        session_id: str,
        n: int = 5,
    ) -> List[Tuple[Message, Optional[Message]]]:
        """
        获取最近 N 轮对话（配对用户-助手消息）

        返回: [(user_msg, assistant_msg), ...]
        """
        messages = self.get_messages(session_id, limit=n * 2 + 5)
        turns = []
        current_user = None

        for msg in messages:
            if msg.role == MessageRole.USER:
                current_user = msg
            elif msg.role == MessageRole.ASSISTANT and current_user:
                turns.append((current_user, msg))
                current_user = None

        return turns[-n:] if len(turns) > n else turns

    # ──── 管理与统计 ────

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> List[Session]:
        """列出会话（可筛选）"""
        active_ids = self._store.list_active_sessions(limit)
        sessions = []
        for sid in active_ids:
            session = self.get_session(sid)
            if session:
                if user_id and session.user_id != user_id:
                    continue
                if status and session.status != status:
                    continue
                sessions.append(session)
        return sessions

    def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        active_ids = self._store.list_active_sessions(500)
        cleaned = 0
        for sid in active_ids:
            session = self.get_session(sid)
            if session is None or session.is_expired:
                self.close_session(sid, reason="expired")
                cleaned += 1
        if cleaned > 0:
            logger.info(f"[SessionManager] 清理了 {cleaned} 个过期会话")
        return cleaned

    def get_stats(self) -> Dict:
        """获取统计信息"""
        store_stats = self._store.get_stats()
        return {
            **store_stats,
            "sessions_created": self._sessions_created,
            "sessions_closed": self._sessions_closed,
            "session_ttl_default": self.config.app.session_idle_timeout,
        }


# ============================================================================
# 全局单例
# ============================================================================
_session_manager: Optional[SessionManager] = None
_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """获取全局 SessionManager 单例"""
    global _session_manager
    if _session_manager is None:
        with _lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager


# ============================================================================
# 独立测试
# ============================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("DeepService Session Manager — 独立测试")
    logger.info("=" * 60)

    sm = SessionManager()

    # 测试1：创建会话
    session, is_new = sm.get_or_create_session(user_id="test_user_001")
    logger.info(f"[测试1] 创建会话: {session.id} (新会话={is_new})")

    # 测试2：追加消息
    sm.append_message(session.id, Message(role=MessageRole.USER, content="我的订单12345发货了吗？"))
    sm.append_message(session.id, Message(role=MessageRole.ASSISTANT, content="让我帮您查询订单#12345的状态..."))
    logger.info(f"[测试2] 追加消息后 message_count={session.message_count}")

    # 测试3：获取最近消息
    messages = sm.get_recent_messages(session.id, n=5)
    logger.info(f"[测试3] 最近消息数: {len(messages)}")
    for msg in messages:
        logger.info(f"  [{msg.role.value}] {msg.content[:50]}...")

    # 测试4：获取对话轮次
    turns = sm.get_last_n_turns(session.id, n=3)
    logger.info(f"[测试4] 对话轮数: {len(turns)}")

    # 测试5：会话统计
    stats = sm.get_stats()
    logger.info(f"[测试5] 统计: {stats}")

    # 测试6：关闭会话
    sm.close_session(session.id, reason="test_done")
    closed = sm.get_session(session.id)
    logger.info(f"[测试6] 关闭后状态: {closed.status.value if closed else 'N/A'}")

    logger.info("=" * 60)
    logger.info("会话管理测试完成 ✓")
