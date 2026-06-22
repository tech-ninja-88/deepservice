"""
Session lifecycle manager: create, track, expire, and close sessions.
Redis is the primary store with automatic fallback to in-memory dict.
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


# ___ Session status enum
class SessionStatus(str, Enum):
    """Session lifecycle states"""
    ACTIVE = "active"               # user is actively chatting
    IDLE = "idle"                   # no activity > 5 min, not yet closed
    WAITING_TRANSFER = "waiting"    # waiting to be transferred to human
    IN_HUMAN_SERVICE = "human"      # being served by human agent
    CLOSED = "closed"               # closed
    EXPIRED = "expired"             # auto-expired by system
    ERROR = "error"                 # abnormal state


class MessageRole(str, Enum):
    """Sender roles in a conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    HUMAN_AGENT = "human_agent"


# ___ Data structures
@dataclass
class Message:
    """A single message in a conversation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    # metadata may include: intent, confidence, tokens, sources, feedback

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
    """A conversation session"""
    id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    user_id: str = "anonymous"
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    ttl: int = 1800                           # default 30 min expiry

    # meta info
    channel: str = "web"                      # web / mobile / api / wechat
    locale: str = "zh-CN"                     # language
    user_agent: str = ""                      # device info
    ip_address: str = ""                      # client IP

    # counters
    message_count: int = 0
    turn_count: int = 0                       # one user+assistant exchange = 1 turn

    # last message timestamps for idle detection
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
            "ip_address": self.ip_address,
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
            ip_address=data.get("ip_address", ""),
            message_count=data.get("message_count", 0),
            turn_count=data.get("turn_count", 0),
            last_user_message_at=data.get("last_user_message_at", 0.0),
            last_assistant_message_at=data.get("last_assistant_message_at", 0.0),
        )

    @property
    def is_expired(self) -> bool:
        """True if the session has exceeded its TTL"""
        return (time.time() - self.updated_at) > self.ttl

    @property
    def idle_seconds(self) -> float:
        """Seconds since last user message"""
        return time.time() - self.last_user_message_at


# ___ Storage backend abstraction
class SessionStoreBackend(ABC):
    """Abstract base for session storage backends"""

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
    Redis-backed session storage.
    Keys: session:{id}:info (Hash), session:{id}:messages (List), session:index (Sorted Set).
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or "redis://localhost:6379"
        self._redis = None
        self._connect()

    def _connect(self):
        """Establish Redis connection"""
        try:
            import redis
            self._redis = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            self._redis.ping()
            logger.info(f"[RedisSessionStore] Redis connected: {self.redis_url}")
        except Exception as e:
            logger.error(f"[RedisSessionStore] Redis connection failed: {e}")
            raise ConnectionError(f"Failed to connect Redis: {e}")

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
            # type conversion
            for int_field in ["message_count", "turn_count", "ttl"]:
                if int_field in data:
                    data[int_field] = int(data[int_field])
            for float_field in ["created_at", "updated_at", "closed_at",
                                "last_user_message_at", "last_assistant_message_at"]:
                if float_field in data and data[float_field] is not None and data[float_field] != "":
                    data[float_field] = float(data[float_field])
            return data
        except Exception as e:
            logger.error(f"[RedisSessionStore] get_session failed {session_id}: {e}")
            return None

    def save_session(self, session: Session) -> bool:
        try:
            session.updated_at = time.time()
            data = session.to_dict()
            # Convert all values to strings (Redis Hash requirement)
            str_data = {k: str(v) if v is not None else "" for k, v in data.items()}

            pipe = self._redis.pipeline()
            pipe.hset(self._key_info(session.id), mapping=str_data)
            pipe.expire(self._key_info(session.id), session.ttl)
            pipe.expire(self._key_messages(session.id), session.ttl)
            # Update active index
            pipe.zadd(self._key_index(), {session.id: time.time()})
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"[RedisSessionStore] save_session failed {session.id}: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        try:
            pipe = self._redis.pipeline()
            pipe.delete(self._key_info(session_id))
            pipe.delete(self._key_messages(session_id))
            pipe.zrem(self._key_index(), session_id)
            pipe.execute()
            logger.info(f"[RedisSessionStore] Session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"[RedisSessionStore] delete_session failed {session_id}: {e}")
            return False

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        try:
            raw_messages = self._redis.lrange(
                self._key_messages(session_id), -limit, -1
            )
            return [json.loads(m) for m in raw_messages]
        except Exception as e:
            logger.error(f"[RedisSessionStore] get_messages failed {session_id}: {e}")
            return []

    def append_message(self, session_id: str, message: Message) -> bool:
        try:
            pipe = self._redis.pipeline()
            pipe.rpush(self._key_messages(session_id), json.dumps(
                message.to_dict(), ensure_ascii=False
            ))
            # Cap message list at 500 entries (prevent memory overflow)
            pipe.ltrim(self._key_messages(session_id), -500, -1)
            pipe.expire(self._key_messages(session_id), 3600)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"[RedisSessionStore] append_message failed {session_id}: {e}")
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
            # Get most recently updated sessions
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
    In-memory session store (dev / Redis fallback).
    Thread-safe dict storage; data lost on restart.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        self._messages: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()
        logger.warning("[MemorySessionStore] Using in-memory storage — data will be lost on restart!")

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._lock:
            data = self._sessions.get(session_id)
            if data:
                # Check expiry
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
            # Keep last 500 messages
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


# ___ Session manager
class SessionManager:
    """
    Session manager — the single public interface.
    Auto-selects backend (Redis first, degrades to memory).
    Manages TTL, idle cleanup, thread-safe.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.config = get_config()
        self._store = self._init_store(redis_url)
        self._cleanup_thread = None

        # counters
        self._sessions_created = 0
        self._sessions_closed = 0

        logger.info(f"[SessionManager] Initialized, backend: {self._store.get_stats().get('backend', 'unknown')}")

    def _init_store(self, redis_url: Optional[str] = None) -> SessionStoreBackend:
        """Initialize storage backend (Redis first, fallback to memory)"""
        # If Redis URL is explicitly provided, use it directly
        if redis_url:
            try:
                return RedisSessionStore(redis_url)
            except Exception:
                logger.warning("Redis unavailable, falling back to memory storage")

        # Try Redis URL from environment variables
        import os
        env_redis = os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")
        if env_redis:
            try:
                return RedisSessionStore(env_redis)
            except Exception:
                logger.warning("Redis unavailable, falling back to memory storage")

        return MemorySessionStore()

    # ___ Session CRUD

    def create_session(
        self,
        user_id: str = "anonymous",
        channel: str = "web",
        locale: str = "zh-CN",
        user_agent: str = "",
        ip_address: str = "",
        ttl: int = 1800,
    ) -> Session:
        """Create a new session. Returns the Session object."""
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
                f"[SessionManager] Session created: {session.id} "
                f"(user={user_id}, channel={channel}, ttl={ttl}s)"
            )
        else:
            logger.error(f"[SessionManager] Failed to create session: {session.id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session. If expired, auto-marks as EXPIRED and returns None.
        """
        data = self._store.get_session(session_id)
        if not data:
            return None
        session = Session.from_dict(data)
        if session.is_expired:
            logger.info(f"[SessionManager] Session expired: {session_id}")
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
        Get or create a session. Returns (Session, is_new).
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session, False

        session = self.create_session(user_id=user_id, **kwargs)
        return session, True

    def update_session(self, session: Session) -> bool:
        """Save updated session info"""
        return self._store.save_session(session)

    def update_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status"""
        session = self.get_session(session_id)
        if not session:
            return False
        session.status = status
        if status == SessionStatus.CLOSED:
            session.closed_at = time.time()
        return self._store.save_session(session)

    def close_session(self, session_id: str, reason: str = "user_requested") -> bool:
        """
        Close a session. Reason options: user_requested, expired, transferred, error.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        # If in human service, don't close directly
        if session.status == SessionStatus.IN_HUMAN_SERVICE and reason != "transferred":
            logger.info(f"[SessionManager] Session {session_id} in human service, marking as closed")
            session.status = SessionStatus.CLOSED
            session.closed_at = time.time()
        else:
            session.status = SessionStatus.CLOSED if reason != "expired" else SessionStatus.EXPIRED
            session.closed_at = time.time()

        success = self._store.save_session(session)
        if success:
            self._sessions_closed += 1
            logger.info(f"[SessionManager] Session closed: {session_id} (reason: {reason})")
        return success

    def delete_session(self, session_id: str) -> bool:
        """Physically delete a session"""
        return self._store.delete_session(session_id)

    def refresh_session(self, session_id: str) -> bool:
        """Refresh session TTL"""
        return self._store.refresh_ttl(session_id)

    # ___ Message management

    def append_message(self, session_id: str, message: Message) -> bool:
        """Append a message to the session"""
        success = self._store.append_message(session_id, message)

        if success:
            # Update session counters
            session = self.get_session(session_id)
            if session:
                session.message_count += 1
                if message.role == MessageRole.USER:
                    session.last_user_message_at = time.time()
                    session.turn_count += 1
                elif message.role == MessageRole.ASSISTANT:
                    session.last_assistant_message_at = time.time()
                self._store.save_session(session)

            # Refresh TTL
            self._store.refresh_ttl(session_id)

        return success

    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Message]:
        """
        Retrieve session messages. offset=0 means most recent.
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
        """Get last N messages (convenience method)"""
        return self.get_messages(session_id, limit=n, offset=0)

    def get_last_n_turns(
        self,
        session_id: str,
        n: int = 5,
    ) -> List[Tuple[Message, Optional[Message]]]:
        """
        Get last N turns as paired (user_msg, assistant_msg) tuples.
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

    # ___ Management & stats

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> List[Session]:
        """List sessions with optional filters"""
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
        """Clean up expired sessions"""
        active_ids = self._store.list_active_sessions(500)
        cleaned = 0
        for sid in active_ids:
            session = self.get_session(sid)
            if session is None or session.is_expired:
                self.close_session(sid, reason="expired")
                cleaned += 1
        if cleaned > 0:
            logger.info(f"[SessionManager] Cleaned up {cleaned} expired sessions")
        return cleaned

    def get_stats(self) -> Dict:
        """Get aggregate statistics"""
        store_stats = self._store.get_stats()
        return {
            **store_stats,
            "sessions_created": self._sessions_created,
            "sessions_closed": self._sessions_closed,
            "session_ttl_default": self.config.app.session_idle_timeout,
        }


# Global session manager — double-checked locking for thread safety at startup

_session_manager: Optional[SessionManager] = None
_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        with _lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager


# ___ Self-check
if __name__ == "__main__":
    logger.info("Session Manager self-check")
    sm = SessionManager()

    # Test 1: Create session
    session, is_new = sm.get_or_create_session(user_id="test_user_001")
    logger.info(f"  Created: {session.id} (new={is_new})")

    # Test 2: Append messages
    sm.append_message(session.id, Message(role=MessageRole.USER, content="我的订单12345发货了吗？"))
    sm.append_message(session.id, Message(role=MessageRole.ASSISTANT, content="让我帮您查询订单#12345的状态..."))
    logger.info(f"  Messages appended, count={session.message_count}")

    # Test 3: Get recent messages
    messages = sm.get_recent_messages(session.id, n=5)
    logger.info(f"  Recent messages: {len(messages)}")

    # Test 4: Get turns
    turns = sm.get_last_n_turns(session.id, n=3)
    logger.info(f"  Turns: {len(turns)}")

    # Test 5: Stats
    stats = sm.get_stats()
    logger.info(f"  Stats: backend={stats.get('backend')}, sessions={stats.get('total_active_sessions')}")

    # Test 6: Close session
    sm.close_session(session.id, reason="test_done")
    logger.info("Session manager self-check complete.")
