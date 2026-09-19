"""Conversation memory and session state on Upstash Redis.

Two keys per session:

    chat:{session_id}:messages  — a LIST of JSON message objects
    chat:{session_id}:state     — a HASH of arbitrary session state

Sessions are isolated by construction (the id is part of every key) and both
keys carry a TTL so abandoned terminal sessions expire on their own.

Message history is capped at `HISTORY_MAX_MESSAGES` in Redis, and
`get_history()` returns only the most recent `HISTORY_WINDOW` by default —
the whole transcript is deliberately never pasted into a prompt.

Every method degrades gracefully: if Upstash is unreachable the chatbot keeps
working statelessly rather than crashing (Phase 18).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.config.settings import settings
from app.logging_setup import get_logger

log = get_logger("memory")

Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(raw: str) -> "Message | None":
        try:
            data = json.loads(raw)
            return Message(
                role=data["role"],
                content=data["content"],
                timestamp=float(data.get("timestamp", 0.0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log.warning("discarded malformed history entry")
            return None


class RedisMemory:
    """Thin, failure-tolerant wrapper over the Upstash REST client."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._available: bool | None = None

    # ── connection ────────────────────────────────────────────────────
    @property
    def configured(self) -> bool:
        return settings.redis.configured

    def _connect(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self.configured:
            self._available = False
            return None
        try:
            from upstash_redis import Redis

            self._client = Redis(
                url=settings.redis.url,
                token=settings.redis.token,
            )
            return self._client
        except Exception:
            log.exception("upstash client construction failed")
            self._available = False
            return None

    def ping(self) -> tuple[bool, str]:
        """Round-trips a real write+read. Used by scripts/test_connections.py."""
        if not self.configured:
            return False, "UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN are not set"
        client = self._connect()
        if client is None:
            return False, "could not construct the Upstash client"
        probe_key = "tredev:ai-core:healthcheck"
        try:
            client.set(probe_key, str(time.time()), ex=60)
            value = client.get(probe_key)
            if value is None:
                return False, "write succeeded but read returned nothing"
            self._available = True
            return True, "read/write round-trip OK"
        except Exception as exc:  # noqa: BLE001 — surface the provider's message
            self._available = False
            return False, f"{type(exc).__name__}: {exc}"

    # ── keys ──────────────────────────────────────────────────────────
    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"chat:{session_id}:messages"

    @staticmethod
    def _state_key(session_id: str) -> str:
        return f"chat:{session_id}:state"

    @staticmethod
    def _chart_key(user_id: str) -> str:
        return f"chart:{user_id}"

    # ── chart facts cache ─────────────────────────────────────────────
    # Keyed by user, not session: a chart belongs to a person and does not
    # change between their sessions. Caching it is what lets an API client
    # send only {user_id, question} instead of re-uploading the whole chart
    # on every turn. The cache stores the compact `to_facts()` view, never a
    # full backend payload, so a stale entry can only ever be missing data —
    # never carry invented data.
    def get_chart_facts(self, user_id: str) -> dict[str, Any] | None:
        client = self._connect()
        if client is None:
            return None
        try:
            raw = client.get(self._chart_key(user_id))
        except Exception:
            log.exception("get_chart_facts failed", extra={"user_id": user_id})
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("discarded malformed chart cache entry", extra={"user_id": user_id})
            return None

    def save_chart_facts(self, user_id: str, facts: dict[str, Any], ttl_seconds: int | None = None) -> bool:
        client = self._connect()
        if client is None or not facts:
            return False
        try:
            client.set(
                self._chart_key(user_id),
                json.dumps(facts, ensure_ascii=False),
                ex=ttl_seconds or settings.redis.chart_ttl_seconds,
            )
            return True
        except Exception:
            log.exception("save_chart_facts failed", extra={"user_id": user_id})
            return False

    def clear_chart_facts(self, user_id: str) -> bool:
        client = self._connect()
        if client is None:
            return False
        try:
            client.delete(self._chart_key(user_id))
            return True
        except Exception:
            log.exception("clear_chart_facts failed", extra={"user_id": user_id})
            return False

    # ── rate limiting ─────────────────────────────────────────────────
    def check_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Fixed-window counter. Returns (allowed, remaining).

        Fails *open*: if Redis is unreachable the request is allowed through.
        A memory outage taking the chatbot down would be a worse failure than
        briefly un-enforced quotas, and nothing here guards a paid resource
        that an attacker could reach without an LLM key anyway.
        """
        client = self._connect()
        if client is None:
            return True, limit
        key = f"ratelimit:{identifier}:{int(time.time()) // window_seconds}"
        try:
            count = int(client.incr(key) or 1)
            if count == 1:
                client.expire(key, window_seconds)
            return count <= limit, max(0, limit - count)
        except Exception:
            log.exception("check_rate_limit failed", extra={"identifier": identifier})
            return True, limit

    # ── conversation history ──────────────────────────────────────────
    def save_message(self, session_id: str, role: Role, content: str) -> bool:
        client = self._connect()
        if client is None:
            return False
        key = self._messages_key(session_id)
        message = Message(role=role, content=content, timestamp=time.time())
        try:
            client.rpush(key, message.to_json())
            # Keep only the newest N entries so a long session cannot grow
            # without bound.
            client.ltrim(key, -settings.redis.max_messages, -1)
            client.expire(key, settings.redis.ttl_seconds)
            return True
        except Exception:
            log.exception("save_message failed", extra={"session_id": session_id})
            return False

    def get_history(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Most recent `limit` messages, oldest first. Empty list on failure."""
        client = self._connect()
        if client is None:
            return []
        window = settings.redis.window if limit is None else limit
        if window <= 0:
            return []
        try:
            raw_entries = client.lrange(self._messages_key(session_id), -window, -1) or []
        except Exception:
            log.exception("get_history failed", extra={"session_id": session_id})
            return []
        parsed = (Message.from_json(entry) for entry in raw_entries)
        return [m for m in parsed if m is not None]

    def count_messages(self, session_id: str) -> int:
        client = self._connect()
        if client is None:
            return 0
        try:
            return int(client.llen(self._messages_key(session_id)) or 0)
        except Exception:
            log.exception("count_messages failed", extra={"session_id": session_id})
            return 0

    def clear_session(self, session_id: str) -> bool:
        client = self._connect()
        if client is None:
            return False
        try:
            client.delete(self._messages_key(session_id), self._state_key(session_id))
            return True
        except Exception:
            log.exception("clear_session failed", extra={"session_id": session_id})
            return False

    # ── session state ─────────────────────────────────────────────────
    def save_session_state(self, session_id: str, state: dict[str, Any]) -> bool:
        """Merge-writes arbitrary JSON-serialisable session facts.

        Used for things the assistant learned and should not re-ask — e.g.
        `{"occupation": "software developer", "last_intent": "career"}`.
        """
        client = self._connect()
        if client is None or not state:
            return False
        key = self._state_key(session_id)
        try:
            client.hset(
                key,
                values={k: json.dumps(v, ensure_ascii=False) for k, v in state.items()},
            )
            client.expire(key, settings.redis.ttl_seconds)
            return True
        except Exception:
            log.exception("save_session_state failed", extra={"session_id": session_id})
            return False

    def get_session_state(self, session_id: str) -> dict[str, Any]:
        client = self._connect()
        if client is None:
            return {}
        try:
            raw = client.hgetall(self._state_key(session_id)) or {}
        except Exception:
            log.exception("get_session_state failed", extra={"session_id": session_id})
            return {}
        state: dict[str, Any] = {}
        for key, value in raw.items():
            try:
                state[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                state[key] = value
        return state


memory = RedisMemory()

__all__ = ["RedisMemory", "Message", "memory"]
