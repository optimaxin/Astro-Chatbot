"""Structured JSON logging with a per-turn trace id.

Every log line carries the `request_id` of the turn that produced it, so a
single conversational turn can be reconstructed end to end from the log file.
Console output stays human-readable and quiet (WARNING+) so it never fights
with the terminal chat UI; the full INFO-level stream goes to the JSONL file.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from pathlib import Path
from typing import Any

from app.config.settings import BASE_DIR, settings

# Set once per turn in the graph runner; read by the formatter below.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SENSITIVE = ("api_key", "token", "secret", "password", "authorization")

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


def _redact(key: str, value: Any) -> Any:
    """Never let a credential reach the log file (Phase 17 requirement)."""
    if any(marker in key.lower() for marker in _SENSITIVE):
        return "***redacted***"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _redact(key, value)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"[{record.levelname.lower()}] {record.name}: {record.getMessage()}"


_configured = False


def setup_logging() -> None:
    """Idempotent — safe to call from main, scripts and tests alike."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("aicore")
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    log_path = Path(settings.log_file)
    if not log_path.is_absolute():
        log_path = BASE_DIR / log_path
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    except OSError:
        # A read-only or unwritable log dir must not stop the chatbot.
        root.warning("could not open log file, continuing with console only")

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"aicore.{name}")


__all__ = ["get_logger", "setup_logging", "request_id_var"]
