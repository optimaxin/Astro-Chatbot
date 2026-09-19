"""Terminal output helpers.

Windows consoles still default to cp1252, which cannot encode the check/cross
marks the terminal UI uses. This reconfigures stdout/stderr to UTF-8 where the
runtime allows it and silently degrades to ASCII markers where it does not, so
the chatbot never dies on a `UnicodeEncodeError` mid-answer.
"""

from __future__ import annotations

import sys

LINE = "=" * 40
RULE = "-" * 40

_UNICODE_OK = False


def enable_utf8() -> bool:
    """Switch the streams to UTF-8. Returns whether marks can be drawn."""
    global _UNICODE_OK
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    _UNICODE_OK = "utf" in encoding
    return _UNICODE_OK


enable_utf8()


def ok_mark() -> str:
    return "[✓]" if _UNICODE_OK else "[OK]"


def fail_mark() -> str:
    return "[✗]" if _UNICODE_OK else "[X]"


def warn_mark() -> str:
    return "[!]"


def safe_print(*parts: object, **kwargs: object) -> None:
    """print() that survives an un-encodable character rather than raising."""
    try:
        print(*parts, **kwargs)  # type: ignore[arg-type]
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        cleaned = [
            str(p).encode(encoding, errors="replace").decode(encoding, errors="replace")
            for p in parts
        ]
        print(*cleaned, **kwargs)  # type: ignore[arg-type]


__all__ = ["LINE", "RULE", "ok_mark", "fail_mark", "warn_mark", "safe_print", "enable_utf8"]
