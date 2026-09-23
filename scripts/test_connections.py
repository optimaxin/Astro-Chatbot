"""Independently verify every external dependency before touching the chatbot.

    python -m scripts.test_connections

Each service is checked in isolation and a failure in one does not prevent
the others from being reported. Exit code is 0 only when all three pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import settings  # noqa: E402
from app.console import LINE, fail_mark, ok_mark, safe_print  # noqa: E402
from app.llm.llm_factory import (  # noqa: E402
    PROVIDER_GEMINI,
    PROVIDER_OPENROUTER,
    ROLE_MAIN,
    LLMNotConfigured,
    active_provider,
    describe_roles,
    get_llm,
)
from app.memory.redis_memory import memory  # noqa: E402
from app.rag.pinecone_client import pinecone_client  # noqa: E402


def check_llm() -> tuple[bool, str]:
    """Round-trips one real completion against whichever provider is active."""
    provider = active_provider()
    if provider == PROVIDER_GEMINI and not settings.gemini.configured:
        return False, "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set (see ai-core/.env.example)"
    if provider == PROVIDER_OPENROUTER and not settings.openrouter.configured:
        return False, "OPENROUTER_API_KEY is not set (see ai-core/.env.example)"
    try:
        llm = get_llm(role=ROLE_MAIN)
        result = llm.complete(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            session_id="connection-test",
            user_id="connection-test",
            # Generous for a one-word reply, because a thinking model (Gemini
            # 2.5 Flash) spends part of this budget on internal reasoning
            # before it writes anything. At max_tokens=10 it returned an empty
            # message and the check reported a spurious fallback.
            max_tokens=64,
        )
    except LLMNotConfigured as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    usage = result.usage
    tokens = (
        f"{usage.input_tokens} in / {usage.output_tokens} out"
        if usage.tokens_available
        else "usage not reported by provider"
    )
    return True, f"model '{usage.model}' responded in {usage.latency_ms} ms ({tokens})"


def check_pinecone() -> tuple[bool, str]:
    return pinecone_client.ping()


def check_redis() -> tuple[bool, str]:
    return memory.ping()


def main() -> int:
    safe_print(LINE)
    safe_print("TREDEV ASTRO AI")
    safe_print("INFRASTRUCTURE TEST")
    safe_print(LINE)
    safe_print()

    checks = [
        (f"LLM ({active_provider()})", check_llm),
        ("Pinecone", check_pinecone),
        ("Upstash Redis", check_redis),
    ]

    results: list[tuple[str, bool, str]] = []
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 — one broken check must not hide the rest
            ok, detail = False, f"unexpected {type(exc).__name__}: {exc}"
        results.append((name, ok, detail))
        marker = ok_mark() if ok else fail_mark()
        status = "connected" if ok else "FAILED"
        safe_print(f"{marker} {name} {status}")
        safe_print(f"    {detail}")

    safe_print()
    safe_print(LINE)
    safe_print("MODEL ROUTING")
    safe_print(LINE)
    for role, cfg in describe_roles().items():
        safe_print(f"  {role}")
        safe_print(f"      model    : {cfg['model']}")
        safe_print(f"      fallback : {cfg['fallback'] or '(none)'}")
        price = cfg["price_per_1m"]
        safe_print(f"      price/1M : ${price['input']} in / ${price['output']} out")
    safe_print()

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        safe_print(f"{fail_mark()} {len(failed)} service(s) not working: {', '.join(failed)}")
        safe_print()
        safe_print("Fix the configuration above, then re-run:")
        safe_print("    python -m scripts.test_connections")
        return 1

    safe_print("All infrastructure services are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
