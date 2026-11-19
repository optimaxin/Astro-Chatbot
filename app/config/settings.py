"""Single source of truth for every environment-driven value.

Mirrors the convention already used by the Node backend's `core/config.ts`:
nothing else in the codebase reads `os.environ` directly, so every required
variable stays discoverable in one file and `.env.example` is easy to keep
in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ai-core/app/config/settings.py -> ai-core/
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

load_dotenv(BASE_DIR / ".env")


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    raw = _str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = _str(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _parse_prices(raw: str) -> dict[str, tuple[float, float]]:
    """Parse `model=in:out,model2=in:out` into {model: (input, output)}.

    Prices are USD per 1,000,000 tokens. Malformed entries are skipped rather
    than raising — a bad price string must never take the chatbot down, it
    only degrades a cost *estimate*.
    """
    prices: dict[str, tuple[float, float]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        model, _, pair = entry.partition("=")
        in_price, _, out_price = pair.partition(":")
        try:
            prices[model.strip()] = (float(in_price), float(out_price))
        except ValueError:
            continue
    return prices


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str = field(repr=False, default_factory=lambda: _str("OPENROUTER_API_KEY"))
    base_url: str = field(
        default_factory=lambda: _str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    app_url: str = field(default_factory=lambda: _str("OPENROUTER_APP_URL"))
    app_title: str = field(default_factory=lambda: _str("OPENROUTER_APP_TITLE", "Tredev Astro AI"))
    timeout_seconds: float = field(default_factory=lambda: _float("LLM_TIMEOUT_SECONDS", 60.0))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class GeminiSettings:
    """Google's Gemini API, used through its OpenAI-compatible endpoint.

    Google exposes `/v1beta/openai/`, which speaks the same chat-completions
    protocol as OpenRouter. That is why this provider needs no second SDK and
    no parallel code path in `llm_factory` — only a different base URL, key
    and model naming (`gemini-2.5-flash`, not `google/gemini-2.5-flash`).

    Added as a *testing* provider while the OpenRouter balance is empty.
    OpenRouter remains the production path; `LLM_PROVIDER` chooses.
    """

    api_key: str = field(repr=False,
        # GOOGLE_API_KEY is the name Google's own tooling uses, so it is
        # accepted as an alias rather than forcing a project-specific name.
        default_factory=lambda: _str("GEMINI_API_KEY") or _str("GOOGLE_API_KEY")
    )
    base_url: str = field(
        default_factory=lambda: _str(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    )
    timeout_seconds: float = field(default_factory=lambda: _float("LLM_TIMEOUT_SECONDS", 60.0))

    # Gemini 2.5 models "think" before answering, and those hidden reasoning
    # tokens are billed against max_tokens WITHOUT appearing in the reported
    # completion_tokens. Left on, a 700-token budget produced ~28 visible
    # tokens and replies were cut off mid-sentence (finish_reason=length).
    # Nothing this service asks of the model needs chain-of-thought, so it is
    # off by default. Set to low/medium/high to re-enable.
    reasoning_effort: str = field(
        default_factory=lambda: _str("GEMINI_REASONING_EFFORT", "none").lower()
    )

    main: str = field(default_factory=lambda: _str("GEMINI_MODEL", "gemini-2.5-flash"))
    fallback: str = field(
        default_factory=lambda: _str("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
    )
    classifier: str = field(
        default_factory=lambda: _str("GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash-lite")
    )
    # No OpenAI model is reachable on this endpoint, so the judge falls back to
    # a Gemini model rather than the gpt-4o-mini used via OpenRouter.
    evaluator: str = field(
        default_factory=lambda: _str("GEMINI_EVALUATOR_MODEL", "gemini-2.5-flash-lite")
    )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(p.strip().lower() for p in _str(name, default).split(",") if p.strip())


@dataclass(frozen=True)
class NvidiaSettings:
    """NVIDIA's hosted inference (build.nvidia.com / NIM), OpenAI-compatible.

    The primary provider for the beta phase, chosen with `LLM_PROVIDER=nvidia`.
    OpenRouter stays fully configured alongside it and is switched back to by
    changing that one variable — no code path is removed.

    Model ids carry the vendor prefix NVIDIA's catalogue uses
    (`mistralai/...`, `nvidia/...`), verified against GET /v1/models.
    """

    api_key: str = field(repr=False, default_factory=lambda: _str("NVIDIA_API_KEY"))
    base_url: str = field(
        default_factory=lambda: _str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )
    # 25 s: Lightning's measured p95 is ~8 s on the free tier, but single calls
    # have stalled for ~60 s. A stall now hands over to the fallback model
    # after 25 s instead of holding the user for a minute.
    timeout_seconds: float = field(
        default_factory=lambda: _float("NVIDIA_TIMEOUT_SECONDS", 25.0)
    )

    # Nemotron Lightning answers. Chosen 2026-11-12 after live tests: Mistral
    # Large 4 (the original plan) took 77-181s per reply on the free tier
    # (~60s queue, ~2 tok/s, reasons by default); Lightning replies in ~1-2s.
    # Lightning is text-only — image turns are routed to Bolt by model_router.
    main: str = field(
        default_factory=lambda: _str("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    # Nemotron Bolt: technical fallback for the main model, image turns, and
    # the "simple" intents listed in `light_intents`.
    fallback: str = field(
        default_factory=lambda: _str("NVIDIA_FALLBACK_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    )
    classifier: str = field(
        default_factory=lambda: _str("NVIDIA_CLASSIFIER_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    )
    evaluator: str = field(
        default_factory=lambda: _str("NVIDIA_EVALUATOR_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    )
    # Never routed to by production traffic; offered to the benchmark suite.
    # Set NVIDIA_MODEL to this to make Mistral Large 4 primary again.
    benchmark: str = field(
        default_factory=lambda: _str(
            "NVIDIA_BENCHMARK_MODEL", "mistralai/mistral-large-4-1t-a52b"
        )
    )
    # Model answering the intents below instead of `main`. Empty disables the
    # light route, so every answer goes to `main`.
    light_model: str = field(
        default_factory=lambda: _str("NVIDIA_LIGHT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    )
    # Intents (from the classifier) treated as "simple". Default is greetings
    # and thanks only: anything that reads a chart stays on the main model.
    light_intents: tuple[str, ...] = field(
        default_factory=lambda: _csv("NVIDIA_LIGHT_INTENTS", "smalltalk")
    )
    # Independent quality checker (shadow evaluator): nvidia/nemotron-3-ultra-550b-a55b, verified
    # live 2026-11-18. Advisory only (QUALITY_CHECKER_MODE=shadow, 10% sample). Set
    # NVIDIA_QUALITY_CHECKER_MODEL to another id to change it, or QUALITY_CHECKER_MODE=off to stop.
    quality_checker: str = field(
        default_factory=lambda: _str("NVIDIA_QUALITY_CHECKER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"))
    # Humanizer model. Empty = the main model, with the humanizer's own prompt/temperature.
    humanizer: str = field(default_factory=lambda: _str("NVIDIA_HUMANIZER_MODEL", ""))
    # Models that answer the MAIN turn with Nemotron reasoning on (classifier,
    # evaluator and fallback calls of the same model stay reasoning-off).
    # Empty = off everywhere (default). RC1 evidence (docs/rc1/rag-groundedness.md):
    # Bolt with reasoning scored 120/120 on the RAG suites at ~5 s per call.
    main_reasoning_models: tuple[str, ...] = field(
        default_factory=lambda: _csv("NVIDIA_MAIN_REASONING_MODELS", "")
    )
    # Extra max_tokens for those calls: reasoning tokens count against it.
    reasoning_token_budget: int = field(
        default_factory=lambda: _int("NVIDIA_REASONING_TOKEN_BUDGET", 1500)
    )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class ReliabilitySettings:
    """Retry, backoff and circuit-breaker policy for every LLM call.

    Applied by the gateway (`llm_factory`) to whichever provider is active, so
    the SDK's own retries are turned off — two retry layers stacked would
    multiply attempts and hide them from the usage ledger.
    """

    # Extra attempts on the SAME model after a retryable failure (timeout,
    # connection reset, 429, 5xx). 2 => at most 3 tries before fallback.
    max_retries: int = field(default_factory=lambda: max(0, _int("LLM_MAX_RETRIES", 2)))
    backoff_base_seconds: float = field(default_factory=lambda: _float("LLM_BACKOFF_BASE_SECONDS", 0.5))
    backoff_max_seconds: float = field(default_factory=lambda: _float("LLM_BACKOFF_MAX_SECONDS", 8.0))
    # Wall-clock budget for one logical call: all retries plus the fallback.
    # No new attempt starts once it is spent, so a turn can never hang.
    deadline_seconds: float = field(default_factory=lambda: _float("LLM_DEADLINE_SECONDS", 120.0))
    # Consecutive technical failures that open a model's circuit, and how long
    # it stays open before one trial request is let through.
    circuit_failure_threshold: int = field(
        default_factory=lambda: max(1, _int("LLM_CIRCUIT_FAILURE_THRESHOLD", 5))
    )
    circuit_cooldown_seconds: float = field(
        default_factory=lambda: _float("LLM_CIRCUIT_COOLDOWN_SECONDS", 30.0)
    )
    # Main answer only: if the primary has not answered after this many
    # seconds, also ask the fallback and take whichever answers first.
    # 0 disables hedging. See app/llm/hedging.py.
    hedge_after_seconds: float = field(default_factory=lambda: _float("LLM_HEDGE_AFTER_SECONDS", 6.0))


@dataclass(frozen=True)
class PipelineFlags:
    """Accuracy / latency / quality switches. Each one's "off" value restores
    the behaviour from before it existed, so any change can be rolled back
    from .env without a deploy."""

    # Full structured chart (house table, lords, degrees, D9, what is NOT
    # computed) in the prompt, instead of the older compact summary.
    structured_chart_context: bool = field(
        default_factory=lambda: _bool("STRUCTURED_CHART_CONTEXT", True))
    # auto   = strip the wrong sentences when a real answer remains (no extra
    #          model call), otherwise one corrective regeneration (default
    #          since 2026-11-15: round-4 profiling showed repairs on 35% of
    #          turns at ~2.7 s each, most of them a single wrong sentence);
    # repair = always one corrective regeneration, then strip what is still wrong;
    # strip  = drop contradicting sentences (no extra model call);
    # log    = detect and record only. Validation itself is never switched off.
    chart_validation_mode: str = field(
        default_factory=lambda: _str("CHART_VALIDATION_MODE", "auto").lower())
    # hybrid = keyword classifier when it is unambiguous, LLM otherwise;
    # llm = always the LLM (previous behaviour); keyword = never the LLM.
    classifier_mode: str = field(default_factory=lambda: _str("CLASSIFIER_MODE", "hybrid").lower())
    # Run the LLM judge after the reply is returned instead of before.
    evaluation_async: bool = field(default_factory=lambda: _bool("EVALUATION_ASYNC", True))
    # Write usage rows (JSONL / Redis counters / Postgres) on a background
    # thread instead of inside every LLM call.
    metering_async: bool = field(default_factory=lambda: _bool("METERING_ASYNC", True))
    # Batch the end-of-turn Redis writes into one pipelined request.
    redis_pipeline: bool = field(default_factory=lambda: _bool("REDIS_PIPELINE", True))
    # Re-ask once when the reply's script does not match the user's
    # (e.g. Devanagari reply to a Roman-script Hinglish question).
    language_guard: bool = field(default_factory=lambda: _bool("LANGUAGE_GUARD", True))
    # Retrieved passages are tagged [K#] with real titles, the model cites
    # them, and invalid citations are removed (app/rag/grounding.py).
    grounded_citations: bool = field(default_factory=lambda: _bool("GROUNDED_CITATIONS", True))
    # footer = one "Source: …" line under the answer; metadata = API field
    # only; inline = keep the [K#] tags in the text.
    citation_display: str = field(default_factory=lambda: _str("CITATION_DISPLAY", "metadata").lower())
    # Second pass that checks a knowledge answer against its passages and
    # removes what they do not state. off | cited (only answers that cite a
    # passage) | always (every answer with retrieved passages).
    # Off by default: measured 2026-11-13 on the 40 RAG cases it gave no
    # gain for Lightning (87.5% with and without) and cost ~1.2 s per answer.
    grounding_verifier: str = field(default_factory=lambda: _str("GROUNDING_VERIFIER", "off").lower())
    # segments = 1-4 bubbles, one per topic paragraph the model wrote, split
    # only at paragraph breaks (default); single = one bubble per reply;
    # bubbles = the old 120-char splitter with a closing question (rollback).
    reply_shape: str = field(default_factory=lambda: _str("REPLY_SHAPE", "segments").lower())
    # Deterministic reply-quality fixes (unneeded closing questions, repeated
    # sentences, hybrid terms, Devanagari glosses). off = report only.
    quality_fixes: bool = field(default_factory=lambda: _bool("REPLY_QUALITY_FIXES", True))
    # Contextual Hinglish grammar layer (app/nodes/language_quality.py):
    # off | detect (measure only) | deterministic (no model call) | correct.
    hinglish_grammar: str = field(default_factory=lambda: _str("HINGLISH_GRAMMAR", "correct").lower())
    # Independent quality checker (app/nodes/quality_check.py): off | shadow (default). Shadow
    # scores a sample (default 10%) of delivered replies in the background, advisory only:
    # deterministic validation stays the only gate. Needs NVIDIA_QUALITY_CHECKER_MODEL.
    quality_checker_mode: str = field(default_factory=lambda: _str("QUALITY_CHECKER_MODE", "shadow").lower())
    quality_checker_sample_rate: float = field(
        default_factory=lambda: max(0.0, min(1.0, _float("QUALITY_CHECKER_SAMPLE_RATE", 0.1))))
    # Humanizer (app/nodes/humanize.py): off | shadow (default) | on. shadow runs it in the
    # background and records what it would have changed without changing the reply; on ships a rewrite only if it passes the
    # fact-preservation guard. Runs only when the reply still has language issues
    # (HUMANIZER_MIN_ISSUES) — "only when it provides meaningful improvement".
    humanizer_mode: str = field(default_factory=lambda: _str("HUMANIZER_MODE", "shadow").lower())
    humanizer_min_issues: int = field(default_factory=lambda: max(1, _int("HUMANIZER_MIN_ISSUES", 1)))
    # Shadow mode only: share of otherwise-eligible Hindi/Hinglish replies the grammar layer already
    # marked clean ("already_natural") that are still sent to the humanizer, and the rewrite compared
    # blind against the original by the independent judge. Measures whether the humanizer can improve
    # replies that have no known issue. Never applied to a user-facing reply (ignored in "on" mode).
    humanizer_clean_sample_rate: float = field(
        default_factory=lambda: max(0.0, min(1.0, _float("HUMANIZER_CLEAN_SAMPLE_RATE", 0.05))))
    # Guarded concise rewrite (app/nodes/concise.py): off (default, awaiting approval) | shadow | on.
    # Rewrites only over-long bubbles, keeps the original on any guard failure or timeout.
    concise_rewrite_mode: str = field(default_factory=lambda: _str("CONCISE_REWRITE_MODE", "off").lower())
    concise_rewrite_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("CONCISE_REWRITE_TIMEOUT_SECONDS", 6.0)))
    # Shadow evaluation only: also write the original and proposed reply TEXT to the
    # shadow log, for human review. Off by default: production logs keep no reply text.
    concise_shadow_log_text: bool = field(default_factory=lambda: _bool("CONCISE_SHADOW_LOG_TEXT", False))


@dataclass(frozen=True)
class AuthSettings:
    """Identity for /chat, /chart and /feedback. See app/api/auth.py."""

    # optional | required | off   (off = trust the body's user_id: rollback only)
    mode: str = field(default_factory=lambda: _str("AUTH_MODE", "optional").lower())
    # The backend that issued the access token verifies it (signature, expiry,
    # suspension, revoked sessions) — ai-core never holds JWT_SECRET.
    backend_url: str = field(
        default_factory=lambda: _str("AUTH_BACKEND_URL", _str("TREDEV_BACKEND_URL", "http://localhost:4000")))
    verify_path: str = field(default_factory=lambda: _str("AUTH_VERIFY_PATH", "/api/auth/me"))
    verify_timeout_seconds: float = field(default_factory=lambda: _float("AUTH_VERIFY_TIMEOUT_SECONDS", 3.0))
    cache_ttl_seconds: float = field(default_factory=lambda: _float("AUTH_CACHE_TTL_SECONDS", 60.0))
    # Signs anonymous session tokens. Set it in production: without it a
    # random per-process key is used and anonymous sessions reset on restart.
    session_secret: str = field(default_factory=lambda: _str("CHAT_SESSION_SECRET"))
    # Refuse to start without a shared CHAT_SESSION_SECRET (≥32 chars). With a
    # per-process key, a token issued by one worker is rejected by the next,
    # so anonymous chats silently lose their history. Set to false only for a
    # single-process local run.
    require_session_secret: bool = field(
        default_factory=lambda: _bool("AUTH_REQUIRE_SESSION_SECRET", True))


@dataclass(frozen=True)
class AnthropicSettings:
    """Cross-provider fallback for the main-turn answer only.

    Used when `LLM_PROVIDER=gemini` and Gemini's own call fails — the turn
    retries once against Claude Haiku instead of a second Gemini model, so an
    outage at Google doesn't take the whole answer path down with it. Classify
    and evaluate roles are unaffected and keep falling back within Gemini.
    """

    api_key: str = field(repr=False, default_factory=lambda: _str("ANTHROPIC_API_KEY"))
    base_url: str = field(default_factory=lambda: _str("ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    model: str = field(default_factory=lambda: _str("ANTHROPIC_FALLBACK_MODEL", "claude-haiku-5-5"))
    timeout_seconds: float = field(default_factory=lambda: _float("LLM_TIMEOUT_SECONDS", 60.0))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class ModelSettings:
    """Per-role model configuration.

    Roles are addressed by name (`job-main-turn`, `job-classify`,
    `job-evaluate`) so nothing hard-codes a model id. Today the values come
    from env; the `llm_factory` reads this object, so moving the source to
    Postgres later means changing only this class.
    """

    main: str = field(default_factory=lambda: _str("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    fallback: str = field(
        default_factory=lambda: _str("OPENROUTER_FALLBACK_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
    )
    classifier: str = field(
        default_factory=lambda: _str("CLASSIFIER_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
    )
    evaluator: str = field(default_factory=lambda: _str("EVALUATOR_MODEL", "google/gemini-2.0-flash-001"))

    main_temperature: float = field(default_factory=lambda: _float("MAIN_TEMPERATURE", 0.7))
    main_max_tokens: int = field(default_factory=lambda: _int("MAIN_MAX_TOKENS", 700))
    classifier_temperature: float = field(default_factory=lambda: _float("CLASSIFIER_TEMPERATURE", 0.0))
    evaluator_temperature: float = field(default_factory=lambda: _float("EVALUATOR_TEMPERATURE", 0.0))


@dataclass(frozen=True)
class PricingSettings:
    table: dict[str, tuple[float, float]] = field(
        default_factory=lambda: _parse_prices(_str("MODEL_PRICES"))
    )
    default_input_per_1m: float = field(default_factory=lambda: _float("DEFAULT_INPUT_PRICE_PER_1M", 0.0))
    default_output_per_1m: float = field(default_factory=lambda: _float("DEFAULT_OUTPUT_PRICE_PER_1M", 0.0))

    def for_model(self, model: str) -> tuple[float, float]:
        return self.table.get(model, (self.default_input_per_1m, self.default_output_per_1m))


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str = field(repr=False, default_factory=lambda: _str("PINECONE_API_KEY"))
    index_name: str = field(default_factory=lambda: _str("PINECONE_INDEX_NAME", "tredev-astro-knowledge"))
    namespace: str = field(default_factory=lambda: _str("PINECONE_NAMESPACE", "astrology-knowledge"))
    cloud: str = field(default_factory=lambda: _str("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: _str("PINECONE_REGION", "us-east-1"))
    embedding_model: str = field(default_factory=lambda: _str("EMBEDDING_MODEL", "llama-text-embed-v2"))
    embedding_dimension: int = field(default_factory=lambda: _int("EMBEDDING_DIMENSION", 1024))
    top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 4))
    # Absolute floor, calibrated against measured llama-text-embed-v2 scores:
    # a short query like "Will I get a promotion?" tops out near 0.14 against
    # these long passages, while an off-topic greeting sits below 0.09. A fixed
    # 0.15 discarded correct career hits, so this floor only removes clear
    # noise and `relative_min` below does the real filtering.
    min_score: float = field(default_factory=lambda: _float("RAG_MIN_SCORE", 0.10))
    # Keep a chunk only if it scores at least this fraction of the best hit.
    # A relative cutoff stays correct across query lengths and survives a
    # change of embedding model, which an absolute threshold does not.
    relative_min: float = field(default_factory=lambda: _float("RAG_RELATIVE_MIN_SCORE", 0.55))
    # Upper bound on one retrieval (embed + query). 0 disables the bound.
    timeout_seconds: float = field(default_factory=lambda: _float("RAG_TIMEOUT_SECONDS", 6.0))

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class PremiumKnowledgeSettings:
    """The Tradeva Premium knowledge corpus, through its existing retrieval
    service (kb_rag, `POST /api/rag/retrieve`). See app/rag/premium_knowledge.py.

    Chatbot-only configuration: the Premium report generator keeps its own
    settings and is not affected by anything here. Unset URL = off (the legacy
    `astrology-knowledge` namespace is used, exactly as before)."""

    url: str = field(default_factory=lambda: _str("PREMIUM_RAG_URL"))
    # premium = Premium corpus, legacy namespace only as fallback;
    # legacy  = previous behaviour (ignore the Premium corpus).
    source: str = field(default_factory=lambda: _str("KNOWLEDGE_SOURCE", "premium").lower())
    timeout_seconds: float = field(default_factory=lambda: _float("PREMIUM_RAG_TIMEOUT_SECONDS", 6.0))
    max_queries: int = field(default_factory=lambda: _int("PREMIUM_RAG_MAX_QUERIES", 3))
    top_k_per_query: int = field(default_factory=lambda: _int("PREMIUM_RAG_TOP_K", 2))
    max_chunks: int = field(default_factory=lambda: _int("PREMIUM_RAG_MAX_CHUNKS", 4))
    max_chars_per_chunk: int = field(default_factory=lambda: _int("PREMIUM_RAG_MAX_CHARS", 900))
    # Same calibrated floor the Premium service uses (relevant ≈0.56–0.74,
    # off-topic ≤0.46 on this index).
    min_score: float = field(default_factory=lambda: _float("PREMIUM_RAG_MIN_SCORE", 0.48))
    # Reuse identical sub-query results for this long (the corpus is static).
    # 0 disables the cache.
    cache_seconds: float = field(default_factory=lambda: _float("PREMIUM_RAG_CACHE_SECONDS", 3600.0))

    @property
    def configured(self) -> bool:
        return bool(self.url) and self.source == "premium"


@dataclass(frozen=True)
class RedisSettings:
    url: str = field(default_factory=lambda: _str("UPSTASH_REDIS_REST_URL"))
    token: str = field(default_factory=lambda: _str("UPSTASH_REDIS_REST_TOKEN"))
    max_messages: int = field(default_factory=lambda: _int("HISTORY_MAX_MESSAGES", 40))
    window: int = field(default_factory=lambda: _int("HISTORY_WINDOW", 8))
    ttl_seconds: int = field(default_factory=lambda: _int("SESSION_TTL_SECONDS", 604_800))
    # A chart is derived from immutable birth data, so it can be cached far
    # longer than a conversation. 24h bounds how long a backend correction
    # takes to propagate.
    chart_ttl_seconds: int = field(default_factory=lambda: _int("CHART_CACHE_TTL_SECONDS", 86_400))

    @property
    def configured(self) -> bool:
        return bool(self.url and self.token)


@dataclass(frozen=True)
class KundliSettings:
    source: str = field(default_factory=lambda: _str("KUNDLI_SOURCE", "sample").lower())
    backend_url: str = field(default_factory=lambda: _str("TREDEV_BACKEND_URL", "http://localhost:4000"))
    sample_path: Path = field(default_factory=lambda: DATA_DIR / "sample_kundli.json")


@dataclass(frozen=True)
class UsageStoreSettings:
    """Postgres sink for per-request AI usage, read by the admin Token Meter.

    Points at the same database the Node backend uses. Unset means the sink is
    simply off: usage still reaches the JSONL ledger and the Redis counters,
    and only the admin dashboard goes without data.
    """

    dsn: str = field(
        # DATABASE_URL is accepted as an alias so a deployment that already
        # exports the backend's connection string needs no second variable.
        default_factory=lambda: _str("AI_USAGE_DATABASE_URL") or _str("DATABASE_URL")
    )
    pool_size: int = field(default_factory=lambda: _int("AI_USAGE_POOL_SIZE", 4))
    timeout_seconds: float = field(default_factory=lambda: _float("AI_USAGE_DB_TIMEOUT", 5.0))
    # Providers publish prices in USD; the admin panel reports rupees. The rate
    # is snapshotted onto every row, so changing it here affects new rows only
    # and never rewrites what past months reportedly cost.
    usd_to_inr: float = field(default_factory=lambda: _float("USD_TO_INR", 88.0))

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)


@dataclass(frozen=True)
class ApiSettings:
    """HTTP layer. Only the API reads these; the terminal REPL ignores them."""

    host: str = field(default_factory=lambda: _str("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("API_PORT", 8000))
    # Comma-separated origins allowed to call /chat from a browser. "*" is the
    # dev default; a real deployment must list its own origins, because the
    # API is the only thing standing between a browser and the LLM key.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _str("API_CORS_ORIGINS", "*").split(",") if o.strip()
        )
    )
    default_persona_id: str = field(default_factory=lambda: _str("DEFAULT_PERSONA_ID", "vedic_general"))
    # auto | whatsapp_hinglish | english | hindi | formal | detailed.
    # "auto" replies in the language the user wrote in (English -> English,
    # Hinglish -> Hinglish, Devanagari -> Hindi); a concrete id pins every turn.
    default_response_style: str = field(
        default_factory=lambda: _str("DEFAULT_RESPONSE_STYLE", "auto").lower()
    )
    rate_limit_per_minute: int = field(default_factory=lambda: _int("RATE_LIMIT_PER_MINUTE", 20))
    rate_limit_enabled: bool = field(default_factory=lambda: _bool("RATE_LIMIT_ENABLED", True))
    max_question_chars: int = field(default_factory=lambda: _int("MAX_QUESTION_CHARS", 2000))


@dataclass(frozen=True)
class IngestionSettings:
    """Offline knowledge ingestion. Never read during a /chat request."""

    # Approximate target chunk size in tokens; converted to characters with
    # the usual ~4 chars/token heuristic at the point of use. Used only by the
    # PDF path — the curated .txt files are split on their own headings.
    chunk_size_tokens: int = field(default_factory=lambda: _int("CHUNK_SIZE", 500))
    chunk_overlap_tokens: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 60))
    chars_per_token: int = field(default_factory=lambda: _int("CHARS_PER_TOKEN", 4))

    @property
    def chunk_chars(self) -> int:
        return self.chunk_size_tokens * self.chars_per_token

    @property
    def overlap_chars(self) -> int:
        return self.chunk_overlap_tokens * self.chars_per_token


@dataclass(frozen=True)
class Settings:
    # "nvidia" (beta default), "openrouter" (kept ready, switch back by
    # setting this) or "gemini" (testing, direct Google API). Everything else
    # about the pipeline is provider-agnostic; only `llm_factory` reads this.
    llm_provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "nvidia").lower())

    openrouter: OpenRouterSettings = field(default_factory=OpenRouterSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    nvidia: NvidiaSettings = field(default_factory=NvidiaSettings)
    reliability: ReliabilitySettings = field(default_factory=ReliabilitySettings)
    flags: PipelineFlags = field(default_factory=PipelineFlags)
    auth: AuthSettings = field(default_factory=AuthSettings)
    anthropic: AnthropicSettings = field(default_factory=AnthropicSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    pricing: PricingSettings = field(default_factory=PricingSettings)
    pinecone: PineconeSettings = field(default_factory=PineconeSettings)
    premium_knowledge: PremiumKnowledgeSettings = field(default_factory=PremiumKnowledgeSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    kundli: KundliSettings = field(default_factory=KundliSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    usage_store: UsageStoreSettings = field(default_factory=UsageStoreSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)

    # Tag for usage rows from this process when no script overrides it.
    traffic_source: str = field(default_factory=lambda: _str("AI_TRAFFIC_SOURCE", "user").lower())
    default_session_id: str = field(default_factory=lambda: _str("DEFAULT_SESSION_ID", "test-session-001"))
    default_user_id: str = field(default_factory=lambda: _str("DEFAULT_USER_ID", "test-user-001"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())
    log_file: str = field(default_factory=lambda: _str("LOG_FILE", "logs/ai-core.jsonl"))
    # Size of the in-process ring buffer of recent LLM calls (app/llm/usage_tracker.py).
    # Memory only: the ledger and Postgres keep everything.
    usage_tracker_max_records: int = field(default_factory=lambda: max(1, _int("USAGE_TRACKER_MAX_RECORDS", 5000)))
    enable_rag: bool = field(default_factory=lambda: _bool("ENABLE_RAG", True))
    enable_evaluation: bool = field(default_factory=lambda: _bool("ENABLE_EVALUATION", True))
    # Fraction of turns the LLM judge actually grades, 0.0-1.0.
    #
    # Evaluation is a second LLM call per turn. Running it on every production
    # request roughly doubles cost for a metric nobody reads in real time, so
    # production samples and development sets this to 1.0. Sampling is on the
    # *judge* only — the deterministic checks are free and always run, so an
    # unsampled turn still catches a prompt leak or an invented placement.
    evaluation_sample_rate: float = field(
        default_factory=lambda: max(0.0, min(1.0, _float("EVALUATION_SAMPLE_RATE", 1.0)))
    )

    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    knowledge_dir: Path = KNOWLEDGE_DIR


settings = Settings()

__all__ = ["settings", "Settings", "BASE_DIR", "DATA_DIR", "KNOWLEDGE_DIR"]
