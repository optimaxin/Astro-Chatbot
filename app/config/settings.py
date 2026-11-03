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
    api_key: str = field(default_factory=lambda: _str("OPENROUTER_API_KEY"))
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

    api_key: str = field(
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


@dataclass(frozen=True)
class AnthropicSettings:
    """Cross-provider fallback for the main-turn answer only.

    Used when `LLM_PROVIDER=gemini` and Gemini's own call fails — the turn
    retries once against Claude Haiku instead of a second Gemini model, so an
    outage at Google doesn't take the whole answer path down with it. Classify
    and evaluate roles are unaffected and keep falling back within Gemini.
    """

    api_key: str = field(default_factory=lambda: _str("ANTHROPIC_API_KEY"))
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
    api_key: str = field(default_factory=lambda: _str("PINECONE_API_KEY"))
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

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


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
    # "openrouter" (production) or "gemini" (testing, direct Google API).
    # Everything else about the pipeline is provider-agnostic; only
    # `llm_factory` reads this.
    llm_provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "openrouter").lower())

    openrouter: OpenRouterSettings = field(default_factory=OpenRouterSettings)
    gemini: GeminiSettings = field(default_factory=GeminiSettings)
    anthropic: AnthropicSettings = field(default_factory=AnthropicSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    pricing: PricingSettings = field(default_factory=PricingSettings)
    pinecone: PineconeSettings = field(default_factory=PineconeSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    kundli: KundliSettings = field(default_factory=KundliSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    usage_store: UsageStoreSettings = field(default_factory=UsageStoreSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)

    default_session_id: str = field(default_factory=lambda: _str("DEFAULT_SESSION_ID", "test-session-001"))
    default_user_id: str = field(default_factory=lambda: _str("DEFAULT_USER_ID", "test-user-001"))
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())
    log_file: str = field(default_factory=lambda: _str("LOG_FILE", "logs/ai-core.jsonl"))
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
