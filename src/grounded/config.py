"""Typed application settings, loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every configurable value lives here — never read os.environ directly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "ci", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    host: str = "127.0.0.1"
    """Loopback by default: the admin has no authentication until step 60, so
    it must not be reachable from the network. The container sets 0.0.0.0,
    where the port mapping is the boundary."""
    port: int = 8000

    model_cache_dir: str = ".cache/models"
    """Where fastembed keeps downloaded models. Its default is the system temp
    directory, which Windows clears — forcing a silent re-download."""

    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openrouter/free"
    """OpenRouter's free router: it always selects an available free model that
    supports tool calling, and every draft reports which model it actually got.
    Free models come and go — the first pinned default stopped being free
    within a day — so drafts use the router, and evaluation runs, which must
    compare like with like, require a pinned model (ADR-007). List current
    candidates with `python -m grounded.generation models`."""
    openrouter_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    """Only for `openai:<model>` specs (ADR-012)."""
    ollama_base_url: str = "http://localhost:11434/v1"
    """Only for `ollama:<model>` specs: a local model, no key, nothing leaves the machine."""

    auth: Literal["off", "jwt"] = "off"
    """`off`: local use - anonymous callers are the default tenant and may read
    and write evidence (not generate). `jwt`: every call needs a bearer token or
    API key, and the admin and UI pages are not served (ADR-017)."""
    jwt_secret: SecretStr = SecretStr("")
    jwt_issuer: str = "grounded"
    jwt_audience: str = "grounded-api"
    default_daily_tokens: int = 50_000
    """Budget for tokens that do not carry a `budget` claim."""

    api_keys: str = ""
    """`name=sha256:daily_tokens,...` - hashes only; make one with
    `python -m grounded.api keygen <name>` (ADR-016)."""
    rate_limit: str = "10/minute"
    """Per API key on POST /v1/drafts (slowapi syntax)."""
    circuit_failures: int = 3
    """Consecutive provider failures that open the circuit breaker."""
    circuit_cooldown_s: float = 60.0

    otel_exporter: Literal["none", "console", "otlp"] = "none"
    """Where traces go (ADR-018). `otlp` reads OTEL_EXPORTER_OTLP_ENDPOINT."""
    llm_input_price_per_mtok: float | None = None
    llm_output_price_per_mtok: float | None = None
    """USD per million tokens, for cost on traces; free-tier models cost $0."""

    llm_max_tokens: int = 2000
    """Output-token cap per call (ADR-014): a draft of at most 8 bullets of at most
    300 characters needs far less; the cap bounds what a hostile JD can cost."""
    local_only: bool = False
    """Refuse every provider that is not on this machine (`ollama:`) - the fully
    local mode (ADR-015)."""
    redaction: Literal["off", "patterns", "presidio"] = "patterns"
    """What is redacted before a request goes to a non-local provider (ADR-015)."""
    redact_terms: str = ""
    """Comma-separated names to always redact - employers, clients, products."""

    prompt_version: str = "generate-v2"
    """The registered prompt drafts use (ADR-012); list them with
    `python -m grounded.generation prompts`."""

    semantic_cache_path: str = ".cache/semantic-cache.jsonl"
    """Drafts reused for near-identical requests (ADR-013). Gitignored."""

    @property
    def semantic_cache_path_obj(self) -> Path:
        return Path(self.semantic_cache_path)

    cache_threshold: float = 0.90
    """Cosine similarity of job descriptions needed for a cache hit, set from
    `cache-bench`: the lowest threshold with no false hits even on similarity
    alone; with the evidence rule it hits 50% of repeated requests (ADR-013)."""

    verifier: Literal["nli-deberta"] = "nli-deberta"
    """The entailment model that decides which bullets survive (ADR-009)."""
    verifier_threshold: float = 0.95
    """Entailment probability a bullet needs, set from calibration: 0.95 halved
    false accepts against 0.8 and rejected no supported pair (ADR-009)."""

    embedder: Literal["bge-small", "hashing"] = "bge-small"
    """The dense model (ADR-004). `hashing` needs no download and is what the
    tests use; it is a lexical baseline, never a silent stand-in."""

    database_url: str = "postgresql+psycopg://app:app@localhost:5433/app"
    """The local compose database. Port 5433, not 5432, so this stack and the
    lakehouse's can run side by side; the driver is named so SQLAlchemy never
    has to guess between psycopg2 and psycopg 3."""


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
