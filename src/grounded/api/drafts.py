"""The generation API: /v1 (ADR-016).

    GET  /v1/health     liveness, provider circuit state - no key needed
    GET  /v1/prompts    the prompt registry
    POST /v1/drafts     a job description in, a gated draft out
    GET  /v1/usage      this key's budget for today

Every draft runs the same pipeline as the CLI (`generation.service`): the
grounding gate decides what is returned, whatever the model wrote.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from sqlalchemy.orm import Session

from grounded.api.deps import get_session
from grounded.api.guards import ApiKey, CircuitOpenError, authenticate, parse_keys
from grounded.config import Settings, get_settings
from grounded.generation import prompt
from grounded.generation.adapters import make_client
from grounded.generation.generate import ChatClient, GenerationError
from grounded.generation.llm import LLMError
from grounded.generation.service import create_draft
from grounded.retrieval.search import MAX_QUERY_CHARS

router = APIRouter(prefix="/v1", tags=["generation"])


def _limit_key(request: Request) -> str:
    presented = request.headers.get("x-api-key")
    key = authenticate(presented, parse_keys(get_settings().api_keys))
    return f"key:{key.name}" if key else f"ip:{request.client.host if request.client else '-'}"


limiter = Limiter(key_func=_limit_key)


def _rate() -> str:
    return get_settings().rate_limit


def require_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> ApiKey:
    key = authenticate(x_api_key, parse_keys(get_settings().api_keys))
    if key is None:
        raise HTTPException(401, "missing or unknown X-API-Key")
    return key


KeyDep = Annotated[ApiKey, Depends(require_key)]
SessionDep = Annotated[Session, Depends(get_session)]


class DraftIn(BaseModel):
    job_description: str = Field(min_length=20, max_length=MAX_QUERY_CHARS)
    k: int = Field(12, ge=1, le=30)
    rerank: bool = False
    prompt_version: str | None = None
    cache: bool = False


class BulletOut(BaseModel):
    text: str
    evidence_ids: list[str]
    entailment: float | None = None
    reason: str | None = None


class DraftOut(BaseModel):
    model: str
    prompt_version: str
    prompt_fingerprint: str
    verified: bool
    threshold: float
    cache_hit: bool
    redacted: int
    tokens: int
    kept: list[BulletOut]
    dropped: list[BulletOut]


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "provider_circuit": request.app.state.breaker.state}


@router.get("/prompts")
def prompts() -> list[dict[str, str]]:
    default = get_settings().prompt_version
    return [
        {
            "version": s.version,
            "fingerprint": s.fingerprint,
            "notes": s.notes,
            "default": str(s.version == default).lower(),
        }
        for s in prompt.REGISTRY.values()
    ]


@router.get("/usage")
def usage(request: Request, key: KeyDep) -> dict[str, Any]:
    ledger = request.app.state.ledger
    return {
        "key": key.name,
        "daily_tokens": key.daily_tokens,
        "used_today": ledger.used(key),
        "remaining_today": ledger.remaining(key),
    }


@router.post("/drafts", response_model=DraftOut, responses={429: {}, 503: {}})
@limiter.limit(_rate)
def create(request: Request, body: DraftIn, key: KeyDep, session: SessionDep) -> Any:
    state = request.app.state
    if state.ledger.remaining(key) <= 0:
        raise HTTPException(429, f"daily token budget of {key.daily_tokens:,} used up")
    breaker = state.breaker
    settings: Settings = get_settings()

    def guarded(spec: str, config: Settings) -> ChatClient:
        breaker.before_call()
        return state.client_factory(spec, config)  # type: ignore[no-any-return]

    try:
        outcome = create_draft(
            session,
            body.job_description,
            settings,
            guarded,
            state.get_verifier(),
            k=body.k,
            rerank=body.rerank,
            prompt_version=body.prompt_version,
            use_cache=body.cache,
        )
    except CircuitOpenError as exc:
        return JSONResponse(
            {"detail": str(exc)},
            status_code=503,
            headers={"Retry-After": str(max(int(exc.retry_after), 1))},
        )
    except LLMError as exc:
        breaker.failure()
        raise HTTPException(502, f"provider error: {exc}") from exc
    except GenerationError as exc:
        breaker.success()  # the provider answered; the output was unusable
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if outcome.cache_similarity is None:
        breaker.success()
    state.ledger.charge(key, outcome.tokens)
    g, report = outcome.generation, outcome.report
    return DraftOut(
        model=g.model,
        prompt_version=g.prompt_version,
        prompt_fingerprint=g.prompt_fingerprint,
        verified=report.verified,
        threshold=report.threshold,
        cache_hit=outcome.cache_similarity is not None,
        redacted=g.redacted,
        tokens=outcome.tokens,
        kept=[
            BulletOut(
                text=k.bullet.text,
                evidence_ids=list(k.bullet.evidence_ids),
                entailment=k.entailment,
            )
            for k in report.kept
        ],
        dropped=[
            BulletOut(text=d.bullet.text, evidence_ids=list(d.bullet.evidence_ids), reason=d.reason)
            for d in report.dropped
        ],
    )


def default_client_factory(spec: str, settings: Settings) -> ChatClient:
    return make_client(spec, settings)
