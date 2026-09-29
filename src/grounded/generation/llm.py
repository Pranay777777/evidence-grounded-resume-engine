"""A minimal client for OpenAI-compatible chat APIs — OpenRouter first.

One endpoint, no SDK: easy to mock, explicit about retries. Free tiers rate
limit hard, so 429 and 5xx responses are retried with exponential backoff
that honours `Retry-After`; authentication and request errors are not,
because retrying them cannot help.

The API key never appears in an exception, a log line or a repr: it lives in
a `SecretStr` and only in the Authorization header.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import SecretStr

RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The provider could not produce a response."""


@dataclass(frozen=True)
class ChatResponse:
    content: str | None
    tool_arguments: str | None
    """The raw JSON string of the first tool call's arguments, if any."""
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        require_key: bool = True,
        key_hint: str = "OPENROUTER_API_KEY (a free key from openrouter.ai/keys)",
        attribution: bool = True,
        max_tokens: int = 2000,
    ) -> None:
        if require_key and not api_key.get_secret_value():
            raise LLMError(f"no API key - set {key_hint}")
        self.model = model
        self.max_tokens = max_tokens
        """Caps every completion: one request can never cost more than this many
        output tokens (OWASP LLM06:2026, unbounded consumption)."""
        self.max_retries = max_retries
        self._sleep = sleep
        headers = {}
        if api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        if attribution:
            # Optional attribution headers OpenRouter documents.
            headers["HTTP-Referer"] = (
                "https://github.com/Pranay777777/evidence-grounded-resume-engine"
            )
            headers["X-Title"] = "evidence-grounded-resume-engine"
        self._http = httpx.Client(
            base_url=base_url, timeout=timeout, transport=transport, headers=headers
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice

        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"could not reach the provider: {type(exc).__name__}") from exc
                self._sleep(2**attempt)
                continue
            if response.status_code in RETRYABLE and attempt < self.max_retries:
                self._sleep(self._backoff(response, attempt))
                continue
            if response.status_code >= 400:
                raise LLMError(f"provider returned {response.status_code}: {_reason(response)}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise LLMError("provider returned a body that is not JSON") from exc
            failure = _body_error(payload)
            if failure is not None:
                # OpenRouter can report an upstream failure inside a 200 response.
                status, message = failure
                if status in RETRYABLE and attempt < self.max_retries:
                    self._sleep(float(2**attempt))
                    continue
                raise LLMError(f"provider returned {status}: {message}")
            return _parse(payload)
        raise LLMError("retries exhausted")  # pragma: no cover — loop always returns or raises

    @staticmethod
    def _backoff(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            return min(float(retry_after), 60.0)
        return float(2**attempt)


def _body_error(payload: Any) -> tuple[int, str] | None:
    """An error object in a 200 body with no choices, as (status, message)."""
    if not isinstance(payload, dict) or payload.get("choices"):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    status = code if isinstance(code, int) and 400 <= code < 600 else 502
    return status, _describe_error(error) or "unknown upstream error"


def _describe_error(error: dict[str, Any]) -> str:
    message = str(error.get("message", ""))
    metadata = error.get("metadata") or {}
    upstream = metadata.get("provider_name")
    raw = " ".join(str(metadata.get("raw", "")).split())
    if upstream or raw:
        message += f" [{upstream or 'upstream'}: {raw[:300] or 'no detail'}]"
    return message[:400]


def _reason(response: httpx.Response) -> str:
    """The provider's message, plus the upstream detail OpenRouter nests in metadata.

    OpenRouter wraps upstream failures as "Provider returned error"; the actual
    cause (a rate limit, an unsupported parameter) is in `metadata.raw`.
    """
    try:
        error = response.json().get("error", {})
        return _describe_error(error) if error.get("message") else response.reason_phrase
    except (ValueError, AttributeError):
        return response.reason_phrase


def _parse(payload: dict[str, Any]) -> ChatResponse:
    try:
        choice = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected response shape: {json.dumps(payload)[:200]}") from exc
    calls = choice.get("tool_calls") or []
    arguments = calls[0].get("function", {}).get("arguments") if calls else None
    return ChatResponse(
        content=choice.get("content"),
        tool_arguments=arguments,
        model=str(payload.get("model", "")),
        usage={k: int(v) for k, v in (payload.get("usage") or {}).items() if isinstance(v, int)},
    )


FREE_ROUTER = "openrouter/free"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int


def free_tool_models(
    base_url: str = "https://openrouter.ai/api/v1", transport: httpx.BaseTransport | None = None
) -> list[ModelInfo]:
    """Free models that accept tool calls, from the provider's live catalogue.

    The catalogue is public — no key needed. Free availability changes often
    enough that a hard-coded list would be wrong within days; this reads it
    at the moment of asking.
    """
    with httpx.Client(base_url=base_url, timeout=30.0, transport=transport) as http:
        response = http.get("/models")
    if response.status_code >= 400:
        raise LLMError(f"model catalogue returned {response.status_code}")
    models = []
    for item in response.json().get("data", []):
        pricing = item.get("pricing") or {}
        free = str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
        if free and "tools" in (item.get("supported_parameters") or []):
            models.append(
                ModelInfo(id=str(item["id"]), context_length=int(item.get("context_length") or 0))
            )
    return sorted(models, key=lambda m: m.id)
