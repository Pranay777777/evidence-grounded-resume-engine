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
    ) -> None:
        if not api_key.get_secret_value():
            raise LLMError(
                "no API key — set OPENROUTER_API_KEY (a free key from openrouter.ai/keys)"
            )
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                # Optional attribution headers OpenRouter documents.
                "HTTP-Referer": "https://github.com/Pranay777777/evidence-grounded-resume-engine",
                "X-Title": "evidence-grounded-resume-engine",
            },
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
            return _parse(response.json())
        raise LLMError("retries exhausted")  # pragma: no cover — loop always returns or raises

    @staticmethod
    def _backoff(response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            return min(float(retry_after), 60.0)
        return float(2**attempt)


def _reason(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        return str(error.get("message", ""))[:200] or response.reason_phrase
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
