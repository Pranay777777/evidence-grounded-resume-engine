"""Model adapters: one spec string chooses the provider and the model (ADR-012).

    openrouter/free                    OpenRouter (the default provider)
    google/gemma-4-31b-it:free         OpenRouter, pinned
    ollama:<model>                     a local model through Ollama: no key, nothing leaves
                                       the machine
    openai:<model>                     OpenAI directly

Every provider here speaks the OpenAI chat-completions protocol, so one
client serves them all; an adapter is the endpoint, the credential and the
headers. That keeps a provider swap - forced by a deprecation date, a price
change or a free tier ending - to a config line, and it keeps the grounding
gate identical whichever model wrote the draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from grounded.config import Settings
from grounded.generation.llm import OpenAICompatibleClient


@dataclass(frozen=True)
class Provider:
    name: str
    key_setting: str | None
    """The Settings field holding the key; None = no key (a local server)."""
    key_hint: str = ""
    attribution: bool = False


PROVIDERS: dict[str, Provider] = {
    "openrouter": Provider(
        "openrouter",
        "openrouter_api_key",
        "OPENROUTER_API_KEY (a free key from openrouter.ai/keys)",
        attribution=True,
    ),
    "openai": Provider("openai", "openai_api_key", "OPENAI_API_KEY"),
    "ollama": Provider("ollama", None),
}
BASE_URLS = {"openai": "https://api.openai.com/v1"}


def parse(spec: str) -> tuple[Provider, str]:
    """Split `provider:model`. OpenRouter ids contain ':' too (`...:free`), so a
    prefix only counts when it names a known provider."""
    prefix, sep, rest = spec.partition(":")
    if sep and prefix in PROVIDERS and "/" not in prefix:
        if not rest:
            raise ValueError(f"'{spec}' names a provider but no model")
        return PROVIDERS[prefix], rest
    return PROVIDERS["openrouter"], spec


def base_url(provider: Provider, settings: Settings) -> str:
    if provider.name == "openrouter":
        return settings.llm_base_url
    if provider.name == "ollama":
        return settings.ollama_base_url
    return BASE_URLS[provider.name]


LOCAL_PROVIDERS = frozenset({"ollama"})


def is_local(spec: str) -> bool:
    return parse(spec)[0].name in LOCAL_PROVIDERS


class LocalOnlyError(ValueError):
    pass


def make_client(spec: str, settings: Settings, **kwargs: Any) -> OpenAICompatibleClient:
    provider, model = parse(spec)
    if settings.local_only and provider.name not in LOCAL_PROVIDERS:
        raise LocalOnlyError(
            f"LOCAL_ONLY is set: '{spec}' would send evidence off this machine - "
            "use an ollama:<model> spec"
        )
    key = getattr(settings, provider.key_setting) if provider.key_setting else SecretStr("")
    return OpenAICompatibleClient(
        key,
        model,
        base_url(provider, settings),
        require_key=provider.key_setting is not None,
        key_hint=provider.key_hint,
        attribution=provider.attribution,
        max_tokens=settings.llm_max_tokens,
        **kwargs,
    )
