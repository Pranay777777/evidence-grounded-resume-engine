"""Generate a structured draft, retrying only when the output does not parse.

ADR-001 draws the line this module enforces: a response that is not valid
JSON, or does not match the schema, may be sent back with the validation
errors and regenerated — that is a *structural* failure. A response that
parses but cites the wrong evidence is a *grounding* failure, and it is never
retried: steps 48 and 49 drop the offending bullets instead. Retrying would teach
the model to phrase claims until they pass, which is not the same as true.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import ValidationError

from grounded.evidence.models import Evidence
from grounded.generation import prompt
from grounded.generation.llm import ChatResponse
from grounded.generation.schema import Draft
from grounded.retrieval.search import MAX_QUERY_CHARS

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


class ChatClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = ...,
        tool_choice: dict[str, Any] | None = ...,
        temperature: float = ...,
    ) -> ChatResponse: ...


class GenerationError(RuntimeError):
    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


@dataclass
class Generation:
    draft: Draft
    attempts: int
    model: str
    prompt_version: str = prompt.PROMPT_VERSION
    errors: list[str] = field(default_factory=list)
    """Validation errors from failed attempts, oldest first."""
    usage: dict[str, int] = field(default_factory=dict)
    """Token counts summed over every attempt, as the provider reported them."""

    @property
    def prompt_fingerprint(self) -> str:
        return prompt.get(self.prompt_version).fingerprint


def _payload(response: ChatResponse) -> str:
    raw = response.tool_arguments or response.content or ""
    fenced = _FENCE.match(raw)
    return fenced.group(1) if fenced else raw.strip()


def _describe(exc: ValidationError | ValueError) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()
        )
    return str(exc)


def generate(
    client: ChatClient,
    job_description: str,
    records: Sequence[Evidence],
    max_attempts: int = 3,
    prompt_version: str = prompt.PROMPT_VERSION,
) -> Generation:
    """Ask for a draft; on a structural failure, show the model its errors and retry."""
    if not records:
        raise GenerationError("no citable evidence to generate from", [])
    conversation = prompt.messages(job_description, records, MAX_QUERY_CHARS, prompt_version)
    errors: list[str] = []
    usage: dict[str, int] = {}
    for attempt in range(1, max_attempts + 1):
        response = client.complete(
            conversation, tools=[prompt.draft_tool()], tool_choice=prompt.FORCE_TOOL
        )
        for key, value in response.usage.items():
            usage[key] = usage.get(key, 0) + value
        raw = _payload(response)
        try:
            draft = Draft.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            problem = (
                _describe(exc) if isinstance(exc, ValidationError) else f"not valid JSON: {exc}"
            )
            errors.append(f"attempt {attempt}: {problem}")
            conversation = [
                *conversation,
                {"role": "assistant", "content": raw[:4000]},
                {
                    "role": "user",
                    "content": "That output was rejected because it is not valid for the "
                    f"emit_draft schema: {problem}. Call emit_draft again with a corrected, "
                    "complete response. Keep the same rules.",
                },
            ]
            continue
        return Generation(
            draft=draft,
            attempts=attempt,
            model=response.model or client.model,
            prompt_version=prompt_version,
            errors=errors,
            usage=usage,
        )
    raise GenerationError(
        f"no valid draft after {max_attempts} attempts: output never matched the schema",
        errors,
    )
