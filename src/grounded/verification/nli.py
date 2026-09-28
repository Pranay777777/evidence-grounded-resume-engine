"""The entailment verifier: does the cited evidence actually say this?

A natural-language-inference model reads a premise (the cited statements)
and a hypothesis (the bullet) and classifies the pair as entailment,
neutral or contradiction. Only entailment above a threshold keeps a bullet
(ADR-009). "Neutral" is the important case: an embellishment is rarely
contradicted by the evidence — it is simply not supported by it.

`Xenova/nli-deberta-v3-xsmall`: DeBERTa-v3 fine-tuned on SNLI and MultiNLI,
exported to ONNX, run with onnxruntime and a Hugging Face tokenizer — no
PyTorch. The label order is read from the model's own config rather than
assumed, because NLI checkpoints disagree about it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class Label(StrEnum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class Verdict:
    label: Label
    entailment: float
    """Probability of entailment, from 0 to 1."""


class Verifier(Protocol):
    name: str

    def check(self, premise: str, hypothesis: str) -> Verdict: ...


VERIFIERS = ("nli-deberta",)
_HINT = 'the verifier needs the embeddings extra — install it with: pip install -e ".[embeddings]"'


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def label_order(config: dict[str, Any]) -> list[Label]:
    """Map output indices to labels using the checkpoint's own id2label."""
    id2label = config.get("id2label") or {}
    order = []
    for index in range(len(id2label)):
        name = str(id2label.get(str(index), id2label.get(index, ""))).lower()
        try:
            order.append(Label(name))
        except ValueError as exc:
            raise ValueError(f"unexpected NLI label '{name}' at index {index}") from exc
    if set(order) != set(Label):
        raise ValueError(f"model labels {order} are not entailment/neutral/contradiction")
    return order


class NLIVerifier:
    name = "nli-deberta"
    repo = "Xenova/nli-deberta-v3-xsmall"
    onnx_file = "onnx/model_quantized.onnx"
    max_tokens = 512

    def __init__(
        self,
        cache_dir: str | None = None,
        session: Any = None,
        tokenizer: Any = None,
        labels: list[Label] | None = None,
        to_array: Callable[[list[int]], Any] | None = None,
    ) -> None:
        if session is None or tokenizer is None or labels is None:
            session, tokenizer, labels = self._load(cache_dir)
        self._session, self._tokenizer, self._labels = session, tokenizer, labels
        self._to_array = to_array or _int64_batch
        self._inputs = {i.name for i in self._session.get_inputs()}

    def _load(self, cache_dir: str | None) -> tuple[Any, Any, list[Label]]:
        try:
            import onnxruntime
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(_HINT) from exc

        def fetch(name: str) -> str:
            return str(hf_hub_download(self.repo, name, cache_dir=cache_dir))

        config = json.loads(Path(fetch("config.json")).read_text(encoding="utf-8"))
        tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
        tokenizer.enable_truncation(self.max_tokens)
        session = onnxruntime.InferenceSession(
            fetch(self.onnx_file), providers=["CPUExecutionProvider"]
        )
        return session, tokenizer, label_order(config)

    def check(self, premise: str, hypothesis: str) -> Verdict:
        encoding = self._tokenizer.encode(premise, hypothesis)
        feed: dict[str, Any] = {}
        if "input_ids" in self._inputs:
            feed["input_ids"] = self._to_array(encoding.ids)
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = self._to_array(encoding.attention_mask)
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = self._to_array(encoding.type_ids)
        logits = self._session.run(None, feed)[0][0]
        probabilities = dict(zip(self._labels, _softmax([float(x) for x in logits]), strict=True))
        best = max(probabilities, key=lambda label: probabilities[label])
        return Verdict(label=best, entailment=probabilities[Label.ENTAILMENT])


def _int64_batch(values: list[int]) -> Any:
    """A batch of one, as the int64 array ONNX Runtime expects. numpy arrives
    with the embeddings extra, so it is imported only when a real model runs."""
    import numpy as np

    return np.array([values], dtype=np.int64)


def get_verifier(name: str, cache_dir: str | None = None) -> Verifier:
    if name == "nli-deberta":
        return NLIVerifier(cache_dir=cache_dir)
    raise ValueError(f"unknown verifier '{name}' — choose one of: {', '.join(VERIFIERS)}")
