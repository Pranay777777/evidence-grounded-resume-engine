"""Cross-encoder reranking — the precision stage after recall (ADR-005).

Retrieval scores the query and each record *separately* (BM25 term weights,
two independent embeddings). A cross-encoder reads the query and a record
*together*, so it can tell "built dashboards" from "consumed dashboards" —
far more accurate, and far too slow to run over the whole store. So it runs
only on the fused top candidates.

`ms-marco-MiniLM-L-6-v2` via fastembed: 80 MB, Apache-2.0, ONNX, CPU. Part of
the `[embeddings]` extra; asking for it without the extra is an error, never
a silent skip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

RERANKERS = ("minilm",)


class Reranker(Protocol):
    name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


_HINT = 'rerankers need the embeddings extra — install it with: pip install -e ".[embeddings]"'


class CrossEncoderReranker:
    name = "minilm"
    model_name = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model: Any | None = None) -> None:
        if model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise ImportError(_HINT) from exc
            from grounded.config import get_settings

            model = TextCrossEncoder(self.model_name, cache_dir=get_settings().model_cache_dir)
        self._model = model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(documents))]


def get_reranker(name: str) -> Reranker:
    if name == "minilm":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker '{name}' — choose one of: {', '.join(RERANKERS)}")
