"""Embedders: one real model, one dependency-free baseline, one interface.

- **`bge-small`** — `BAAI/bge-small-en-v1.5` through `fastembed` (ONNX, no
  PyTorch, 67 MB, MIT). The dense model. An optional extra:
  `pip install -e ".[embeddings]"`.
- **`hashing`** — signed feature hashing of word unigrams and bigrams into
  the same 384 dimensions. No model, no download, fully deterministic. It
  is what tests and CI use, and it is reported in the retrieval ablation as
  what it is: a lexical baseline dressed as a vector, not a semantic model.

Asking for `bge-small` without the extra is an error, never a quiet switch
to `hashing` — a silent fallback would make every retrieval number reported
afterwards a lie about which model produced it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Protocol

DIM = 384
"""Both embedders produce this, so one pgvector column and index serve both."""

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*[+#]*")
"""Words, dotted names (`node.js`, `3.12`) and trailing `+`/`#` (`c++`, `c#`).
A tokeniser that splits `c++` into `c` makes BM25 blind to exactly the
technical terms job descriptions are made of."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def tokens(text: str) -> list[str]:
    """Lowercased word tokens; keeps `c++`, `c#`, `node.js`, `3.12` intact."""
    return _TOKEN.findall(text.lower())


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashingEmbedder:
    """Deterministic feature hashing. A baseline, labelled as one."""

    name = "hashing"
    dim = DIM

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            words = tokens(text)
            features = words + [f"{a} {b}" for a, b in pairwise(words)]
            vector = [0.0] * self.dim
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            out.append(_normalise(vector))
        return out


_INSTALL_HINT = (
    "the 'bge-small' embedder needs the embeddings extra — "
    'install it with: pip install -e ".[embeddings]"'
)


class FastEmbedEmbedder:
    """`BAAI/bge-small-en-v1.5` via fastembed. Downloads the model on first use."""

    name = "bge-small"
    dim = DIM
    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, model: Any | None = None) -> None:
        if model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
            from grounded.config import get_settings

            model = TextEmbedding(self.model_name, cache_dir=get_settings().model_cache_dir)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.embed(list(texts))]


EMBEDDERS = ("bge-small", "hashing")


def get_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "bge-small":
        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder '{name}' — choose one of: {', '.join(EMBEDDERS)}")
