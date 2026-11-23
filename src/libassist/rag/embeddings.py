"""Embedders: turn text into vectors so that similar meanings end up close together.

Two implementations behind one small interface (`Embedder`):

* HashingEmbedder (default): no model, no network, fully deterministic. It uses *feature hashing*: each
  word and each character trigram is hashed into one of `dim` buckets. Texts that share words or word
  pieces get similar vectors. It catches morphology and typos ("renewl" ~ "renewal"), not true synonyms.
  It exists so that the whole pipeline, and its tests, run offline.

* OpenAICompatibleEmbedder: calls any OpenAI-compatible POST /embeddings endpoint (OpenAI, Ollama with
  nomic-embed-text, vLLM, ...). Use this to get real semantic search. Enabled with EMBEDDINGS_BASE_URL.

All vectors are L2-normalized, so cosine similarity is simply a dot product.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Protocol

import httpx

from libassist.rag.text import tokenize, words


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class HashingEmbedder:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        for token in tokenize(text):
            yield f"w:{token}", 1.0
        for word in words(text):
            padded = f"<{word}>"
            for i in range(len(padded) - 2):
                yield f"c:{padded[i : i + 3]}", 0.35

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, weight in self._features(text):
            digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if digest & 1 else -1.0  # a random sign keeps hash collisions from always adding up
            vector[(digest >> 1) % self.dim] += sign * weight
        return normalize(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai-compatible:{model}"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [normalize(item["embedding"]) for item in data]


def make_embedder(base_url: str | None, model: str, api_key: str | None) -> Embedder:
    if base_url:
        return OpenAICompatibleEmbedder(base_url, model, api_key)
    return HashingEmbedder()
