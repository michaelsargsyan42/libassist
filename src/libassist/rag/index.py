from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from libassist.domain.models import Book
from libassist.rag.bm25 import BM25
from libassist.rag.chunker import Chunk, load_policy_chunks
from libassist.rag.embeddings import Embedder, dot
from libassist.rag.text import tokenize

RRF_K = 60


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # used RRF score for ordering
    bm25_rank: int | None
    vector_rank: int | None
    cosine: float


class HybridIndex:
    def __init__(
        self, chunks: list[Chunk], embedder: Embedder, *, min_cosine: float = 0.25, min_bm25: float = 1.0
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.min_cosine = min_cosine
        self.min_bm25 = min_bm25
        texts = [c.indexed_text for c in chunks]
        self._bm25 = BM25([tokenize(t) for t in texts])
        self._vectors = embedder.embed(texts)

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        if not query.strip() or not self.chunks:
            return []
        n = len(self.chunks)
        lexical = self._bm25.scores(tokenize(query))
        query_vector = self.embedder.embed([query])[0]
        cosines = [dot(query_vector, v) for v in self._vectors]

        lexical_ranking = [i for i in sorted(range(n), key=lambda i: -lexical[i]) if lexical[i] >= self.min_bm25]
        vector_ranking = [i for i in sorted(range(n), key=lambda i: -cosines[i]) if cosines[i] >= self.min_cosine]

        fused: dict[int, float] = defaultdict(float)
        bm25_rank: dict[int, int] = {}
        vector_rank: dict[int, int] = {}
        for rank, i in enumerate(lexical_ranking, start=1):
            fused[i] += 1.0 / (RRF_K + rank)
            bm25_rank[i] = rank
        for rank, i in enumerate(vector_ranking, start=1):
            fused[i] += 1.0 / (RRF_K + rank)
            vector_rank[i] = rank

        best = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        return [SearchHit(self.chunks[i], fused[i], bm25_rank.get(i), vector_rank.get(i), cosines[i]) for i in best]


def build_policy_index(policies_dir: Path, embedder: Embedder) -> HybridIndex:
    return HybridIndex(load_policy_chunks(policies_dir), embedder)


def build_catalog_index(books: list[Book], embedder: Embedder) -> HybridIndex:
    """Index the *static* text of each book. Availability changes constantly, so it is never indexed:
    the API reads it from the database at query time."""
    chunks = [
        Chunk(
            id=b.id,
            title=f"{b.title} by {b.author}",
            text=f"Subjects: {', '.join(b.subjects)}. {b.description}",
            source="catalog",
        )
        for b in books
    ]
    return HybridIndex(chunks, embedder, min_bm25=0.5)
