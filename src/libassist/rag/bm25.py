from __future__ import annotations

import bm25s


# Okapi BM25
class BM25:
    def __init__(self, documents: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.n_docs = len(documents)
        self._index = bm25s.BM25(k1=k1, b=b, method="lucene")
        if documents:
            self._index.index(documents, show_progress=False)

    def scores(self, query: list[str]) -> list[float]:
        terms = list(dict.fromkeys(query))  # to avoid counting query terms more than once
        if not self.n_docs or not terms:  # fail on empty
            return [0.0] * self.n_docs
        return (self._index.get_scores(terms) * (self.k1 + 1)).tolist()
