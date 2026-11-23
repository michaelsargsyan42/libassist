"""Retrieval-augmented generation (RAG), the retrieval half.

The pipeline is: documents -> chunks (chunker.py) -> two indexes over the same chunks:
  * a lexical index, BM25 (bm25.py), good at exact terms such as "renewal" or "eduroam";
  * a vector index (embeddings.py), good at paraphrases such as "extend my book" -> renewals.
The two rankings are merged with Reciprocal Rank Fusion (index.py). Every chunk has a stable id
("loans#renewals") that the assistant cites, so answers can be checked against the source.

Everything here is plain Python with no ML dependencies, so the tests run anywhere. A real embedding model
can be plugged in through any OpenAI-compatible /embeddings endpoint (see embeddings.py).
"""
