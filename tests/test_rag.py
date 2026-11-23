from __future__ import annotations

import math
import re

import pytest

from libassist.config import REPO_ROOT
from libassist.domain import policy
from libassist.domain.models import MemberType
from libassist.rag.bm25 import BM25
from libassist.rag.chunker import Chunk, chunk_markdown, load_policy_chunks
from libassist.rag.embeddings import HashingEmbedder, dot
from libassist.rag.index import HybridIndex, build_policy_index
from libassist.rag.text import tokenize

POLICIES = REPO_ROOT / "data" / "policies"


@pytest.fixture(scope="module")
def policy_index() -> HybridIndex:
    return build_policy_index(POLICIES, HashingEmbedder())


# ------------------------------------------------------------------ building blocks


def test_chunks_are_sections_with_citation_ids():
    chunks = chunk_markdown("loans", "# Loans\n\nintro\n\n## Loan periods\n\nText A\n\n## Renewals\n\nText B\n")
    assert [(c.id, c.title, c.text) for c in chunks] == [
        ("loans#loan-periods", "Loans > Loan periods", "Text A"),
        ("loans#renewals", "Loans > Renewals", "Text B"),
    ]


def test_duplicate_headings_get_distinct_ids():
    chunks = chunk_markdown("faq", "## Question\n\na\n\n## Question\n\nb\n")
    assert [c.id for c in chunks] == ["faq#question", "faq#question-2"]


def test_all_policy_sections_are_indexed():
    ids = [c.id for c in load_policy_chunks(POLICIES)]
    assert len(ids) == len(set(ids)) == 18
    assert "loans#renewals" in ids and "services#wi-fi" in ids


def test_tokenizer_folds_accents_stems_and_joins_compounds():
    assert tokenize("Réservations renewed") == ["reserv", "renew"]
    assert "wifi" in tokenize("Wi-Fi") and "wifi" in tokenize("wifi")
    assert tokenize("the and of") == []


def test_bm25_prefers_documents_with_rare_query_terms():
    docs = [tokenize("renewals are allowed twice"), tokenize("books books books"), tokenize("eduroam wifi network")]
    scores = BM25(docs).scores(tokenize("eduroam"))
    assert scores[2] > 0 and scores[0] == scores[1] == 0


def test_hashing_embedder_is_deterministic_normalized_and_similarity_aware():
    embedder = HashingEmbedder()
    a, b, c = embedder.embed(["renew my loan", "renewing loans", "opening hours on saturday"])
    assert embedder.embed(["renew my loan"])[0] == a
    assert math.isclose(dot(a, a), 1.0, rel_tol=1e-9)
    assert dot(a, b) > dot(a, c)


def test_fusion_ranks_a_document_found_by_both_retrievers_first():
    chunks = [
        Chunk("d1", "Renewals", "renew a loan up to two times", "t"),
        Chunk("d2", "Opening hours", "the library opens at eight", "t"),
        Chunk("d3", "Loans", "loan periods are 21 days", "t"),
    ]
    hits = HybridIndex(chunks, HashingEmbedder(), min_bm25=0.1, min_cosine=0.1).search("renew loan")
    assert hits[0].chunk.id == "d1"
    assert hits[0].bm25_rank == 1 and hits[0].vector_rank == 1


# ------------------------------------------------------------------ retrieval quality

LABELLED_QUESTIONS = {
    "How many times can I renew a book?": "loans#renewals",
    "can i extend my loan": "loans#renewals",
    "How long can staff keep a book?": "loans#loan-periods",
    "where do I return books at night": "loans#returning-books",
    "What happens if I don't pick up my reservation?": "holds#pickup-window",
    "how many holds can I have": "holds#placing-a-hold",
    "how do I cancel my reservation": "holds#cancelling-a-hold",
    "how much is the late fee per day": "fines#overdue-fines",
    "Why can't I renew, my account says blocked": "fines#blocked-accounts",
    "I lost a book, what do I pay?": "fines#lost-or-damaged-items",
    "who can see my borrowing history": "accounts#privacy",
    "change my email address": "accounts#updating-contact-details",
    "when does the library open on saturday": "services#opening-hours",
    "Can I book a study room through you?": "services#study-rooms",
    "is there wifi": "services#wi-fi",
}


def test_recall_at_3_on_labelled_questions(policy_index):
    misses = [
        q
        for q, expected in LABELLED_QUESTIONS.items()
        if expected not in [h.chunk.id for h in policy_index.search(q, k=3)]
    ]
    recall = 1 - len(misses) / len(LABELLED_QUESTIONS)
    assert recall >= 0.9, f"recall@3 = {recall:.0%}, missed: {misses}"


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "recipe for lasagna",
        "quantum chromodynamics lattice simulation",
        "Can I bring my dog?",
    ],
)
def test_unrelated_questions_retrieve_nothing(policy_index, question):
    assert policy_index.search(question) == []


def test_catalogue_search_by_topic(app):
    index = app.state.catalog_index  # built by create_app from the seeded books
    assert index.search("java best practices", k=1)[0].chunk.id == "B-007"
    assert "B-011" in [h.chunk.id for h in index.search("sql transactions and indexing", k=3)]


# ------------------------------------------------------------------ documents and code agree


def test_policy_documents_match_the_code():
    """The agent explains rules from the documents; the code enforces them. They must say the same thing."""
    text = " ".join(p.read_text(encoding="utf-8") for p in POLICIES.glob("*.md"))

    def number(pattern: str) -> float:
        match = re.search(pattern, text)
        assert match, f"pattern not found in policies: {pattern}"
        return float(match.group(1))

    assert number(r"Students can borrow books for (\d+) days") == policy.LOAN_DAYS[MemberType.STUDENT]
    assert number(r"Staff can borrow books for (\d+) days") == policy.LOAN_DAYS[MemberType.STAFF]
    assert number(r"maximum of (\d+) renewals per loan") == policy.MAX_RENEWALS
    assert number(r"more than (\d+) days overdue") == policy.MAX_OVERDUE_DAYS_FOR_RENEWAL
    assert number(r"at most (\d+) active holds") == policy.MAX_ACTIVE_HOLDS
    assert number(r"you have (\d+) days to collect") == policy.PICKUP_WINDOW_DAYS
    assert number(r"€(\d+\.\d+) per day per item") * 100 == policy.FINE_PER_DAY_CENTS
    assert number(r"stops growing once it reaches €(\d+)") * 100 == policy.FINE_CAP_PER_ITEM_CENTS
    assert number(r"unpaid fines reach €(\d+) or more") * 100 == policy.BLOCK_THRESHOLD_CENTS
