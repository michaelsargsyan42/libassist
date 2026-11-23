"""REST API tests through FastAPI's TestClient (in-process, no network).

Seed data reminder (business date 2026-03-02):
  M-1001 Léa Martin (student): L-5001 renewable, L-5002 at the renewal limit, L-5006 3 days overdue (€0.75)
  M-1002 Karim Dubois (staff): L-5004 Effective Java, blocked by Sofia's hold H-7001
  M-1003 Sofia Rossi: H-7001 queued for B-007, H-7002 ready for B-002
  M-1004 Tom Novak: L-5005 45 days overdue -> €10 fine, account blocked
"""

from __future__ import annotations

import pytest

from tests.conftest import as_member

PROBLEM = "application/problem+json"


def assert_problem(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM
    body = response.json()
    assert body["status"] == status and body["code"] == code
    assert body["type"].endswith("/" + code) and body["detail"] and body["instance"]
    return body


# ------------------------------------------------------------------ public endpoints


def test_health_and_openapi(client):
    assert client.get("/v1/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/loans/{loan_id}/renewals", "/v1/holds", "/v1/policies/search"} <= set(paths)


def test_request_id_is_echoed_or_generated(client):
    assert client.get("/v1/health", headers={"X-Request-Id": "abc-123"}).headers["X-Request-Id"] == "abc-123"
    assert len(client.get("/v1/health").headers["X-Request-Id"]) == 32


def test_catalogue_search_returns_live_availability(client):
    results = client.get("/v1/books", params={"q": "python programming"}).json()["results"]
    assert results[0]["id"] == "B-009"
    assert results[0]["available_copies"] == 2


def test_catalogue_without_query_lists_books(client):
    results = client.get("/v1/books", params={"limit": 50}).json()["results"]
    assert len(results) == 13
    effective_java = next(b for b in results if b["id"] == "B-007")
    assert effective_java["available_copies"] == 0 and effective_java["queue_length"] == 1


def test_unknown_book_is_a_problem_document(client):
    assert_problem(client.get("/v1/books/B-999"), 404, "not-found")


def test_malformed_id_is_a_validation_problem(client):
    body = assert_problem(client.get("/v1/books/not-an-id"), 422, "validation-error")
    assert body["errors"][0]["loc"] == ["path", "book_id"]


def test_policy_search_returns_passages_with_citations(client):
    results = client.get("/v1/policies/search", params={"q": "How many times can I renew a book?"}).json()["results"]
    assert results[0]["citation"] == "loans#renewals"
    assert "maximum of 2 renewals" in results[0]["text"]


def test_policy_search_returns_nothing_for_unrelated_questions(client):
    assert client.get("/v1/policies/search", params={"q": "What is the capital of France?"}).json()["results"] == []


def test_policy_documents(client):
    assert "## Renewals" in client.get("/v1/policies/loans").json()["markdown"]
    assert_problem(client.get("/v1/policies/nope"), 404, "not-found")
    assert client.get("/v1/policies/..%2Fseed").status_code in (404, 422)  # no path traversal


@pytest.mark.parametrize("last_name", ["Martin", "MARTIN", "  martin ", "Martín"])
def test_verification_ignores_case_accents_and_spaces(client, last_name):
    body = client.post("/v1/members/verify", json={"member_id": "M-1001", "last_name": last_name}).json()
    assert body == {"verified": True, "member_id": "M-1001", "first_name": "Léa"}


def test_failed_verification_does_not_reveal_whether_the_member_exists(client):
    wrong_name = client.post("/v1/members/verify", json={"member_id": "M-1001", "last_name": "Smith"}).json()
    unknown_id = client.post("/v1/members/verify", json={"member_id": "M-9999", "last_name": "Martin"}).json()
    assert wrong_name == unknown_id == {"verified": False, "member_id": None, "first_name": None}


# ------------------------------------------------------------------ authentication and authorization


def test_member_endpoints_require_a_caller(client):
    assert_problem(client.get("/v1/members/M-1001/account"), 401, "unauthenticated")
    assert_problem(client.get("/v1/members/M-1001/account", headers=as_member("M-9999")), 401, "unauthenticated")


def test_other_members_records_look_like_they_do_not_exist(client):
    me = as_member("M-1001")
    assert_problem(client.get("/v1/members/M-1003/account", headers=me), 404, "not-found")
    assert_problem(client.post("/v1/loans/L-5003/renewals", headers=me), 404, "not-found")
    assert_problem(client.get("/v1/holds/H-7001", headers=me), 404, "not-found")
    assert_problem(client.delete("/v1/holds/H-7001", headers=me), 404, "not-found")


# ------------------------------------------------------------------ account


def test_account_shows_fines_and_renewal_eligibility(client):
    account = client.get("/v1/members/M-1001/account", headers=as_member("M-1001")).json()
    assert account["fines_total_cents"] == 75 and account["fines_total"] == "€0.75"
    assert account["blocked"] is False
    loans = {loan["id"]: loan for loan in account["loans"]}
    assert set(loans) == {"L-5001", "L-5002", "L-5006"}
    assert loans["L-5001"]["renewal"] == {"allowed": True, "reason": None, "detail": None, "new_due_on": "2026-03-28"}
    assert loans["L-5002"]["renewal"]["reason"] == "renewal-limit"
    assert loans["L-5006"]["overdue_days"] == 3 and loans["L-5006"]["fine"] == "€0.75"


def test_blocked_account(client):
    account = client.get("/v1/members/M-1004/account", headers=as_member("M-1004")).json()
    assert account["fines_total_cents"] == 1000 and account["blocked"] is True
    assert account["loans"][0]["renewal"]["reason"] == "account-blocked"


def test_queue_position_is_shown(client):
    holds = client.get("/v1/members/M-1003/account", headers=as_member("M-1003")).json()["holds"]
    assert [(h["id"], h["status"], h["queue_position"]) for h in holds] == [
        ("H-7001", "QUEUED", 1),
        ("H-7002", "READY_FOR_PICKUP", None),
    ]


# ------------------------------------------------------------------ renewals


def test_renewal_extends_the_due_date(client):
    response = client.post("/v1/loans/L-5001/renewals", headers=as_member("M-1001"))
    assert response.status_code == 200
    body = response.json()
    assert body["previous_due_on"] == "2026-03-07"
    assert body["loan"]["due_on"] == "2026-03-28" and body["loan"]["renewal_count"] == 1


@pytest.mark.parametrize(
    ("member", "loan", "code"),
    [
        ("M-1001", "L-5002", "renewal-limit"),
        ("M-1002", "L-5004", "hold-queue"),
        ("M-1004", "L-5005", "account-blocked"),
    ],
)
def test_renewal_refused_by_business_rules(client, db, member, loan, code):
    before = db.execute("SELECT due_on, renewal_count FROM loans WHERE id = ?", (loan,)).fetchone()
    body = assert_problem(client.post(f"/v1/loans/{loan}/renewals", headers=as_member(member)), 409, code)
    assert body["title"] == "Renewal refused" and body["loan_id"] == loan
    after = db.execute("SELECT due_on, renewal_count FROM loans WHERE id = ?", (loan,)).fetchone()
    assert tuple(before) == tuple(after)  # nothing changed


def test_renewing_an_overdue_loan_keeps_the_fine(client):
    me = as_member("M-1001")
    renewed = client.post("/v1/loans/L-5006/renewals", headers=me).json()["loan"]
    assert renewed["due_on"] == "2026-03-20"  # 2026-02-27 + 21 days
    assert renewed["overdue_days"] == 0 and renewed["fine_cents"] == 75
    assert client.get("/v1/members/M-1001/account", headers=me).json()["fines_total_cents"] == 75


# ------------------------------------------------------------------ idempotency


def test_retry_with_the_same_idempotency_key_is_not_applied_twice(client, db):
    headers = as_member("M-1001", **{"Idempotency-Key": "renew-L-5001-attempt"})
    first = client.post("/v1/loans/L-5001/renewals", headers=headers)
    retry = client.post("/v1/loans/L-5001/renewals", headers=headers)
    assert first.status_code == retry.status_code == 200
    assert retry.json() == first.json()
    assert "Idempotent-Replayed" not in first.headers and retry.headers["Idempotent-Replayed"] == "true"
    assert db.execute("SELECT renewal_count FROM loans WHERE id = 'L-5001'").fetchone()[0] == 1


def test_same_key_with_a_different_request_is_rejected(client):
    headers = as_member("M-1001", **{"Idempotency-Key": "hold-key-1"})
    assert client.post("/v1/holds", json={"book_id": "B-006"}, headers=headers).status_code == 201
    assert_problem(client.post("/v1/holds", json={"book_id": "B-009"}, headers=headers), 422, "idempotency-key-reused")


def test_idempotency_keys_are_scoped_to_the_caller(client):
    key = {"Idempotency-Key": "same-key-for-both"}
    first = client.post("/v1/holds", json={"book_id": "B-006"}, headers=as_member("M-1001", **key))
    second = client.post("/v1/holds", json={"book_id": "B-006"}, headers=as_member("M-1002", **key))
    assert (first.status_code, second.status_code) == (201, 201)
    assert first.json()["id"] != second.json()["id"]


def test_a_failed_request_does_not_consume_the_key(client, db):
    headers = as_member("M-1001", **{"Idempotency-Key": "will-fail-1"})
    assert_problem(client.post("/v1/loans/L-5002/renewals", headers=headers), 409, "renewal-limit")
    assert db.execute("SELECT COUNT(*) FROM idempotency_keys").fetchone()[0] == 0


# ------------------------------------------------------------------ holds


def test_hold_is_ready_when_a_copy_is_on_the_shelf(client):
    response = client.post("/v1/holds", json={"book_id": "B-006"}, headers=as_member("M-1001"))
    assert response.status_code == 201
    hold = response.json()
    assert response.headers["Location"] == f"/v1/holds/{hold['id']}"
    assert hold["status"] == "READY_FOR_PICKUP" and hold["ready_until"] == "2026-03-07"
    assert client.get("/v1/books/B-006").json()["available_copies"] == 0


def test_hold_joins_the_queue_when_no_copy_is_left(client):
    hold = client.post("/v1/holds", json={"book_id": "B-007"}, headers=as_member("M-1001")).json()
    assert hold["status"] == "QUEUED" and hold["queue_position"] == 2  # behind Sofia's H-7001


@pytest.mark.parametrize(
    ("member", "book", "code"),
    [
        ("M-1003", "B-007", "duplicate-hold"),
        ("M-1001", "B-001", "already-on-loan"),
        ("M-1004", "B-009", "account-blocked"),
    ],
)
def test_hold_refusals(client, member, book, code):
    assert_problem(client.post("/v1/holds", json={"book_id": book}, headers=as_member(member)), 409, code)


def test_cancelling_a_ready_hold_gives_the_copy_to_the_next_member(client):
    lea, karim = as_member("M-1001"), as_member("M-1002")
    ready = client.post("/v1/holds", json={"book_id": "B-006"}, headers=lea).json()
    queued = client.post("/v1/holds", json={"book_id": "B-006"}, headers=karim).json()
    assert (ready["status"], queued["status"]) == ("READY_FOR_PICKUP", "QUEUED")

    assert client.delete(f"/v1/holds/{ready['id']}", headers=lea).json()["status"] == "CANCELLED"
    promoted = client.get(f"/v1/holds/{queued['id']}", headers=karim).json()
    assert promoted["status"] == "READY_FOR_PICKUP" and promoted["ready_until"] == "2026-03-07"
    assert client.get("/v1/books/B-006").json()["available_copies"] == 0  # the copy moved, not freed


def test_cancelling_the_last_ready_hold_returns_the_copy_to_the_shelf(client):
    assert client.get("/v1/books/B-002").json()["available_copies"] == 0
    client.delete("/v1/holds/H-7002", headers=as_member("M-1003"))
    assert client.get("/v1/books/B-002").json()["available_copies"] == 1


def test_cancel_is_idempotent(client):
    sofia = as_member("M-1003")
    first, second = client.delete("/v1/holds/H-7001", headers=sofia), client.delete("/v1/holds/H-7001", headers=sofia)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() and second.json()["status"] == "CANCELLED"


# ------------------------------------------------------------------ optimistic concurrency (ETag / If-Match)


def test_etag_protects_against_lost_updates(client):
    me = as_member("M-1001")
    etag = client.get("/v1/members/M-1001", headers=me).headers["ETag"]
    assert etag == '"1"'
    new_email = {"email": "lea.m@example.edu"}

    assert_problem(client.patch("/v1/members/M-1001", json=new_email, headers=me), 428, "precondition-required")
    stale = assert_problem(
        client.patch("/v1/members/M-1001", json=new_email, headers={**me, "If-Match": '"7"'}),
        412,
        "precondition-failed",
    )
    assert stale["current_etag"] == '"1"'

    updated = client.patch("/v1/members/M-1001", json=new_email, headers={**me, "If-Match": etag})
    assert updated.status_code == 200 and updated.headers["ETag"] == '"2"'
    assert updated.json()["email"] == "lea.m@example.edu"
    # A second client still holding the old ETag cannot overwrite the change.
    assert_problem(
        client.patch("/v1/members/M-1001", json={"email": "x@example.edu"}, headers={**me, "If-Match": etag}),
        412,
        "precondition-failed",
    )


def test_invalid_email_is_rejected(client):
    assert_problem(
        client.patch("/v1/members/M-1001", json={"email": "nope"}, headers=as_member("M-1001", **{"If-Match": '"1"'})),
        422,
        "validation-error",
    )


def test_unknown_route_is_a_problem_document(client):
    assert_problem(client.get("/v1/does-not-exist"), 404, "not-found")
