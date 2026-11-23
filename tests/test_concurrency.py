"""Concurrency tests: real threads, each with its own SQLite connection, released at the same instant.

These are the bugs that unit tests miss: two requests both read "1 copy left" (or "1 renewal left") and
both act on it. The fix is to check and write inside one transaction that takes the write lock first
(BEGIN IMMEDIATE), plus conditional UPDATEs as a second line of defence.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from libassist.api.idempotency import fingerprint, run_idempotent
from libassist.db.database import connect
from libassist.domain.errors import Conflict
from libassist.domain.service import LibraryService
from tests.conftest import TODAY


def run_concurrently(n: int, task: Callable[[int], Any]) -> list[Any]:
    """Start n threads that wait at a barrier, then all call task(i) at once. Returns results or exceptions."""
    barrier = threading.Barrier(n)
    results: list[Any] = [None] * n

    def worker(i: int) -> None:
        barrier.wait()
        try:
            results[i] = task(i)
        except Exception as exc:  # collected and asserted on by the test
            results[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


def with_service(settings, fn: Callable[[LibraryService], Any]) -> Any:
    conn = connect(settings.db_path)
    try:
        return fn(LibraryService(conn, TODAY))
    finally:
        conn.close()


def test_two_members_race_for_the_last_copy(app, settings, db):
    # B-006 has exactly one copy on the shelf.
    members = ["M-1001", "M-1002"]
    results = run_concurrently(2, lambda i: with_service(settings, lambda s: s.place_hold(members[i], "B-006")))

    statuses = sorted(r.hold.status.value for r in results)
    assert statuses == ["QUEUED", "READY_FOR_PICKUP"]  # exactly one member got the copy
    assert db.execute("SELECT available_copies FROM books WHERE id = 'B-006'").fetchone()[0] == 0


def test_concurrent_renewals_never_exceed_the_limit(app, settings, db):
    # L-5001 has 0 renewals; the limit is 2. Five simultaneous renewals: exactly two may succeed.
    results = run_concurrently(5, lambda i: with_service(settings, lambda s: s.renew_loan("M-1001", "L-5001")))

    successes = [r for r in results if not isinstance(r, Exception)]
    refusals = [r for r in results if isinstance(r, Conflict)]
    assert len(successes) == 2 and len(refusals) == 3
    assert {r.code for r in refusals} == {"renewal-limit"}
    assert db.execute("SELECT renewal_count FROM loans WHERE id = 'L-5001'").fetchone()[0] == 2


def test_concurrent_retries_with_the_same_key_apply_once(app, settings, db):
    request_hash = fingerprint("POST", "/v1/loans/L-5001/renewals", None)

    def retry(_: int):
        def call(service: LibraryService):
            def handler() -> tuple[int, dict[str, Any]]:
                result = service.renew_loan("M-1001", "L-5001")
                return 200, {"due_on": result.loan.loan.due_on.isoformat()}

            return run_idempotent(
                service.conn, principal="M-1001", key="same-key", request_hash=request_hash, handler=handler
            )

        return with_service(settings, call)

    results = run_concurrently(4, retry)
    assert not [r for r in results if isinstance(r, Exception)], results
    assert len({r.body["due_on"] for r in results}) == 1  # every caller saw the same answer
    assert sum(not r.replayed for r in results) == 1  # and it was computed exactly once
    assert db.execute("SELECT renewal_count FROM loans WHERE id = 'L-5001'").fetchone()[0] == 1
