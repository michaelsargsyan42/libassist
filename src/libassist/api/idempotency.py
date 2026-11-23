# idempotency key for post
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from libassist.db.database import write_transaction
from libassist.domain.errors import BadRequest, UnprocessableContent

KEY_TTL = timedelta(hours=24)
MAX_KEY_LENGTH = 255


@dataclass(frozen=True)
class IdempotentResult:
    status_code: int
    body: dict[str, Any]
    replayed: bool


def fingerprint(method: str, path: str, body: dict[str, Any] | None) -> str:
    canonical = json.dumps({"method": method, "path": path, "body": body}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_idempotent(
    conn: sqlite3.Connection,
    *,
    principal: str,
    key: str | None,
    request_hash: str,
    handler: Callable[[], tuple[int, dict[str, Any]]],
) -> IdempotentResult:
    if key is None:  # the header is optional; without it the request simply runs
        status, body = handler()
        return IdempotentResult(status, body, replayed=False)
    if not (0 < len(key) <= MAX_KEY_LENGTH) or not key.isprintable():
        raise BadRequest("Idempotency-Key must be 1-255 printable characters.", code="invalid-idempotency-key")

    scoped_key = f"{principal}:{key}"
    with write_transaction(conn):
        row = conn.execute(
            "SELECT request_hash, status_code, response_body FROM idempotency_keys WHERE scoped_key = ?",
            (scoped_key,),
        ).fetchone()
        if row is not None:
            if row["request_hash"] != request_hash:
                raise UnprocessableContent(
                    "This Idempotency-Key was already used for a different request.", code="idempotency-key-reused"
                )
            return IdempotentResult(row["status_code"], json.loads(row["response_body"]), replayed=True)

        status, body = handler()  # the service's own write_transaction joins this one
        conn.execute(
            "INSERT INTO idempotency_keys (scoped_key, request_hash, status_code, response_body, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (scoped_key, request_hash, status, json.dumps(body), datetime.now(UTC).isoformat()),
        )
        return IdempotentResult(status, body, replayed=False)


def purge_expired(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    cutoff = ((now or datetime.now(UTC)) - KEY_TTL).isoformat()
    with write_transaction(conn):
        return conn.execute("DELETE FROM idempotency_keys WHERE created_at < ?", (cutoff,)).rowcount
