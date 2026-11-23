from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception. Joins an already-open transaction."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def init_db(db_path: str | Path, seed_path: str | Path, today: date, *, reset: bool = False) -> None:
    """Create the schema and load the seed data if the database is empty."""
    db_path = Path(db_path)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        if conn.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 0:
            seed(conn, json.loads(Path(seed_path).read_text(encoding="utf-8")), today)
    finally:
        conn.close()


def seed(conn: sqlite3.Connection, data: dict, today: date) -> None:
    """Insert demo data. Dates in the seed file are offsets from `today`."""
    midnight = datetime.combine(today, datetime.min.time())
    with write_transaction(conn):
        for m in data["members"]:
            conn.execute(
                "INSERT INTO members (id, first_name, last_name, email, member_type) VALUES (?, ?, ?, ?, ?)",
                (m["id"], m["first_name"], m["last_name"], m["email"], m["member_type"]),
            )
        for b in data["books"]:
            # available_copies starts at total; loans and ready holds below take copies off the shelf.
            conn.execute(
                "INSERT INTO books (id, title, author, year, subjects, description, total_copies, "
                "available_copies) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    b["id"],
                    b["title"],
                    b["author"],
                    b.get("year"),
                    ",".join(b.get("subjects", [])),
                    b.get("description", ""),
                    b["total_copies"],
                    b["total_copies"],
                ),
            )
        for loan in data["loans"]:
            conn.execute(
                "INSERT INTO loans (id, member_id, book_id, borrowed_on, due_on, renewal_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    loan["id"],
                    loan["member_id"],
                    loan["book_id"],
                    (today - timedelta(days=loan["borrowed_days_ago"])).isoformat(),
                    (today + timedelta(days=loan["due_in_days"])).isoformat(),
                    loan.get("renewal_count", 0),
                ),
            )
            _take_copy(conn, loan["book_id"])
        for h in data["holds"]:
            ready_until = None
            if h["status"] == "READY_FOR_PICKUP":
                ready_until = (today + timedelta(days=h.get("ready_for_days", 5))).isoformat()
                _take_copy(conn, h["book_id"])
            created_at = (midnight - timedelta(days=h.get("created_days_ago", 0))).isoformat()
            conn.execute(
                "INSERT INTO holds (id, member_id, book_id, status, created_at, ready_until) VALUES (?, ?, ?, ?, ?, ?)",
                (h["id"], h["member_id"], h["book_id"], h["status"], created_at, ready_until),
            )


def _take_copy(conn: sqlite3.Connection, book_id: str) -> None:
    cur = conn.execute(
        "UPDATE books SET available_copies = available_copies - 1 WHERE id = ? AND available_copies > 0",
        (book_id,),
    )
    if cur.rowcount != 1:
        raise ValueError(f"Seed data is inconsistent: no copy of {book_id} left to assign")
