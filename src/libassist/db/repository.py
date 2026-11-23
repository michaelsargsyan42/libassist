from __future__ import annotations

import sqlite3
from datetime import date

from libassist.domain.models import Book, Hold, HoldStatus, Loan, Member, MemberType

# A constant spliced into SQL text (never user input; user values always go through "?" parameters).
ACTIVE = "('QUEUED', 'READY_FOR_PICKUP')"


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _member(row: sqlite3.Row) -> Member:
    return Member(
        row["id"], row["first_name"], row["last_name"], row["email"], MemberType(row["member_type"]), row["version"]
    )


def _book(row: sqlite3.Row) -> Book:
    subjects = tuple(s for s in row["subjects"].split(",") if s)
    return Book(
        row["id"],
        row["title"],
        row["author"],
        row["year"],
        subjects,
        row["description"],
        row["total_copies"],
        row["available_copies"],
    )


def _loan(row: sqlite3.Row) -> Loan:
    return Loan(
        row["id"],
        row["member_id"],
        row["book_id"],
        date.fromisoformat(row["borrowed_on"]),
        date.fromisoformat(row["due_on"]),
        row["renewal_count"],
        row["fine_carried_cents"],
        _date(row["returned_on"]),
    )


def _hold(row: sqlite3.Row) -> Hold:
    return Hold(
        row["id"],
        row["member_id"],
        row["book_id"],
        HoldStatus(row["status"]),
        row["created_at"],
        _date(row["ready_until"]),
    )


# ---------------------------------------------------------------- members


def get_member(conn: sqlite3.Connection, member_id: str) -> Member | None:
    row = conn.execute("SELECT * FROM members WHERE id = ?", (member_id,)).fetchone()
    return _member(row) if row else None


def update_member_email(conn: sqlite3.Connection, member_id: str, email: str, expected_version: int) -> bool:
    """Optimistic concurrency: only succeeds if nobody changed the row since `expected_version` was read."""
    cur = conn.execute(
        "UPDATE members SET email = ?, version = version + 1 WHERE id = ? AND version = ?",
        (email, member_id, expected_version),
    )
    return cur.rowcount == 1


# ---------------------------------------------------------------- books


def get_book(conn: sqlite3.Connection, book_id: str) -> Book | None:
    row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
    return _book(row) if row else None


def list_books(conn: sqlite3.Connection) -> list[Book]:
    return [_book(r) for r in conn.execute("SELECT * FROM books ORDER BY id")]


def take_copy(conn: sqlite3.Connection, book_id: str) -> bool:
    """Take one copy off the shelf. Returns False if none is left; never lets the count go negative."""
    cur = conn.execute(
        "UPDATE books SET available_copies = available_copies - 1 WHERE id = ? AND available_copies > 0",
        (book_id,),
    )
    return cur.rowcount == 1


def release_copy(conn: sqlite3.Connection, book_id: str) -> None:
    conn.execute(
        "UPDATE books SET available_copies = available_copies + 1 WHERE id = ? AND available_copies < total_copies",
        (book_id,),
    )


# ---------------------------------------------------------------- loans


def get_loan(conn: sqlite3.Connection, loan_id: str) -> Loan | None:
    row = conn.execute("SELECT * FROM loans WHERE id = ?", (loan_id,)).fetchone()
    return _loan(row) if row else None


def active_loans(conn: sqlite3.Connection, member_id: str) -> list[tuple[Loan, str]]:
    rows = conn.execute(
        "SELECT l.*, b.title FROM loans l JOIN books b ON b.id = l.book_id "
        "WHERE l.member_id = ? AND l.returned_on IS NULL ORDER BY l.due_on, l.id",
        (member_id,),
    )
    return [(_loan(r), r["title"]) for r in rows]


def has_active_loan(conn: sqlite3.Connection, member_id: str, book_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM loans WHERE member_id = ? AND book_id = ? AND returned_on IS NULL",
            (member_id, book_id),
        ).fetchone()
        is not None
    )


def renew_loan(
    conn: sqlite3.Connection, loan_id: str, *, new_due: date, carried_fine_cents: int, expected_renewal_count: int
) -> bool:
    cur = conn.execute(
        "UPDATE loans SET due_on = ?, renewal_count = renewal_count + 1, fine_carried_cents = ? "
        "WHERE id = ? AND renewal_count = ? AND returned_on IS NULL",
        (new_due.isoformat(), carried_fine_cents, loan_id, expected_renewal_count),
    )
    return cur.rowcount == 1


# ---------------------------------------------------------------- holds


def get_hold(conn: sqlite3.Connection, hold_id: str) -> Hold | None:
    row = conn.execute("SELECT * FROM holds WHERE id = ?", (hold_id,)).fetchone()
    return _hold(row) if row else None


def hold_title(conn: sqlite3.Connection, hold: Hold) -> str:
    return conn.execute("SELECT title FROM books WHERE id = ?", (hold.book_id,)).fetchone()["title"]


def active_holds(conn: sqlite3.Connection, member_id: str) -> list[tuple[Hold, str]]:
    rows = conn.execute(
        "SELECT h.*, b.title FROM holds h JOIN books b ON b.id = h.book_id "
        f"WHERE h.member_id = ? AND h.status IN {ACTIVE} ORDER BY h.created_at, h.id",
        (member_id,),
    )
    return [(_hold(r), r["title"]) for r in rows]


def count_active_holds(conn: sqlite3.Connection, member_id: str) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM holds WHERE member_id = ? AND status IN {ACTIVE}", (member_id,)
    ).fetchone()[0]


def has_active_hold(conn: sqlite3.Connection, member_id: str, book_id: str) -> bool:
    return (
        conn.execute(
            f"SELECT 1 FROM holds WHERE member_id = ? AND book_id = ? AND status IN {ACTIVE}",
            (member_id, book_id),
        ).fetchone()
        is not None
    )


def count_queued(conn: sqlite3.Connection, book_id: str) -> int:
    """Members waiting for a copy of this book."""
    return conn.execute("SELECT COUNT(*) FROM holds WHERE book_id = ? AND status = 'QUEUED'", (book_id,)).fetchone()[0]


def queue_position(conn: sqlite3.Connection, hold: Hold) -> int:
    """1-based position among QUEUED holds for the same book (first come, first served)."""
    ahead = conn.execute(
        "SELECT COUNT(*) FROM holds WHERE book_id = ? AND status = 'QUEUED' "
        "AND (created_at < ? OR (created_at = ? AND id < ?))",
        (hold.book_id, hold.created_at, hold.created_at, hold.id),
    ).fetchone()[0]
    return ahead + 1


def oldest_queued_hold(conn: sqlite3.Connection, book_id: str) -> Hold | None:
    row = conn.execute(
        "SELECT * FROM holds WHERE book_id = ? AND status = 'QUEUED' ORDER BY created_at, id LIMIT 1",
        (book_id,),
    ).fetchone()
    return _hold(row) if row else None


def next_hold_id(conn: sqlite3.Connection) -> str:
    """Readable sequential ids (H-7003...). Safe because writers hold the lock (BEGIN IMMEDIATE)."""
    current = conn.execute("SELECT MAX(CAST(SUBSTR(id, 3) AS INTEGER)) FROM holds").fetchone()[0]
    return f"H-{(current or 7000) + 1}"


def insert_hold(conn: sqlite3.Connection, hold: Hold) -> None:
    conn.execute(
        "INSERT INTO holds (id, member_id, book_id, status, created_at, ready_until) VALUES (?, ?, ?, ?, ?, ?)",
        (
            hold.id,
            hold.member_id,
            hold.book_id,
            hold.status.value,
            hold.created_at,
            hold.ready_until.isoformat() if hold.ready_until else None,
        ),
    )


def update_hold_status(
    conn: sqlite3.Connection,
    hold_id: str,
    *,
    new_status: HoldStatus,
    expected_status: HoldStatus,
    ready_until: date | None = None,
) -> bool:
    cur = conn.execute(
        "UPDATE holds SET status = ?, ready_until = ? WHERE id = ? AND status = ?",
        (new_status.value, ready_until.isoformat() if ready_until else None, hold_id, expected_status.value),
    )
    return cur.rowcount == 1
