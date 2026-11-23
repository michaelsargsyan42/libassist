from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class MemberType(StrEnum):
    STUDENT = "student"
    STAFF = "staff"


class HoldStatus(StrEnum):
    QUEUED = "QUEUED"
    READY_FOR_PICKUP = "READY_FOR_PICKUP"
    CANCELLED = "CANCELLED"
    FULFILLED = "FULFILLED"
    EXPIRED = "EXPIRED"


ACTIVE_HOLD_STATUSES = (HoldStatus.QUEUED, HoldStatus.READY_FOR_PICKUP)


@dataclass(frozen=True)
class Member:
    id: str
    first_name: str
    last_name: str
    email: str
    member_type: MemberType
    version: int


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    author: str
    year: int | None
    subjects: tuple[str, ...]
    description: str
    total_copies: int
    available_copies: int


@dataclass(frozen=True)
class Loan:
    id: str
    member_id: str
    book_id: str
    borrowed_on: date
    due_on: date
    renewal_count: int
    fine_carried_cents: int  # fine frozen by an earlier renewal of an overdue loan
    returned_on: date | None


@dataclass(frozen=True)
class Hold:
    id: str
    member_id: str
    book_id: str
    status: HoldStatus
    created_at: str
    ready_until: date | None


@dataclass(frozen=True)
class RenewalDecision:
    allowed: bool
    reason: str | None = None  # machine-readable (hold-queue)
    detail: str | None = None  # human-readable explanation
    new_due_on: date | None = None


@dataclass(frozen=True)
class LoanView:
    loan: Loan
    title: str
    overdue_days: int
    fine_cents: int
    renewal: RenewalDecision


@dataclass(frozen=True)
class HoldView:
    hold: Hold
    title: str
    queue_position: int | None  # (only for queued)


@dataclass(frozen=True)
class BookView:
    book: Book
    queue_length: int


@dataclass(frozen=True)
class AccountView:
    member: Member
    loans: list[LoanView]
    holds: list[HoldView]
    fines_total_cents: int
    blocked: bool


@dataclass(frozen=True)
class RenewalResult:
    loan: LoanView
    previous_due_on: date
