"""Request and response models: the API contract.

FastAPI turns these Pydantic models into the OpenAPI document served at /openapi.json (and the docs UI at
/docs), which is what a Java or TypeScript client would be generated from. The domain dataclasses stay
internal; these models decide exactly which fields leave the service.

Money is sent as integer cents plus a display string. Agents (and people) should never have to do
arithmetic on floats to tell a member what they owe.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field

from libassist.domain.models import AccountView, BookView, HoldView, LoanView, Member

MemberId = Annotated[str, Field(pattern=r"^M-\d{4}$", examples=["M-1001"])]
BookId = Annotated[str, Field(pattern=r"^B-\d{3}$", examples=["B-001"])]
LoanId = Annotated[str, Field(pattern=r"^L-\d{4}$", examples=["L-5001"])]
HoldId = Annotated[str, Field(pattern=r"^H-\d{4}$", examples=["H-7001"])]


def euros(cents: int) -> str:
    return f"€{cents // 100}.{cents % 100:02d}"


# ------------------------------------------------------------------ requests


class VerifyMemberIn(BaseModel):
    member_id: MemberId
    last_name: Annotated[str, Field(min_length=1, max_length=100)]


class PlaceHoldIn(BaseModel):
    book_id: BookId


class UpdateMemberIn(BaseModel):
    email: Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254)]


# ------------------------------------------------------------------ responses


class BookOut(BaseModel):
    id: str
    title: str
    author: str
    year: int | None
    subjects: list[str]
    description: str
    total_copies: int
    available_copies: int
    queue_length: int
    score: float | None = None  # relevance, only on search results

    @classmethod
    def build(cls, view: BookView, score: float | None = None) -> BookOut:
        b = view.book
        return cls(
            id=b.id,
            title=b.title,
            author=b.author,
            year=b.year,
            subjects=list(b.subjects),
            description=b.description,
            total_copies=b.total_copies,
            available_copies=b.available_copies,
            queue_length=view.queue_length,
            score=score,
        )


class BookSearchOut(BaseModel):
    query: str | None
    results: list[BookOut]


class PolicyPassageOut(BaseModel):
    citation: str = Field(description="Stable id to cite, e.g. loans#renewals")
    title: str
    text: str
    score: float


class PolicySearchOut(BaseModel):
    query: str
    results: list[PolicyPassageOut] = Field(description="Empty when nothing relevant was found")


class VerifyMemberOut(BaseModel):
    verified: bool
    member_id: str | None = None
    first_name: str | None = None


class MemberOut(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    member_type: str
    version: int

    @classmethod
    def build(cls, m: Member) -> MemberOut:
        return cls(
            id=m.id,
            first_name=m.first_name,
            last_name=m.last_name,
            email=m.email,
            member_type=m.member_type.value,
            version=m.version,
        )


class RenewalEligibilityOut(BaseModel):
    allowed: bool
    reason: str | None = None
    detail: str | None = None
    new_due_on: date | None = None


class LoanOut(BaseModel):
    id: str
    book_id: str
    title: str
    borrowed_on: date
    due_on: date
    renewal_count: int
    overdue_days: int
    fine_cents: int
    fine: str
    renewal: RenewalEligibilityOut

    @classmethod
    def build(cls, view: LoanView) -> LoanOut:
        loan, r = view.loan, view.renewal
        return cls(
            id=loan.id,
            book_id=loan.book_id,
            title=view.title,
            borrowed_on=loan.borrowed_on,
            due_on=loan.due_on,
            renewal_count=loan.renewal_count,
            overdue_days=view.overdue_days,
            fine_cents=view.fine_cents,
            fine=euros(view.fine_cents),
            renewal=RenewalEligibilityOut(allowed=r.allowed, reason=r.reason, detail=r.detail, new_due_on=r.new_due_on),
        )


class HoldOut(BaseModel):
    id: str
    book_id: str
    title: str
    status: str
    created_at: str
    ready_until: date | None
    queue_position: int | None

    @classmethod
    def build(cls, view: HoldView) -> HoldOut:
        h = view.hold
        return cls(
            id=h.id,
            book_id=h.book_id,
            title=view.title,
            status=h.status.value,
            created_at=h.created_at,
            ready_until=h.ready_until,
            queue_position=view.queue_position,
        )


class AccountOut(BaseModel):
    member: MemberOut
    loans: list[LoanOut]
    holds: list[HoldOut]
    fines_total_cents: int
    fines_total: str
    blocked: bool

    @classmethod
    def build(cls, view: AccountView) -> AccountOut:
        return cls(
            member=MemberOut.build(view.member),
            loans=[LoanOut.build(v) for v in view.loans],
            holds=[HoldOut.build(v) for v in view.holds],
            fines_total_cents=view.fines_total_cents,
            fines_total=euros(view.fines_total_cents),
            blocked=view.blocked,
        )


class RenewalOut(BaseModel):
    loan: LoanOut
    previous_due_on: date


class ProblemOut(BaseModel):
    """RFC 9457 Problem Details, documented for OpenAPI."""

    type: str
    title: str
    status: int
    detail: str
    code: str
    instance: str | None = None
