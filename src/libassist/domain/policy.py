from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from libassist.domain.models import MemberType, RenewalDecision

LOAN_DAYS: dict[MemberType, int] = {MemberType.STUDENT: 21, MemberType.STAFF: 42}
MAX_RENEWALS = 2
MAX_OVERDUE_DAYS_FOR_RENEWAL = 14
MAX_ACTIVE_HOLDS = 5
PICKUP_WINDOW_DAYS = 5
FINE_PER_DAY_CENTS = 25
FINE_CAP_PER_ITEM_CENTS = 1000
BLOCK_THRESHOLD_CENTS = 1000


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None
    detail: str | None = None


ALLOWED = Decision(True)


def overdue_days(due_on: date, today: date) -> int:
    return max(0, (today - due_on).days)


# 0.25 per day + fine + capped
def fine_cents(due_on: date, today: date, carried_cents: int = 0) -> int:
    return min(FINE_CAP_PER_ITEM_CENTS, carried_cents + overdue_days(due_on, today) * FINE_PER_DAY_CENTS)


def is_blocked(total_fines_cents: int) -> bool:
    return total_fines_cents >= BLOCK_THRESHOLD_CENTS


# renewal extends current date by one loan period
def renewed_due_date(member_type: MemberType, current_due: date) -> date:
    return current_due + timedelta(days=LOAN_DAYS[member_type])


# can we renew the loan?
def check_renewal(
    *,
    member_type: MemberType,
    due_on: date,
    renewal_count: int,
    today: date,
    waiting_holds: int,
    account_blocked: bool,
) -> RenewalDecision:
    if account_blocked:
        return RenewalDecision(False, "account-blocked", "The account is blocked because unpaid fines reached €10.")
    if renewal_count >= MAX_RENEWALS:
        return RenewalDecision(False, "renewal-limit", f"This loan has already been renewed {MAX_RENEWALS} times.")
    if waiting_holds > 0:
        return RenewalDecision(False, "hold-queue", "Another member is waiting for this book, so it cannot be renewed.")
    if overdue_days(due_on, today) > MAX_OVERDUE_DAYS_FOR_RENEWAL:
        return RenewalDecision(
            False, "overdue-too-long", f"Loans more than {MAX_OVERDUE_DAYS_FOR_RENEWAL} days overdue cannot be renewed."
        )
    return RenewalDecision(True, new_due_on=renewed_due_date(member_type, due_on))


def check_hold(
    *,
    active_holds: int,
    has_active_hold_for_title: bool,
    has_title_on_loan: bool,
    account_blocked: bool,
) -> Decision:
    if account_blocked:
        return Decision(False, "account-blocked", "The account is blocked because unpaid fines reached €10.")
    if has_title_on_loan:
        return Decision(False, "already-on-loan", "You already have this book on loan.")
    if has_active_hold_for_title:
        return Decision(False, "duplicate-hold", "You already have an active hold on this book.")
    if active_holds >= MAX_ACTIVE_HOLDS:
        return Decision(False, "hold-limit", f"You already have {MAX_ACTIVE_HOLDS} active holds.")
    return ALLOWED


def pickup_deadline(today: date) -> date:
    return today + timedelta(days=PICKUP_WINDOW_DAYS)
