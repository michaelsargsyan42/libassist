from __future__ import annotations

from datetime import date, timedelta

import pytest

from libassist.domain import policy
from libassist.domain.models import MemberType

TODAY = date(2026, 3, 2)


def days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


def renewal(**overrides):
    args = dict(
        member_type=MemberType.STUDENT,
        due_on=TODAY + timedelta(days=5),
        renewal_count=0,
        today=TODAY,
        waiting_holds=0,
        account_blocked=False,
    )
    args.update(overrides)
    return policy.check_renewal(**args)


def test_no_fine_until_the_day_after_the_due_date():
    assert policy.fine_cents(TODAY, TODAY) == 0
    assert policy.fine_cents(TODAY + timedelta(days=3), TODAY) == 0
    assert policy.fine_cents(days_ago(1), TODAY) == 25


def test_fine_is_25_cents_per_day_and_capped_at_10_euros_per_item():
    assert policy.fine_cents(days_ago(3), TODAY) == 75
    assert policy.fine_cents(days_ago(40), TODAY) == 1000
    assert policy.fine_cents(days_ago(45), TODAY) == 1000


def test_carried_fine_is_added_and_still_capped():
    assert policy.fine_cents(TODAY + timedelta(days=10), TODAY, carried_cents=75) == 75
    assert policy.fine_cents(days_ago(2), TODAY, carried_cents=75) == 125
    assert policy.fine_cents(days_ago(10), TODAY, carried_cents=900) == 1000


def test_account_is_blocked_from_exactly_10_euros():
    assert not policy.is_blocked(999)
    assert policy.is_blocked(1000)


# ------------------------------------------------------------------ renewals


@pytest.mark.parametrize(("member_type", "days"), [(MemberType.STUDENT, 21), (MemberType.STAFF, 42)])
def test_renewal_extends_the_current_due_date_by_one_loan_period(member_type, days):
    due = TODAY + timedelta(days=5)
    decision = renewal(member_type=member_type, due_on=due)
    assert decision.allowed
    assert decision.new_due_on == due + timedelta(days=days)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"account_blocked": True}, "account-blocked"),
        ({"renewal_count": 2}, "renewal-limit"),
        ({"waiting_holds": 1}, "hold-queue"),
        ({"due_on": days_ago(15)}, "overdue-too-long"),
    ],
)
def test_renewal_refusals(overrides, reason):
    decision = renewal(**overrides)
    assert not decision.allowed
    assert decision.reason == reason
    assert decision.new_due_on is None
    assert decision.detail  # always a human-readable explanation


def test_overdue_boundary_is_inclusive_at_14_days():
    assert renewal(due_on=days_ago(14)).allowed
    assert not renewal(due_on=days_ago(15)).allowed


def test_first_failing_rule_is_reported():
    # Blocked AND at the limit AND someone waiting: the block is the most useful explanation.
    decision = renewal(account_blocked=True, renewal_count=2, waiting_holds=3)
    assert decision.reason == "account-blocked"


def hold(**overrides):
    args = dict(active_holds=0, has_active_hold_for_title=False, has_title_on_loan=False, account_blocked=False)
    args.update(overrides)
    return policy.check_hold(**args)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"account_blocked": True}, "account-blocked"),
        ({"has_title_on_loan": True}, "already-on-loan"),
        ({"has_active_hold_for_title": True}, "duplicate-hold"),
        ({"active_holds": 5}, "hold-limit"),
    ],
)
def test_hold_refusals(overrides, reason):
    decision = hold(**overrides)
    assert not decision.allowed
    assert decision.reason == reason


def test_hold_allowed_below_the_limit():
    assert hold(active_holds=4).allowed


def test_pickup_window_is_5_days():
    assert policy.pickup_deadline(TODAY) == date(2026, 3, 7)
