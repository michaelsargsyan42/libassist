from __future__ import annotations

import logging
import sqlite3
import unicodedata
from datetime import date, datetime

from libassist.db import repository as repo
from libassist.db.database import write_transaction
from libassist.domain import policy
from libassist.domain.errors import Conflict, NotFound, PreconditionFailed
from libassist.domain.models import (
    ACTIVE_HOLD_STATUSES,
    AccountView,
    BookView,
    Hold,
    HoldStatus,
    HoldView,
    Loan,
    LoanView,
    Member,
    RenewalResult,
)

log = logging.getLogger(__name__)


# case-sensitivity and accents
def _normalize_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).casefold().split())


# the operations that API exposes


class LibraryService:
    def __init__(self, conn: sqlite3.Connection, today: date) -> None:
        self.conn = conn
        self.today = today

    # helper functions
    def _now(self) -> str:
        return f"{self.today.isoformat()}T{datetime.now().strftime('%H:%M:%S.%f')}"

    # we can only call ourselves
    def _require_self(self, principal: str, member_id: str) -> Member:
        member = repo.get_member(self.conn, member_id)
        if member is None or member_id != principal:
            raise NotFound(f"Member {member_id} not found.")
        return member

    # if you renew fines don't disappear
    def _fines_total(self, loans: list[Loan]) -> int:
        return sum(policy.fine_cents(loan.due_on, self.today, loan.fine_carried_cents) for loan in loans)

    # wrap the loan in a nice viewer
    def _loan_view(self, member: Member, loan: Loan, title: str, account_blocked: bool) -> LoanView:
        decision = policy.check_renewal(
            member_type=member.member_type,
            due_on=loan.due_on,
            renewal_count=loan.renewal_count,
            today=self.today,
            waiting_holds=repo.count_queued(self.conn, loan.book_id),
            account_blocked=account_blocked,
        )
        return LoanView(
            loan=loan,
            title=title,
            overdue_days=policy.overdue_days(loan.due_on, self.today),
            fine_cents=policy.fine_cents(loan.due_on, self.today, loan.fine_carried_cents),
            renewal=decision,
        )

    def _hold_view(self, hold: Hold, title: str) -> HoldView:
        position = repo.queue_position(self.conn, hold) if hold.status == HoldStatus.QUEUED else None
        return HoldView(hold=hold, title=title, queue_position=position)

    # for members
    def member(self, principal: str, member_id: str) -> Member:
        return self._require_self(principal, member_id)

    # ID check (last name + code)
    def verify_member(self, member_id: str, last_name: str) -> Member | None:

        member = repo.get_member(self.conn, member_id)
        if member is None:
            return None
        same = _normalize_name(member.last_name) == _normalize_name(last_name)
        return member if same else None

    def update_email(self, principal: str, member_id: str, email: str, expected_version: int) -> Member:
        with write_transaction(self.conn):
            member = self._require_self(principal, member_id)
            if not repo.update_member_email(self.conn, member_id, email, expected_version):
                raise PreconditionFailed(
                    "The member record changed since you read it. Fetch it again, then retry.",
                    extra={"current_etag": f'"{member.version}"'},
                )
            return repo.get_member(self.conn, member_id)  # type: ignore[return-value]

    #  catalogue

    def book(self, book_id: str) -> BookView:
        book = repo.get_book(self.conn, book_id)
        if book is None:
            raise NotFound(f"Book {book_id} not found.")
        return BookView(book=book, queue_length=repo.count_queued(self.conn, book_id))

    def books(self, book_ids: list[str]) -> list[BookView]:
        """Fresh availability for search hits, in the order given."""
        return [self.book(book_id) for book_id in book_ids]

    def all_books(self) -> list[BookView]:
        return [BookView(b, repo.count_queued(self.conn, b.id)) for b in repo.list_books(self.conn)]

    # account

    def account(self, principal: str, member_id: str) -> AccountView:
        member = self._require_self(principal, member_id)
        loans = repo.active_loans(self.conn, member_id)
        total = self._fines_total([loan for loan, _ in loans])
        blocked = policy.is_blocked(total)
        return AccountView(
            member=member,
            loans=[self._loan_view(member, loan, title, blocked) for loan, title in loans],
            holds=[self._hold_view(hold, title) for hold, title in repo.active_holds(self.conn, member_id)],
            fines_total_cents=total,
            blocked=blocked,
        )

    # loans

    def renew_loan(self, principal: str, loan_id: str) -> RenewalResult:
        with write_transaction(self.conn):
            loan = repo.get_loan(self.conn, loan_id)
            if loan is None or loan.member_id != principal:
                raise NotFound(f"Loan {loan_id} not found.")
            if loan.returned_on is not None:
                raise Conflict("This loan has already been returned.", code="loan-returned")
            member = repo.get_member(self.conn, principal)
            assert member is not None
            active = [entry for entry, _ in repo.active_loans(self.conn, principal)]
            blocked = policy.is_blocked(self._fines_total(active))
            title = repo.get_book(self.conn, loan.book_id).title  # type: ignore[union-attr]

            decision = self._loan_view(member, loan, title, blocked).renewal
            if not decision.allowed:
                raise Conflict(
                    decision.detail or "Renewal refused.",
                    code=decision.reason,
                    title="Renewal refused",
                    extra={"loan_id": loan_id},
                )
            assert decision.new_due_on is not None

            # Freeze the fine owed so far, so renewing an overdue loan does not wipe it out.
            carried = policy.fine_cents(loan.due_on, self.today, loan.fine_carried_cents)
            if not repo.renew_loan(
                self.conn,
                loan_id,
                new_due=decision.new_due_on,
                carried_fine_cents=carried,
                expected_renewal_count=loan.renewal_count,
            ):
                raise Conflict("The loan was changed by another request. Please retry.", code="concurrent-update")

            updated = repo.get_loan(self.conn, loan_id)
            assert updated is not None
            log.info("loan renewed", extra={"loan_id": loan_id, "new_due": str(updated.due_on)})
            return RenewalResult(loan=self._loan_view(member, updated, title, blocked), previous_due_on=loan.due_on)

    # holds

    def get_hold(self, principal: str, hold_id: str) -> HoldView:
        hold = repo.get_hold(self.conn, hold_id)
        if hold is None or hold.member_id != principal:
            raise NotFound(f"Hold {hold_id} not found.")
        return self._hold_view(hold, repo.hold_title(self.conn, hold))

    def place_hold(self, principal: str, book_id: str) -> HoldView:
        with write_transaction(self.conn):
            book = repo.get_book(self.conn, book_id)
            if book is None:
                raise NotFound(f"Book {book_id} not found.")
            loans = [entry for entry, _ in repo.active_loans(self.conn, principal)]
            decision = policy.check_hold(
                active_holds=repo.count_active_holds(self.conn, principal),
                has_active_hold_for_title=repo.has_active_hold(self.conn, principal, book_id),
                has_title_on_loan=repo.has_active_loan(self.conn, principal, book_id),
                account_blocked=policy.is_blocked(self._fines_total(loans)),
            )
            if not decision.allowed:
                raise Conflict(decision.detail or "Hold refused.", code=decision.reason, title="Hold refused")

            # A copy on the shelf is set aside immediately; otherwise the member joins the queue.
            if repo.take_copy(self.conn, book_id):
                status, ready_until = HoldStatus.READY_FOR_PICKUP, policy.pickup_deadline(self.today)
            else:
                status, ready_until = HoldStatus.QUEUED, None
            hold = Hold(repo.next_hold_id(self.conn), principal, book_id, status, self._now(), ready_until)
            try:
                repo.insert_hold(self.conn, hold)
            except sqlite3.IntegrityError as exc:  # the partial unique index caught a duplicate
                raise Conflict("You already have an active hold on this book.", code="duplicate-hold") from exc
            return self._hold_view(hold, book.title)

    def cancel_hold(self, principal: str, hold_id: str) -> HoldView:
        """Cancel a hold. Cancelling an already-cancelled hold is a no-op (DELETE is idempotent)."""
        with write_transaction(self.conn):
            hold = repo.get_hold(self.conn, hold_id)
            if hold is None or hold.member_id != principal:
                raise NotFound(f"Hold {hold_id} not found.")
            title = repo.hold_title(self.conn, hold)
            if hold.status == HoldStatus.CANCELLED:
                return self._hold_view(hold, title)
            if hold.status not in ACTIVE_HOLD_STATUSES:
                raise Conflict(
                    f"This hold is {hold.status.value.lower()} and can no longer be cancelled.", code="hold-not-active"
                )

            repo.update_hold_status(self.conn, hold_id, new_status=HoldStatus.CANCELLED, expected_status=hold.status)
            if hold.status == HoldStatus.READY_FOR_PICKUP:
                # The copy that was set aside goes to the next member in the queue, or back to the shelf.
                next_hold = repo.oldest_queued_hold(self.conn, hold.book_id)
                if next_hold is not None:
                    repo.update_hold_status(
                        self.conn,
                        next_hold.id,
                        new_status=HoldStatus.READY_FOR_PICKUP,
                        expected_status=HoldStatus.QUEUED,
                        ready_until=policy.pickup_deadline(self.today),
                    )
                    log.info("hold promoted", extra={"hold_id": next_hold.id})
                else:
                    repo.release_copy(self.conn, hold.book_id)
            cancelled = repo.get_hold(self.conn, hold_id)
            assert cancelled is not None
            return self._hold_view(cancelled, title)
