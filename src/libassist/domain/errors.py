from __future__ import annotations

from typing import Any

PROBLEM_BASE = "https://libassist.example/problems/"


class DomainError(Exception):
    status: int = 400
    code: str = "bad-request"
    title: str = "Bad request"

    def __init__(
        self, detail: str, *, code: str | None = None, title: str | None = None, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if code:
            self.code = code
        if title:
            self.title = title
        self.extra = extra or {}

    def to_problem(self, instance: str | None = None) -> dict[str, Any]:
        problem: dict[str, Any] = {
            "type": PROBLEM_BASE + self.code,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "code": self.code,
        }
        if instance:
            problem["instance"] = instance
        problem.update(self.extra)
        return problem


class BadRequest(DomainError):
    pass


class Unauthorized(DomainError):
    status, code, title = 401, "unauthenticated", "Authentication required"


class Forbidden(DomainError):
    status, code, title = 403, "forbidden", "Not allowed"


class NotFound(DomainError):
    status, code, title = 404, "not-found", "Not found"


class Conflict(DomainError):
    """The request is valid but conflicts with the current state (a business rule says no)."""

    status, code, title = 409, "conflict", "Request conflicts with current state"


class PreconditionFailed(DomainError):
    status, code, title = 412, "precondition-failed", "Precondition failed"


class UnprocessableContent(DomainError):
    status, code, title = 422, "unprocessable", "Unprocessable content"


class PreconditionRequired(DomainError):
    status, code, title = 428, "precondition-required", "Precondition required"
