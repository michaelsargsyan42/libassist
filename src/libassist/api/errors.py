# single error format

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from libassist.domain.errors import PROBLEM_BASE, DomainError

log = logging.getLogger(__name__)
PROBLEM_JSON = "application/problem+json"


def problem_response(
    status: int, code: str, detail: str, request: Request, *, title: str | None = None, **extra: object
) -> JSONResponse:
    body = {
        "type": PROBLEM_BASE + code,
        "title": title or HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "code": code,
        "instance": request.url.path,
        **extra,
    }
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(exc.to_problem(instance=request.url.path), status_code=exc.status, media_type=PROBLEM_JSON)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report where and why, but don't echo the submitted values back.
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()
        ]
        return problem_response(
            422, "validation-error", "The request is not valid.", request, title="Validation failed", errors=errors
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "-")
        return problem_response(exc.status_code, code, str(exc.detail), request)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return problem_response(500, "internal-error", "Something went wrong on our side.", request)
