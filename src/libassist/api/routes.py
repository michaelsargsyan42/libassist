from __future__ import annotations

import re
from typing import Annotated, Any

from fastapi import APIRouter, Header, Path, Query, Request, Response
from fastapi.responses import JSONResponse

from libassist.api.deps import Principal, Service
from libassist.api.idempotency import IdempotentResult, fingerprint, run_idempotent
from libassist.api.schemas import (
    AccountOut,
    BookOut,
    BookSearchOut,
    HoldOut,
    LoanOut,
    MemberOut,
    PlaceHoldIn,
    PolicyPassageOut,
    PolicySearchOut,
    ProblemOut,
    RenewalOut,
    UpdateMemberIn,
    VerifyMemberIn,
    VerifyMemberOut,
)
from libassist.domain.errors import NotFound, PreconditionFailed, PreconditionRequired

router = APIRouter(prefix="/v1")

BookIdPath = Annotated[str, Path(pattern=r"^B-\d{3}$")]
MemberIdPath = Annotated[str, Path(pattern=r"^M-\d{4}$")]
LoanIdPath = Annotated[str, Path(pattern=r"^L-\d{4}$")]
HoldIdPath = Annotated[str, Path(pattern=r"^H-\d{4}$")]
IdempotencyKey = Annotated[str | None, Header(description="Reuse the same value when retrying")]

PROBLEMS: dict[int | str, dict[str, Any]] = {
    code: {"model": ProblemOut, "content": {"application/problem+json": {}}} for code in (401, 404, 409, 422)
}


def _respond(result: IdempotentResult, headers: dict[str, str] | None = None) -> JSONResponse:
    headers = dict(headers or {})
    if result.replayed:
        headers["Idempotent-Replayed"] = "true"
    return JSONResponse(result.body, status_code=result.status_code, headers=headers)


def _etag(version: int) -> str:
    return f'"{version}"'


def _parse_if_match(value: str, current_version: int) -> int:
    """Return the version the client expects. '*' means "whatever is current"."""
    value = value.strip()
    if value == "*":
        return current_version
    match = re.fullmatch(r'(?:W/)?"(\d+)"', value)
    if not match:
        raise PreconditionFailed('If-Match must be an ETag returned by GET, for example "3".')
    return int(match.group(1))


# ------------------------------------------------------------------ public endpoints


@router.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/books", response_model=BookSearchOut, tags=["catalogue"])
def search_books(
    request: Request,
    service: Service,
    q: Annotated[str | None, Query(max_length=200, description="Free-text query")] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> BookSearchOut:
    if not q:
        return BookSearchOut(query=None, results=[BookOut.build(v) for v in service.all_books()[:limit]])
    hits = request.app.state.catalog_index.search(q, k=limit)
    views = service.books([hit.chunk.id for hit in hits])
    return BookSearchOut(
        query=q, results=[BookOut.build(v, score=round(h.score, 4)) for v, h in zip(views, hits, strict=True)]
    )


@router.get("/books/{book_id}", response_model=BookOut, responses=PROBLEMS, tags=["catalogue"])
def get_book(book_id: BookIdPath, service: Service) -> BookOut:
    return BookOut.build(service.book(book_id))


@router.get("/policies", tags=["policies"])
def list_policies(request: Request) -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}
    for chunk in request.app.state.policy_index.chunks:
        doc_id = chunk.id.split("#", 1)[0]
        entry = docs.setdefault(doc_id, {"id": doc_id, "title": chunk.title.split(" > ", 1)[0], "sections": []})
        entry["sections"].append(chunk.id)
    return {"documents": list(docs.values())}


@router.get("/policies/search", response_model=PolicySearchOut, tags=["policies"])
def search_policies(
    request: Request,
    q: Annotated[str, Query(min_length=2, max_length=500, description="A question in natural language")],
    k: Annotated[int, Query(ge=1, le=10)] = 3,
) -> PolicySearchOut:
    hits = request.app.state.policy_index.search(q, k=k)
    return PolicySearchOut(
        query=q,
        results=[
            PolicyPassageOut(citation=h.chunk.id, title=h.chunk.title, text=h.chunk.text, score=round(h.score, 4))
            for h in hits
        ],
    )


@router.get("/policies/{doc_id}", responses=PROBLEMS, tags=["policies"])
def get_policy(request: Request, doc_id: Annotated[str, Path(pattern=r"^[a-z0-9-]+$")]) -> dict[str, str]:
    # The pattern above rules out "../" path traversal before we touch the filesystem.
    path = request.app.state.settings.policies_dir / f"{doc_id}.md"
    if not path.is_file():
        raise NotFound(f"Policy document {doc_id} not found.")
    return {"id": doc_id, "markdown": path.read_text(encoding="utf-8")}


@router.post("/members/verify", response_model=VerifyMemberOut, tags=["members"])
def verify_member(body: VerifyMemberIn, service: Service) -> VerifyMemberOut:
    member = service.verify_member(body.member_id, body.last_name)
    if member is None:
        return VerifyMemberOut(verified=False)
    return VerifyMemberOut(verified=True, member_id=member.id, first_name=member.first_name)


# ------------------------------------------------------------------ member-scoped endpoints


@router.get("/members/{member_id}", response_model=MemberOut, responses=PROBLEMS, tags=["members"])
def get_member(member_id: MemberIdPath, service: Service, principal: Principal, response: Response) -> MemberOut:
    member = service.member(principal, member_id)
    response.headers["ETag"] = _etag(member.version)
    return MemberOut.build(member)


@router.patch(
    "/members/{member_id}",
    response_model=MemberOut,
    responses={**PROBLEMS, 412: {"model": ProblemOut}, 428: {"model": ProblemOut}},
    tags=["members"],
)
def update_member(
    member_id: MemberIdPath,
    body: UpdateMemberIn,
    service: Service,
    principal: Principal,
    response: Response,
    if_match: Annotated[str | None, Header(description="ETag from GET /v1/members/{id}")] = None,
) -> MemberOut:
    current = service.member(principal, member_id)
    if if_match is None:
        raise PreconditionRequired("Send If-Match with the ETag from GET, so concurrent edits are not lost.")
    member = service.update_email(principal, member_id, body.email, _parse_if_match(if_match, current.version))
    response.headers["ETag"] = _etag(member.version)
    return MemberOut.build(member)


@router.get("/members/{member_id}/account", response_model=AccountOut, responses=PROBLEMS, tags=["members"])
def get_account(member_id: MemberIdPath, service: Service, principal: Principal) -> AccountOut:
    return AccountOut.build(service.account(principal, member_id))


@router.post("/loans/{loan_id}/renewals", response_model=RenewalOut, responses=PROBLEMS, tags=["loans"])
def renew_loan(
    loan_id: LoanIdPath,
    request: Request,
    service: Service,
    principal: Principal,
    idempotency_key: IdempotencyKey = None,
) -> JSONResponse:
    def handler() -> tuple[int, dict[str, Any]]:
        result = service.renew_loan(principal, loan_id)
        out = RenewalOut(loan=LoanOut.build(result.loan), previous_due_on=result.previous_due_on)
        return 200, out.model_dump(mode="json")

    result = run_idempotent(
        service.conn,
        principal=principal,
        key=idempotency_key,
        request_hash=fingerprint("POST", request.url.path, None),
        handler=handler,
    )
    return _respond(result)


@router.post("/holds", status_code=201, response_model=HoldOut, responses=PROBLEMS, tags=["holds"])
def place_hold(
    body: PlaceHoldIn, request: Request, service: Service, principal: Principal, idempotency_key: IdempotencyKey = None
) -> JSONResponse:
    def handler() -> tuple[int, dict[str, Any]]:
        return 201, HoldOut.build(service.place_hold(principal, body.book_id)).model_dump(mode="json")

    result = run_idempotent(
        service.conn,
        principal=principal,
        key=idempotency_key,
        request_hash=fingerprint("POST", request.url.path, body.model_dump(mode="json")),
        handler=handler,
    )
    return _respond(result, {"Location": f"/v1/holds/{result.body['id']}"})


@router.get("/holds/{hold_id}", response_model=HoldOut, responses=PROBLEMS, tags=["holds"])
def get_hold(hold_id: HoldIdPath, service: Service, principal: Principal) -> HoldOut:
    return HoldOut.build(service.get_hold(principal, hold_id))


@router.delete("/holds/{hold_id}", response_model=HoldOut, responses=PROBLEMS, tags=["holds"])
def cancel_hold(hold_id: HoldIdPath, service: Service, principal: Principal) -> HoldOut:
    return HoldOut.build(service.cancel_hold(principal, hold_id))
