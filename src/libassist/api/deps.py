"""FastAPI dependencies: per-request database connection, service, and the authenticated caller.

Authentication is simulated with an `X-Member-Id` header so the project runs without an identity
provider. In production this dependency would validate an OAuth2/OIDC bearer token (signature, expiry,
audience) and take the member id from its `sub` claim; nothing else in the code would change.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, Request

from libassist.db import repository as repo
from libassist.db.database import connect
from libassist.domain.errors import Unauthorized
from libassist.domain.service import LibraryService


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_service(request: Request, conn: Annotated[sqlite3.Connection, Depends(get_conn)]) -> LibraryService:
    return LibraryService(conn, today=request.app.state.settings.today())


def current_member(
    service: Annotated[LibraryService, Depends(get_service)],
    x_member_id: Annotated[str | None, Header(description="Stand-in for a bearer token")] = None,
) -> str:
    if not x_member_id:
        raise Unauthorized("Missing X-Member-Id header (it stands in for a bearer token in this demo).")
    if repo.get_member(service.conn, x_member_id) is None:
        raise Unauthorized("Unknown member.")
    return x_member_id


Service = Annotated[LibraryService, Depends(get_service)]
Principal = Annotated[str, Depends(current_member)]
