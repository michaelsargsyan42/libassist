from __future__ import annotations

import logging
import os
import time
import uuid

from fastapi import FastAPI, Request, Response

from libassist import __version__
from libassist.api.errors import install_error_handlers
from libassist.api.idempotency import purge_expired
from libassist.api.routes import router
from libassist.config import Settings
from libassist.db import repository as repo
from libassist.db.database import connect, init_db
from libassist.rag.embeddings import Embedder, make_embedder
from libassist.rag.index import build_catalog_index, build_policy_index

log = logging.getLogger("libassist.api")

DESCRIPTION = """University library API used by the LibAssist agent (through its MCP server) and by any
other client. Errors use RFC 9457 Problem Details. Authentication is simulated with the `X-Member-Id`
header. POST endpoints accept an `Idempotency-Key` header."""


def create_app(settings: Settings | None = None, *, embedder: Embedder | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    init_db(settings.db_path, settings.seed_path, settings.today())
    embedder = embedder or make_embedder(
        settings.embeddings_base_url, settings.embeddings_model, settings.embeddings_api_key
    )

    conn = connect(settings.db_path)
    try:
        books = repo.list_books(conn)
        purge_expired(conn)
    finally:
        conn.close()

    app = FastAPI(title="LibAssist Library API", version=__version__, description=DESCRIPTION)
    app.state.settings = settings

    # Indexes are built once at startup. Policies and book descriptions are static in this demo; a real
    # system would re-index on change (or listen to change events) instead.
    app.state.policy_index = build_policy_index(settings.policies_dir, embedder)
    app.state.catalog_index = build_catalog_index(books, embedder)
    log.info(
        "indexes built with %s: %d policy chunks, %d books",
        embedder.name,
        len(app.state.policy_index.chunks),
        len(books),
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        log.info(
            "%s %s -> %d in %.1f ms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app


def run() -> None:
    """Console entry point: `libassist-api`."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        create_app(),
        host=os.environ.get("LIBASSIST_HOST", "127.0.0.1"),
        port=int(os.environ.get("LIBASSIST_PORT", "8000")),
    )


if __name__ == "__main__":
    run()
