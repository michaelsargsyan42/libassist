"""Run the REST API in a background thread on a free port.

Used by the demo, the end-to-end tests, the Java client test and `libassist-chat --with-api`, so that a
single command exercises the real network path (HTTP server, MCP over stdio) without manual setup.
"""

from __future__ import annotations

import tempfile
import threading
import time
from datetime import date
from pathlib import Path
from types import TracebackType

import uvicorn

from libassist.api.main import create_app
from libassist.config import REPO_ROOT, Settings


def demo_settings(today: date | None = None, db_dir: Path | None = None) -> Settings:
    """Settings with a fresh database in a temporary directory and (optionally) a fixed business date."""
    directory = db_dir or Path(tempfile.mkdtemp(prefix="libassist-"))
    return Settings(
        db_path=directory / "libassist.db",
        seed_path=REPO_ROOT / "data" / "seed.json",
        policies_dir=REPO_ROOT / "data" / "policies",
        today_override=today,
    )


class ApiServer:
    def __init__(self, settings: Settings, host: str = "127.0.0.1", port: int = 0) -> None:
        self.settings = settings
        self.host = host
        self.port = port
        self.base_url = ""
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ApiServer:
        config = uvicorn.Config(
            create_app(self.settings), host=self.host, port=self.port, log_level="warning", lifespan="off"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="libassist-api", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("The API server did not start")
            time.sleep(0.02)
        port = self._server.servers[0].sockets[0].getsockname()[1]  # the real port when port=0
        self.base_url = f"http://{self.host}:{port}"
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
