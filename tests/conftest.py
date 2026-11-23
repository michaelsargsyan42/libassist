"""Shared fixtures. Every test gets a fresh SQLite database in a temporary directory and a fixed business date,
so results never depend on the day the tests run or on the order they run in."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client

from libassist.agent.llm import ScriptedChatModel, Step
from libassist.agent.orchestrator import Orchestrator
from libassist.agent.toolbox import McpToolbox
from libassist.api.main import create_app
from libassist.config import Settings
from libassist.db.database import connect
from libassist.devserver import demo_settings
from libassist.mcp_server.api_client import LibraryApiClient
from libassist.mcp_server.server import create_server

TODAY = date(2026, 3, 2)


def as_member(member_id: str, **extra: str) -> dict[str, str]:
    return {"X-Member-Id": member_id, **extra}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return demo_settings(TODAY, tmp_path)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db(settings: Settings):
    """A direct connection, for asserting on the database state."""
    conn = connect(settings.db_path)
    yield conn
    conn.close()


@pytest.fixture
def mcp_server(app: FastAPI):
    """The real MCP server, with its HTTP calls routed to the in-memory API (no sockets)."""
    return create_server(lambda: LibraryApiClient("http://libassist.test", transport=httpx.ASGITransport(app=app)))


@pytest.fixture
def agent_session(mcp_server):
    """`async with agent_session([...steps]) as agent:` gives an orchestrator driven by a scripted model."""

    @asynccontextmanager
    async def session(steps: list[Step], **kwargs) -> AsyncIterator[Orchestrator]:
        async with Client(mcp_server) as mcp_client:
            yield await Orchestrator.create(ScriptedChatModel(steps), McpToolbox(mcp_client), **kwargs)

    return session
