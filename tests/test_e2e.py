"""End-to-end tests over real sockets and processes: uvicorn serving HTTP, the MCP server as a subprocess
speaking JSON-RPC over stdio, and real concurrent HTTP clients. Still fast (a few seconds), still no LLM."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from mcp import Client

from libassist.agent.cli import mcp_server_params
from libassist.agent.llm import ScriptedChatModel, call, last_tool_json, say
from libassist.agent.orchestrator import Orchestrator
from libassist.agent.toolbox import McpToolbox
from libassist.config import REPO_ROOT
from libassist.db.database import connect
from libassist.devserver import ApiServer
from tests.conftest import as_member

pytestmark = pytest.mark.e2e


@pytest.fixture
def api_server(settings):
    with ApiServer(settings) as server:
        yield server


@pytest.mark.anyio
async def test_agent_renews_a_loan_through_stdio_mcp_and_http(api_server, settings):
    steps = [
        call("verify_member", member_id="M-1001", last_name="Martin"),
        call("renew_loan", loan_id="L-5001"),
        lambda m: say("Due " + last_tool_json(m)["loan"]["due_on"]),
    ]
    async with Client(mcp_server_params(api_server.base_url)) as client:
        agent = await Orchestrator.create(ScriptedChatModel(steps), McpToolbox(client))
        assert (await agent.handle("I'm M-1001, Martin. Renew L-5001.")).needs_confirmation
        assert (await agent.handle("yes")).text == "Due 2026-03-28"

    conn = connect(settings.db_path)
    assert conn.execute("SELECT renewal_count FROM loans WHERE id = 'L-5001'").fetchone()[0] == 1
    conn.close()


def test_concurrent_http_retries_with_one_key_renew_once(api_server):
    headers = as_member("M-1001", **{"Idempotency-Key": "e2e-concurrent-key"})

    def post(_: int) -> httpx.Response:
        return httpx.post(f"{api_server.base_url}/v1/loans/L-5001/renewals", headers=headers, timeout=15)

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(post, range(6)))
    assert {r.status_code for r in responses} == {200}
    assert len({r.text for r in responses}) == 1
    assert sum(r.headers.get("Idempotent-Replayed") == "true" for r in responses) == 5
    account = httpx.get(f"{api_server.base_url}/v1/members/M-1001/account", headers=as_member("M-1001")).json()
    assert next(loan for loan in account["loans"] if loan["id"] == "L-5001")["renewal_count"] == 1


def test_demo_runs_end_to_end():
    result = subprocess.run(
        [sys.executable, "-m", "libassist.demo"], capture_output=True, text=True, timeout=120, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "[loans#renewals]" in out
    assert "guardrail: tool_rejected get_account (identity-not-verified)" in out
    assert "orchestrator: injection_suspected" in out
    assert "[('H-7001', 'QUEUED'), ('H-7002', 'READY_FOR_PICKUP')]" in out
    assert "Idempotent-Replayed: true, renewal_count still 1" in out
