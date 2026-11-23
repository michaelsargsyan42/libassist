"""MCP server tests, using the SDK's in-memory client: the full MCP protocol, without a subprocess."""

from __future__ import annotations

import httpx
import pytest
from mcp import Client

from libassist.mcp_server.api_client import ApiError, LibraryApiClient
from libassist.mcp_server.server import create_server

pytestmark = pytest.mark.anyio

READ_ONLY_TOOLS = {"search_catalog", "get_book", "search_policies", "verify_member", "get_account"}
WRITE_TOOLS = {"renew_loan", "place_hold", "cancel_hold"}


async def test_tools_are_listed_with_annotations(mcp_server):
    async with Client(mcp_server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == READ_ONLY_TOOLS | WRITE_TOOLS
    for name in READ_ONLY_TOOLS:
        assert tools[name].annotations.read_only_hint is True
    for name in WRITE_TOOLS:
        assert tools[name].annotations.read_only_hint is False
    assert tools["cancel_hold"].annotations.destructive_hint is True
    assert tools["renew_loan"].annotations.destructive_hint is False


async def test_input_schemas_constrain_ids(mcp_server):
    async with Client(mcp_server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    schema = tools["renew_loan"].input_schema
    assert schema["properties"]["loan_id"]["pattern"] == r"^L-\d{4}$"
    assert set(schema["required"]) == {"member_id", "loan_id", "idempotency_key"}


async def test_policy_search_returns_structured_citations(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("search_policies", {"question": "late fee per day"})
    assert not result.is_error
    assert result.structured_content["results"][0]["citation"] == "fines#overdue-fines"


async def test_invalid_arguments_are_rejected_before_any_http_call(app):
    requests: list[httpx.Request] = []

    class CountingTransport(httpx.ASGITransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return await super().handle_async_request(request)

    server = create_server(lambda: LibraryApiClient("http://t", transport=CountingTransport(app=app)))
    async with Client(server) as client:
        result = await client.call_tool("get_book", {"book_id": "B-1"})
    assert result.is_error and "pattern" in result.content[0].text
    assert requests == []


async def test_business_refusal_becomes_a_readable_tool_error(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "renew_loan", {"member_id": "M-1002", "loan_id": "L-5004", "idempotency_key": "test-key-0001"}
        )
    assert result.is_error
    assert "Refused (hold-queue): Another member is waiting for this book" in result.content[0].text


async def test_write_with_the_same_key_is_applied_once(mcp_server, db):
    args = {"member_id": "M-1001", "loan_id": "L-5001", "idempotency_key": "test-key-0002"}
    async with Client(mcp_server) as client:
        first = await client.call_tool("renew_loan", args)
        retry = await client.call_tool("renew_loan", args)
    assert first.structured_content == retry.structured_content
    assert db.execute("SELECT renewal_count FROM loans WHERE id = 'L-5001'").fetchone()[0] == 1


async def test_policy_resources_and_prompt(mcp_server):
    async with Client(mcp_server) as client:
        assert [r.uri for r in (await client.list_resources()).resources] == ["policy://index"]
        document = await client.read_resource("policy://holds")
        assert document.contents[0].text.startswith("# Holds and reservations")
        prompt = await client.get_prompt("explain_policy", {"question": "Can I renew twice?"})
        assert "search_policies" in prompt.messages[0].content.text


# ------------------------------------------------------------------ the HTTP client's retry policy


def flaky_transport(calls: list[str], failures: int, status: int = 503) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if len(calls) <= failures:
            return httpx.Response(status, json={"code": "unavailable", "detail": "try later"})
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


async def test_get_requests_are_retried():
    calls: list[str] = []
    async with LibraryApiClient("http://t", transport=flaky_transport(calls, 2), backoff_seconds=0) as client:
        assert await client.get_book("B-001") == {"ok": True}
    assert calls == ["GET", "GET", "GET"]


async def test_post_without_idempotency_key_is_not_retried():
    calls: list[str] = []
    async with LibraryApiClient("http://t", transport=flaky_transport(calls, 1), backoff_seconds=0) as client:
        with pytest.raises(ApiError) as error:
            await client._request("POST", "/v1/holds", json={"book_id": "B-001"})
    assert error.value.status == 503 and calls == ["POST"]


async def test_post_with_idempotency_key_is_retried():
    calls: list[str] = []
    async with LibraryApiClient("http://t", transport=flaky_transport(calls, 1), backoff_seconds=0) as client:
        assert await client.place_hold("M-1001", "B-006", "key-123456") == {"ok": True}
    assert calls == ["POST", "POST"]


async def test_business_errors_are_not_retried():
    calls: list[str] = []
    async with LibraryApiClient(
        "http://t", transport=flaky_transport(calls, 5, status=409), backoff_seconds=0
    ) as client:
        with pytest.raises(ApiError) as error:
            await client.renew_loan("M-1001", "L-5002", "key-123456")
    assert error.value.status == 409 and calls == ["POST"]


async def test_unreachable_api_becomes_a_503_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with LibraryApiClient("http://t", transport=httpx.MockTransport(refuse), backoff_seconds=0) as client:
        with pytest.raises(ApiError) as error:
            await client.get_book("B-001")
    assert error.value.status == 503 and error.value.code == "api-unreachable"
    assert "unavailable" in error.value.for_model()
