from __future__ import annotations

from typing import Any

import anyio
import httpx

from libassist import __version__

RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


# tools - routes
class ApiError(Exception):
    """An HTTP error from the API, carrying its Problem Details body."""

    def __init__(self, status: int, problem: dict[str, Any]) -> None:
        self.status = status
        self.problem = problem
        self.code = str(problem.get("code", "error"))
        self.detail = str(problem.get("detail", "The request failed."))
        super().__init__(f"{status} {self.code}: {self.detail}")

    def for_model(self) -> str:
        """A short message an LLM can relay to the user and act on."""
        if self.status == 409:
            return f"Refused ({self.code}): {self.detail}"
        if self.status == 404:
            return f"Not found: {self.detail}"
        if self.status in (401, 403):
            return f"Not allowed: {self.detail}"
        if self.status >= 500:
            return f"The library system is unavailable right now ({self.code}). Try again later."
        return f"Request rejected ({self.code}): {self.detail}"


class LibraryApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.2,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=timeout,
            headers={"User-Agent": f"libassist-mcp/{__version__}"},
        )
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> LibraryApiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        member_id: str | None = None,
        idempotency_key: str | None = None,
        safe_to_retry: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if member_id:
            headers["X-Member-Id"] = member_id
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if safe_to_retry is None:
            safe_to_retry = method in ("GET", "HEAD", "DELETE") or idempotency_key is not None
        attempts = 1 + (self.max_retries if safe_to_retry else 0)

        for attempt in range(attempts):
            last_attempt = attempt == attempts - 1
            try:
                response = await self._client.request(method, path, headers=headers, **kwargs)
            except httpx.TransportError as exc:
                if last_attempt:
                    raise ApiError(
                        503,
                        {
                            "code": "api-unreachable",
                            "detail": f"Could not reach the library API ({type(exc).__name__}).",
                        },
                    ) from exc
                await anyio.sleep(self.backoff_seconds * 2**attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and not last_attempt:
                await anyio.sleep(self.backoff_seconds * 2**attempt)
                continue
            if response.is_success:
                return response.json()
            raise ApiError(response.status_code, _problem(response))
        raise AssertionError("unreachable")  # pragma: no cover

    # ---- catalogue and policies (public)

    async def search_books(self, query: str, limit: int = 5) -> dict[str, Any]:
        return await self._request("GET", "/v1/books", params={"q": query, "limit": limit})

    async def get_book(self, book_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/books/{book_id}")

    async def search_policies(self, question: str, k: int = 3) -> dict[str, Any]:
        return await self._request("GET", "/v1/policies/search", params={"q": question, "k": k})

    async def list_policies(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/policies")

    async def get_policy(self, doc_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/policies/{doc_id}")

    # need member verification

    # post request but we can try there is no harm
    async def verify_member(self, member_id: str, last_name: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/v1/members/verify", safe_to_retry=True, json={"member_id": member_id, "last_name": last_name}
        )

    async def get_account(self, member_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/members/{member_id}/account", member_id=member_id)

    async def renew_loan(self, member_id: str, loan_id: str, idempotency_key: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/v1/loans/{loan_id}/renewals", member_id=member_id, idempotency_key=idempotency_key
        )

    async def place_hold(self, member_id: str, book_id: str, idempotency_key: str) -> dict[str, Any]:
        return await self._request(
            "POST", "/v1/holds", member_id=member_id, idempotency_key=idempotency_key, json={"book_id": book_id}
        )

    async def cancel_hold(self, member_id: str, hold_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/v1/holds/{hold_id}", member_id=member_id)


def _problem(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
        if isinstance(body, dict):
            return body
    except ValueError:
        pass
    return {"code": f"http-{response.status_code}", "detail": response.text[:200] or response.reason_phrase}
