from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio
import httpx

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    invalid_arguments: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    name: str

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn: ...


class LLMError(RuntimeError):
    pass


class OpenAICompatibleChatModel:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        temperature: float | None = 0.0,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.model = model
        self.name = f"openai-compatible:{model}"
        self.temperature = temperature
        self.max_retries = max_retries

    @classmethod
    def from_env(cls) -> OpenAICompatibleChatModel:
        base_url, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
        if not base_url or not model:
            raise LLMError("Set LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY if needed). See .env.example.")
        return cls(base_url, model, os.environ.get("LLM_API_KEY") or None, temperature=_temperature_from_env(base_url))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"LLM endpoint unreachable: {exc!r}") from exc
            else:
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    break
            await anyio.sleep(0.5 * 2**attempt)
        if response.status_code >= 400:
            if os.environ.get("LLM_DEBUG"):
                with open("llm_debug.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {"request": payload, "status": response.status_code, "response": response.text},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            raise LLMError(f"LLM call failed: HTTP {response.status_code} {response.text[:300]}")
        return parse_openai_message(response.json()["choices"][0]["message"])


def _temperature_from_env(base_url: str) -> float | None:
    raw = os.environ.get("LLM_TEMPERATURE")
    if raw is None:
        return None if "anthropic.com" in base_url else 0.0
    if raw.strip().lower() in ("", "none", "default"):
        return None
    return float(raw)


def parse_openai_message(message: dict[str, Any]) -> AssistantTurn:
    calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function", {})
        text = function.get("arguments") or "{}"
        extra = {k: v for k, v in raw.items() if k not in ("id", "type", "function")} or None
        try:
            arguments = json.loads(text)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            calls.append(ToolCall(raw.get("id", ""), function.get("name", ""), arguments, extra=extra))
        except ValueError:
            calls.append(ToolCall(raw.get("id", ""), function.get("name", ""), {}, invalid_arguments=text, extra=extra))
    return AssistantTurn(message.get("content"), calls)


# gemini needs a thought signature on tools it didn't make keeps it and warns the answer quality may be reduced
SYNTHETIC_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def extra_for_synthetic_call(original: dict[str, Any] | None) -> dict[str, Any] | None:
    """Provider fields for a tool call written by code, based on the model call it stands in for."""
    google = ((original or {}).get("extra_content") or {}).get("google") or {}
    if "thought_signature" in google:
        return {"extra_content": {"google": {"thought_signature": SYNTHETIC_THOUGHT_SIGNATURE}}}
    return None


# for demo

Step = AssistantTurn | Callable[[list[Message]], AssistantTurn]


def say(text: str) -> AssistantTurn:
    return AssistantTurn(text)


def call(name: str, call_id: str | None = None, **arguments: Any) -> AssistantTurn:
    return AssistantTurn(None, [ToolCall(call_id or f"call_{name}", name, arguments)])


def last_tool_result(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message["role"] == "tool":
            return str(message["content"])
    return ""


def last_tool_json(messages: list[Message]) -> dict[str, Any]:
    """The most recent tool result parsed as JSON (the orchestrator may prefix a one-line note)."""
    text = last_tool_result(messages)
    start = text.find("{")
    return json.loads(text[start:]) if start >= 0 else {}


class ScriptedChatModel:
    """Returns the scripted steps in order. A step can be a function of the conversation so far."""

    name = "scripted"

    def __init__(self, steps: list[Step]) -> None:
        self._steps = list(steps)
        self.calls = 0

    async def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        self.calls += 1
        if not self._steps:
            raise AssertionError("ScriptedChatModel ran out of steps")
        step = self._steps.pop(0)
        return step(messages) if callable(step) else step

    @property
    def remaining(self) -> int:
        return len(self._steps)
