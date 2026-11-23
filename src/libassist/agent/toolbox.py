from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from mcp import Client
from mcp.types import ToolAnnotations

from libassist.agent.guardrails import hidden_params_for, is_read_only


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: ToolAnnotations | None

    @property
    def parameters(self) -> set[str]:
        return set(self.input_schema.get("properties", {}))

    @property
    def read_only(self) -> bool:
        return is_read_only(self.name, self.annotations)

    def for_model(self) -> dict[str, Any]:
        """The tool as the LLM sees it: hidden parameters removed from the schema."""
        hidden = hidden_params_for(self.name)
        schema = copy.deepcopy(self.input_schema)
        schema["properties"] = {k: v for k, v in schema.get("properties", {}).items() if k not in hidden}
        schema["required"] = [r for r in schema.get("required", []) if r not in hidden]
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    text: str  # llm reads this
    data: dict[str, Any] | None  # for orchestrator


class McpToolbox:
    def __init__(self, client: Client) -> None:
        self.client = client

    async def list_tools(self) -> list[ToolSpec]:
        result = await self.client.list_tools()
        return [ToolSpec(t.name, t.description or "", t.input_schema, t.annotations) for t in result.tools]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        try:
            result = await self.client.call_tool(name, arguments)
        except Exception as exc:  # transport failure, server crash...: report it to the model, don't crash
            return ToolOutcome(False, f"Tool {name} could not be called ({type(exc).__name__}).", None)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return ToolOutcome(not result.is_error, text, result.structured_content)
