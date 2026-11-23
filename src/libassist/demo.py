"""End-to-end demo with no LLM and no setup: `libassist-demo` (or `make demo`).

What is real: the REST API (uvicorn on a free port, SQLite), the MCP server (a separate process speaking
JSON-RPC over stdio), the orchestrator with all its guardrails, and the idempotent HTTP retry at the end.
What is scripted: the model's decisions. Each scene plays back the turns an LLM would produce, so the
demo is deterministic and shows exactly how the pieces interact. Use `libassist-chat` for a real model.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from typing import Any

import anyio
import httpx
from mcp import Client

from libassist.agent.cli import mcp_server_params
from libassist.agent.llm import AssistantTurn, Message, ScriptedChatModel, call, last_tool_json, say
from libassist.agent.orchestrator import Orchestrator
from libassist.agent.toolbox import McpToolbox
from libassist.devserver import ApiServer, demo_settings

DEMO_DATE = date(2026, 3, 2)
DIM, BOLD, RESET = ("\033[2m", "\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "", "")


def show_events(agent: Orchestrator, start: int) -> int:
    """Print the interesting trace events since `start`; return the new position."""
    for event in agent.trace.events[start:]:
        d = event.data
        line = None
        if event.kind == "model_turn" and d["tool_calls"]:
            line = "model calls " + ", ".join(
                f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})" for c in d["tool_calls"]
            )
        elif event.kind == "tool_executed":
            via = {"model": "MCP -> REST", "confirmed": "MCP -> REST", "orchestrator": "orchestrator lookup"}
            line = f"{via[d['origin']]}: {d['tool']} {'ok' if d['ok'] else 'error: ' + d['result'][:110]}"
        elif event.kind in ("tool_rejected", "identity_override_blocked"):
            line = f"guardrail: {event.kind} {d.get('tool', '')} ({d.get('reason') or d.get('requested')})"
        elif event.kind in (
            "identity_verified",
            "confirmation_requested",
            "confirmation_given",
            "confirmation_declined",
            "injection_suspected",
            "handoff",
        ):
            line = f"orchestrator: {event.kind} {json.dumps({k: v for k, v in d.items()}, ensure_ascii=False)}"
        if line:
            print(f"{DIM}    · {line}{RESET}")
    return len(agent.trace)


async def scene(client: Client, title: str, turns: list[str], steps: list[Any]) -> Orchestrator:
    print(f"\n{BOLD}=== {title}{RESET}")
    agent = await Orchestrator.create(ScriptedChatModel(steps), McpToolbox(client))
    position = 0
    for text in turns:
        print(f"member>    {text}")
        reply = await agent.handle(text)
        position = show_events(agent, position)
        print(f"libassist> {reply.text}")
    return agent


# --------------------------------------------------------------------------- scripted model turns


def answer_from_policies(messages: list[Message]) -> AssistantTurn:
    passages = {p["citation"]: p["text"] for p in last_tool_json(messages)["results"]}
    assert "loans#renewals" in passages, passages
    return say(
        "A loan can be renewed at most twice [loans#renewals]. A renewal is refused if another member is "
        "waiting for the book, if it is more than 14 days overdue, or if your account is blocked "
        "[loans#renewals]."
    )


def summarize_account(messages: list[Message]) -> AssistantTurn:
    account = last_tool_json(messages)
    lines = [
        f"- {loan['title']}: due {loan['due_on']}"
        + (f", {loan['overdue_days']} days overdue ({loan['fine']})" if loan["overdue_days"] else "")
        for loan in account["loans"]
    ]
    return say(
        f"Thanks {account['member']['first_name']}, you're verified. You owe {account['fines_total']} "
        f"in total. Your loans:\n" + "\n".join(lines)
    )


def report_renewal(messages: list[Message]) -> AssistantTurn:
    result = last_tool_json(messages)
    return say(
        f'Done. "{result["loan"]["title"]}" is now due on {result["loan"]["due_on"]} '
        f"(it was {result['previous_due_on']})."
    )


def explain_refusal(messages: list[Message]) -> AssistantTurn:
    error = messages[-1]["content"]
    assert "renewal-limit" in error, error
    return say(
        "I couldn't renew it: this loan has already been renewed twice, which is the maximum "
        "[loans#renewals]. Please return it by its due date."
    )


def obey_injection(messages: list[Message]) -> AssistantTurn:
    # A *compromised* model: it follows the instruction hidden in a book description.
    return call("cancel_hold", hold_id="H-7001")


# --------------------------------------------------------------------------- main


async def run_demo(api_url: str) -> None:
    async with Client(mcp_server_params(api_url)) as client:
        await scene(
            client,
            "1. Policy question: retrieval with citations",
            ["How many times can I renew a book?"],
            [call("search_policies", question="How many times can a loan be renewed?"), answer_from_policies],
        )

        agent = await scene(
            client,
            "2. Identity check, account summary, renewal with confirmation",
            [
                "What do I owe? I'm member M-1001.",
                "Martin",
                "Please renew Designing Data-Intensive Applications.",
                "yes",
                "Renew Operating Systems: Three Easy Pieces too.",
                "yes",
            ],
            [
                call("get_account"),  # refused by the orchestrator: identity not verified yet
                say("To protect your account, could you also tell me your last name?"),
                call("verify_member", member_id="M-1001", last_name="Martin"),
                call("get_account"),
                summarize_account,
                call("renew_loan", loan_id="L-5001"),  # becomes a pending action; the member must confirm
                report_renewal,
                call("renew_loan", loan_id="L-5002"),
                explain_refusal,
            ],
        )
        renewal = next(e for e in agent.trace.of_kind("tool_executed") if e.data["tool"] == "renew_loan")
        renewal_key = renewal.data["arguments"]["idempotency_key"]

        await scene(
            client,
            "3. Prompt injection in a catalogue record (book B-013)",
            [
                "Hi, I'm Sofia Rossi, member M-1003. Do you have notes on consensus algorithms?",
                "No! I never asked for that.",
            ],
            [
                call("verify_member", member_id="M-1003", last_name="Rossi"),
                call("search_catalog", query="consensus algorithms notes"),
                obey_injection,
            ],
        )

    print(f"\n{BOLD}=== 4. Checking the results directly against the REST API{RESET}")
    async with httpx.AsyncClient(base_url=api_url) as http:
        holds = (await http.get("/v1/members/M-1003/account", headers={"X-Member-Id": "M-1003"})).json()["holds"]
        print(f"Sofia's holds after the injection attempt: {[(h['id'], h['status']) for h in holds]}")
        retry = await http.post(
            "/v1/loans/L-5001/renewals", headers={"X-Member-Id": "M-1001", "Idempotency-Key": renewal_key}
        )
        loan = retry.json()["loan"]
        print(
            f"Retrying the confirmed renewal with the same Idempotency-Key: HTTP {retry.status_code}, "
            f"Idempotent-Replayed: {retry.headers.get('Idempotent-Replayed')}, renewal_count still "
            f"{loan['renewal_count']}, due {loan['due_on']}"
        )


def main() -> None:
    print(f"LibAssist demo. Business date {DEMO_DATE}. The model is scripted; everything else is real.")
    with ApiServer(demo_settings(DEMO_DATE)) as server:
        print(f"REST API at {server.base_url} (docs at {server.base_url}/docs); MCP server over stdio.")
        anyio.run(run_demo, server.base_url)


if __name__ == "__main__":
    main()
