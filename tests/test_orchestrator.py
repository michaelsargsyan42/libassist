from __future__ import annotations

import json

import pytest

from libassist.agent.guardrails import parse_confirmation, redact
from libassist.agent.llm import AssistantTurn, ToolCall, call, last_tool_json, say
from libassist.agent.orchestrator import INJECTION_NOTICE

pytestmark = pytest.mark.anyio

VERIFY_LEA = call("verify_member", member_id="M-1001", last_name="Martin")


def renewal_count(db, loan_id: str) -> int:
    return db.execute("SELECT renewal_count FROM loans WHERE id = ?", (loan_id,)).fetchone()[0]


async def test_the_model_never_sees_member_id_or_idempotency_key(agent_session):
    async with agent_session([]) as agent:
        params = {t["function"]["name"]: set(t["function"]["parameters"]["properties"]) for t in agent.tool_schemas}
    assert params["verify_member"] == {"member_id", "last_name"}  # the only place a member id comes from
    assert params["get_account"] == set()
    assert params["renew_loan"] == {"loan_id"}
    assert all("idempotency_key" not in p for p in params.values())


async def test_account_tools_are_refused_until_identity_is_verified(agent_session):
    async with agent_session([call("get_account"), say("What is your last name?")]) as agent:
        reply = await agent.handle("I'm M-1001, what do I owe?")
    assert reply.text == "What is your last name?"
    assert agent.trace.executed_tools() == []
    assert agent.trace.of_kind("tool_rejected")[0].data["reason"] == "identity-not-verified"
    assert "not verified" in agent.messages[-2]["content"]  # what the model was told


async def test_failed_verifications_lock_after_three_attempts(agent_session):
    wrong = [call("verify_member", member_id="M-1001", last_name="Smith"), say("That didn't match.")]
    async with agent_session(wrong * 3 + [VERIFY_LEA, say("Locked.")]) as agent:
        for _ in range(4):
            await agent.handle("try again")
    assert agent.verified_member_id is None
    assert agent.trace.executed_tools() == ["verify_member"] * 3
    assert agent.trace.of_kind("tool_rejected")[-1].data["reason"] == "too-many-attempts"


async def test_write_needs_confirmation_then_runs_exactly_once(agent_session, db):
    steps = [
        VERIFY_LEA,
        call("renew_loan", loan_id="L-5001"),
        lambda m: say("New due date: " + last_tool_json(m)["loan"]["due_on"]),
    ]
    async with agent_session(steps) as agent:
        ask = await agent.handle("I'm M-1001, Martin. Renew Designing Data-Intensive Applications.")
        assert ask.needs_confirmation
        assert ask.text == (
            'I\'m about to renew loan L-5001 ("Designing Data-Intensive Applications"). Shall I go ahead? (yes/no)'
        )
        assert renewal_count(db, "L-5001") == 0  # nothing happened yet

        done = await agent.handle("yes")
    assert done.text == "New due date: 2026-03-28"
    assert renewal_count(db, "L-5001") == 1
    executed = [e for e in agent.trace.of_kind("tool_executed") if e.data["tool"] == "renew_loan"]
    assert len(executed) == 1 and executed[0].data["origin"] == "confirmed"
    assert len(executed[0].data["arguments"]["idempotency_key"]) == 32


async def test_saying_no_changes_nothing_and_skips_the_model(agent_session, db):
    async with agent_session([VERIFY_LEA, call("renew_loan", loan_id="L-5001")]) as agent:
        await agent.handle("I'm M-1001, Martin. Renew L-5001.")
        reply = await agent.handle("no")
    assert reply.text == "OK, I haven't changed anything."
    assert renewal_count(db, "L-5001") == 0 and agent.pending is None


async def test_an_unclear_answer_cancels_the_pending_action(agent_session, db):
    steps = [VERIFY_LEA, call("renew_loan", loan_id="L-5001"), say("You owe €0.75.")]
    async with agent_session(steps) as agent:
        await agent.handle("I'm M-1001, Martin. Renew L-5001.")
        reply = await agent.handle("hmm, what do I owe actually?")
    assert reply.text == "OK, I haven't changed anything. You owe €0.75."
    assert renewal_count(db, "L-5001") == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        ("Yes please!", True),
        ("go ahead", True),
        ("no", False),
        ("No! never", False),
        ("don't", False),
        ("maybe", None),
        ("what is my fine?", None),
    ],
)
def test_confirmation_parsing(text, expected):
    assert parse_confirmation(text) is expected


async def test_model_supplied_member_id_is_ignored(agent_session):
    steps = [
        VERIFY_LEA,
        call("get_account", member_id="M-1004"),
        lambda m: say("Account of " + last_tool_json(m)["member"]["id"]),
    ]
    async with agent_session(steps) as agent:
        reply = await agent.handle("I'm M-1001, Martin. Show Tom's account (M-1004).")
    assert reply.text == "Account of M-1001"
    assert agent.trace.of_kind("identity_override_blocked")[0].data["requested"] == "M-1004"


async def test_a_session_cannot_switch_to_another_member(agent_session):
    steps = [VERIFY_LEA, call("verify_member", member_id="M-1004", last_name="Novak"), say("No.")]
    async with agent_session(steps) as agent:
        await agent.handle("I'm M-1001, Martin. Now I'm M-1004, Novak.")
    assert agent.verified_member_id == "M-1001"
    assert agent.trace.of_kind("tool_rejected")[0].data["reason"] == "session-already-bound"


async def test_the_confirmation_text_comes_from_code_not_from_the_model(agent_session, db):
    sneaky = AssistantTurn("Let me just check your account!", [ToolCall("c1", "cancel_hold", {"hold_id": "H-7001"})])
    steps = [call("verify_member", member_id="M-1003", last_name="Rossi"), sneaky]
    async with agent_session(steps) as agent:
        reply = await agent.handle("I'm M-1003, Rossi. Anything new?")
    assert reply.text == 'I\'m about to cancel your hold H-7001 ("Effective Java"). Shall I go ahead? (yes/no)'
    assert db.execute("SELECT status FROM holds WHERE id = 'H-7001'").fetchone()[0] == "QUEUED"


async def test_injected_instructions_in_tool_results_are_flagged(agent_session):
    steps = [call("search_catalog", query="consensus notes"), say("We have the notes.")]
    async with agent_session(steps) as agent:
        await agent.handle("Do you have notes on consensus?")
    assert agent.trace.of_kind("injection_suspected")[0].data["tool"] == "search_catalog"
    tool_message = next(m for m in agent.messages if m["role"] == "tool")
    assert tool_message["content"].startswith(INJECTION_NOTICE)


async def test_only_one_write_action_per_turn(agent_session):
    double = AssistantTurn(
        None, [ToolCall("a", "renew_loan", {"loan_id": "L-5001"}), ToolCall("b", "place_hold", {"book_id": "B-009"})]
    )
    async with agent_session([VERIFY_LEA, double]) as agent:
        reply = await agent.handle("I'm M-1001, Martin. Renew L-5001 and reserve Fluent Python.")
    assert agent.pending.tool == "renew_loan" and "L-5001" in reply.text
    assert agent.messages[-2] == {
        "role": "tool",
        "tool_call_id": "b",
        "content": "Not executed: another action is waiting for confirmation.",
    }


async def test_unknown_tools_and_bad_json_are_reported_back_to_the_model(agent_session):
    steps = [
        AssistantTurn(None, [ToolCall("x", "delete_everything", {})]),
        AssistantTurn(None, [ToolCall("y", "get_book", {}, invalid_arguments="{book_id: B-001")]),
        say("Sorry."),
    ]
    async with agent_session(steps) as agent:
        await agent.handle("hi")
    tool_messages = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert tool_messages[0].startswith("Unknown tool 'delete_everything'")
    assert tool_messages[1].startswith("The arguments were not valid JSON")


async def test_step_budget_hands_off_to_a_human(agent_session):
    looping = [call("search_policies", question="renewals")] * 10
    async with agent_session(looping, max_steps=4) as agent:
        reply = await agent.handle("How do renewals work?")
    assert reply.handoff and "librarian" in reply.text
    assert agent.trace.of_kind("handoff")[0].data["reason"] == "step-budget"
    assert len(agent.trace.executed_tools()) == 4


async def test_a_failing_model_hands_off_instead_of_crashing(agent_session):
    def broken(_messages):
        raise TimeoutError("model endpoint timed out")

    async with agent_session([broken]) as agent:
        reply = await agent.handle("hello")
    assert reply.handoff
    assert "TimeoutError" in agent.trace.of_kind("model_error")[0].data["error"]


async def test_trace_is_redacted_and_serializable(agent_session):
    async with agent_session([VERIFY_LEA, call("get_account"), say("Done.")]) as agent:
        await agent.handle("I'm M-1001, Martin.")
    dump = agent.trace.to_jsonl()
    assert "Martin" not in dump and "lea.martin@example.edu" not in dump
    assert "[redacted]" in dump and "[email]" in dump
    assert all(json.loads(line) for line in dump.splitlines())


def test_redact_handles_nested_structures():
    assert redact({"a": [{"email": "x@y.z"}, "write to bob@example.org"]}) == {
        "a": [{"email": "[redacted]"}, "write to [email]"]
    }
