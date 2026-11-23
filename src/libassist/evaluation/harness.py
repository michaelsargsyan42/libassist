"""Evaluation harness: run conversation scenarios against the agent and check the *outcomes*.

Unit tests check the deterministic parts. Evals measure the probabilistic part: does this model, with this
prompt and these tools, actually do the right thing? Three principles:

1. Check outcomes, not wording. The strongest checks are on state: SQL against the database after the
   conversation ("renewal_count is 1"). Then on behaviour: which tools ran, which were not even attempted.
   Text checks are loose regexes ("mentions a waiting list"), never exact sentences.
2. Run each scenario k times and report pass^k: the scenario counts as passed only if ALL k runs pass.
   An agent that renews the right loan 4 times out of 5 is not reliable, and pass^k makes that visible.
3. Every run starts from a fresh database, so runs are independent and repeatable.

The harness wires the stack in-process (API through an ASGI transport, MCP server through the SDK's
in-memory client) so a run takes milliseconds plus the model's latency.
"""

from __future__ import annotations

import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import yaml
from mcp import Client

from libassist.agent.llm import ChatModel
from libassist.agent.orchestrator import Orchestrator, Trace
from libassist.agent.toolbox import McpToolbox
from libassist.api.main import create_app
from libassist.db.database import connect
from libassist.devserver import demo_settings
from libassist.mcp_server.api_client import LibraryApiClient
from libassist.mcp_server.server import create_server

EVAL_DATE = date(2026, 3, 2)


@dataclass(frozen=True)
class StateCheck:
    sql: str
    equals: Any


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    turns: list[str]
    reply_contains: list[str] = field(default_factory=list)  # substrings, case-insensitive
    reply_matches: list[str] = field(default_factory=list)  # regexes, case-insensitive
    reply_excludes: list[str] = field(default_factory=list)
    tools_executed: list[str] = field(default_factory=list)  # tools the model must have run
    tools_not_executed: list[str] = field(default_factory=list)
    tools_not_attempted: list[str] = field(default_factory=list)  # the model must not even propose these
    state: list[StateCheck] = field(default_factory=list)


@dataclass(frozen=True)
class RunResult:
    scenario_id: str
    passed: bool
    failures: list[str]
    replies: list[str]
    executed_tools: list[str]
    seconds: float


@dataclass(frozen=True)
class ScenarioReport:
    scenario: Scenario
    runs: list[RunResult]

    @property
    def passes(self) -> int:
        return sum(r.passed for r in self.runs)

    @property
    def pass_all(self) -> bool:
        """pass^k: every one of the k runs passed."""
        return self.passes == len(self.runs)


ModelFactory = Callable[[Scenario], ChatModel]


def load_scenarios(path: Path) -> list[Scenario]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    scenarios = []
    for item in raw["scenarios"]:
        state = [StateCheck(**check) for check in item.pop("state", [])]
        scenarios.append(Scenario(**item, state=state))
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario ids must be unique")
    return scenarios


def check_run(scenario: Scenario, replies: list[str], trace: Trace, db_path: Path) -> list[str]:
    failures: list[str] = []
    text = "\n".join(replies)
    for expected in scenario.reply_contains:
        if expected.lower() not in text.lower():
            failures.append(f"reply should contain {expected!r}")
    for pattern in scenario.reply_matches:
        if not re.search(pattern, text, re.IGNORECASE):
            failures.append(f"reply should match /{pattern}/")
    for forbidden in scenario.reply_excludes:
        if forbidden.lower() in text.lower():
            failures.append(f"reply must not contain {forbidden!r}")

    executed = trace.executed_tools(origin="model") + trace.executed_tools(origin="confirmed")
    attempted = trace.attempted_tools()
    failures += [f"tool {t} should have run" for t in scenario.tools_executed if t not in executed]
    failures += [f"tool {t} must not run" for t in scenario.tools_not_executed if t in executed]
    failures += [f"tool {t} must not even be attempted" for t in scenario.tools_not_attempted if t in attempted]

    conn = connect(db_path)
    try:
        for check in scenario.state:
            value = conn.execute(check.sql).fetchone()[0]
            if value != check.equals:
                failures.append(f"state: `{check.sql}` returned {value!r}, expected {check.equals!r}")
    finally:
        conn.close()
    return failures


async def run_once(scenario: Scenario, model_factory: ModelFactory, *, today: date = EVAL_DATE) -> RunResult:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="libassist-eval-") as tmp:
        settings = demo_settings(today, Path(tmp))
        app = create_app(settings)
        server = create_server(
            lambda: LibraryApiClient("http://libassist.test", transport=httpx.ASGITransport(app=app))
        )
        replies: list[str] = []
        async with Client(server) as client:
            agent = await Orchestrator.create(model_factory(scenario), McpToolbox(client))
            for turn in scenario.turns:
                replies.append((await agent.handle(turn)).text)
        failures = check_run(scenario, replies, agent.trace, settings.db_path)
    return RunResult(
        scenario.id, not failures, failures, replies, agent.trace.executed_tools(), time.perf_counter() - started
    )


async def run_suite(
    scenarios: list[Scenario], model_factory: ModelFactory, *, k: int = 1, today: date = EVAL_DATE
) -> list[ScenarioReport]:
    reports = []
    for scenario in scenarios:
        runs = [await run_once(scenario, model_factory, today=today) for _ in range(k)]
        reports.append(ScenarioReport(scenario, runs))
    return reports


def format_report(reports: list[ScenarioReport], k: int) -> str:
    width = max(len(r.scenario.id) for r in reports)
    lines = [f"{'scenario'.ljust(width)}  passes  pass^{k}", "-" * (width + 17)]
    for report in reports:
        lines.append(
            f"{report.scenario.id.ljust(width)}  {report.passes}/{len(report.runs)}     "
            f"{'PASS' if report.pass_all else 'FAIL'}"
        )
        for run in report.runs:
            for failure in run.failures:
                lines.append(f"    - {failure}")
    passed = sum(r.pass_all for r in reports)
    lines.append(f"\npass^{k}: {passed}/{len(reports)} scenarios")
    return "\n".join(lines)
