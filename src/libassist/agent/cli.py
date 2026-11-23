from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

from libassist.agent.llm import LLMError, OpenAICompatibleChatModel
from libassist.agent.orchestrator import Orchestrator
from libassist.agent.toolbox import McpToolbox
from libassist.config import REPO_ROOT, api_base_url
from libassist.devserver import ApiServer, demo_settings

# cli that interacts with user
# mcp host with orchestrator.py, launches the mcp server


# start the mcp server
def mcp_server_params(api_url: str) -> StdioServerParameters:
    src = str(Path(REPO_ROOT) / "src")
    pythonpath = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "libassist.mcp_server.server"],
        env={"LIBASSIST_API_URL": api_url, "PYTHONPATH": pythonpath},
    )


# start chatting
async def chat(model: OpenAICompatibleChatModel, api_url: str) -> None:
    async with Client(mcp_server_params(api_url)) as client:
        agent = await Orchestrator.create(model, McpToolbox(client))
        print(f"LibAssist ({model.name}) - tools: {', '.join(agent.tools)}")
        print(
            "Try: 'How many times can I renew a book?' or "
            "'Hey, my name is Léa Martin, my id is M-1001. Do I owe anything?'\n"
        )
        while True:
            try:
                text = (await anyio.to_thread.run_sync(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/trace":
                print(agent.trace.to_jsonl())
                continue
            reply = await agent.handle(text)
            print(f"libassist> {reply.text}\n")
            if reply.handoff:
                errors = agent.trace.of_kind("model_error")
                reason = agent.trace.of_kind("handoff")[-1].data.get("reason")
                detail = f": {errors[-1].data['error']}" if reason == "model-error" and errors else ""
                print(f"  [handoff reason: {reason}{detail}]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the LibAssist agent.")
    parser.add_argument("--with-api", action="store_true", help="start the REST API with a fresh demo database")
    parser.add_argument("--today", type=date.fromisoformat, help="business date for the demo data (YYYY-MM-DD)")
    args = parser.parse_args()

    try:
        model = OpenAICompatibleChatModel.from_env()
    except LLMError as exc:
        sys.exit(str(exc))

    # for demo stuff
    if args.with_api:
        with ApiServer(demo_settings(args.today)) as server:
            anyio.run(chat, model, server.base_url)
    else:
        anyio.run(chat, model, api_base_url())


if __name__ == "__main__":
    main()
