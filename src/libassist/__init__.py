"""LibAssist: a small, tested reference implementation of an LLM agent.

The agent never touches the database. It talks to an MCP server, which is a
thin adapter over a REST API, which owns the data and enforces the rules:

    agent (orchestrator + LLM)  ->  MCP server (tools)  ->  REST API  ->  SQLite
                                                       \\->  RAG index over policy documents

See docs/ARCHITECTURE.md for the full picture.
"""

__version__ = "0.1.0"
