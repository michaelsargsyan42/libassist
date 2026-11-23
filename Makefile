# Common tasks. Run `make help` for the list.
PY ?= python
TODAY ?= 2026-03-02

.PHONY: help install test test-fast lint format api mcp demo chat evals java diagrams clean

help:  ## show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## install the package in editable mode, with dev tools
	$(PY) -m pip install -e ".[dev]"

test:  ## run every test (unit, API, MCP, agent, concurrency, end-to-end, Java client if a JDK exists)
	$(PY) -m pytest

test-fast:  ## run everything except the end-to-end tests
	$(PY) -m pytest -m "not e2e"

lint:  ## lint and check formatting
	ruff check src tests evals
	ruff format --check src tests evals

format:  ## format the code and apply safe lint fixes
	ruff format src tests evals
	ruff check --fix src tests evals

api:  ## start the REST API on http://127.0.0.1:8000 (docs at /docs)
	LIBASSIST_TODAY=$(TODAY) libassist-api

mcp:  ## start the MCP server on stdio (expects the API at LIBASSIST_API_URL)
	libassist-mcp

demo:  ## scripted end-to-end demo: real HTTP, real MCP over stdio, no LLM needed
	libassist-demo

chat:  ## chat with the agent (needs LLM_BASE_URL and LLM_MODEL); starts its own API
	libassist-chat --with-api --today $(TODAY)

evals:  ## run the eval suite against a real model, 3 runs per scenario (pass^3)
	$(PY) evals/run_evals.py --k 3

java:  ## run the Java client against a running API (make api in another terminal)
	java clients/java/LibraryClient.java http://127.0.0.1:8000

diagrams:  ## re-render docs/images/*.png from docs/diagrams/*.mmd (needs @mermaid-js/mermaid-cli)
	for f in docs/diagrams/*.mmd; do mmdc -i $$f -o docs/images/$$(basename $${f%.mmd}).png -b white -s 2 -w 1400; done

clean:  ## remove caches, build output and local databases
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info libassist.db libassist.db-wal libassist.db-shm
