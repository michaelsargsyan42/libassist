"""Domain layer: records, business rules (policy) and use cases (service).

Nothing in here knows about HTTP, MCP or LLMs. That is the point: the rules are enforced in one place,
in plain code that unit tests can exercise directly (ports-and-adapters / hexagonal style).
"""
