"""HTTP adapter: a versioned REST API (FastAPI) over the domain service.

The API is the system of record and the only component that touches the database. The MCP server, the
Java client and anything else talk to it over HTTP, so all business rules are enforced in one place.
"""
