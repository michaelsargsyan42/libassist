"""MCP adapter: exposes the library REST API as Model Context Protocol tools, resources and prompts.

MCP standardizes how an AI application (the *host*, with an MCP *client* inside) discovers and calls
capabilities offered by a *server*. Any MCP host (this project's agent, Claude Desktop, an IDE...) can use
this server without custom integration code. The server holds no data and no business rules: it
translates tool calls into HTTP calls and HTTP problems into readable tool errors.
"""
