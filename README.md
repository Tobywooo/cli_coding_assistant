# CLI Coding Assistant (working name TBD)

An autonomous command-line coding agent built on LLM tool calling and the Model Context Protocol (MCP).

> **Status: milestone 1, MCP integration.** The MCP client layer is working and connects to the
> **filesystem** server and the **Context7** external-resource server. The agentic loop, LLM providers
> and the custom RAG server are not built yet. See [docs/milestone1_progress.md](docs/milestone1_progress.md).

## Setup

Requirements: Python 3.11+ and Node.js 18+ (the MCP servers are launched with `npx`).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # optional: add a CONTEXT7_API_KEY for higher rate limits
```

## Running the MCP demo

```bash
python mcp_demo.py --demo         # guided walkthrough, pauses between steps
python mcp_demo.py                # interactive shell: servers / tools / schema / call / demo
```

The filesystem server is sandboxed to `./sandbox` by default (`--workspace <dir>` to change it).
Server stderr is written to `logs/<server>.log` instead of the terminal.

## Project layout

| Path | Purpose |
|---|---|
| `mcp_servers.json` | Which MCP servers to launch. Add a server here and its tools load automatically |
| `assistant/config.py` | Loads the server config, expanding `${VAR}` placeholders from the environment |
| `assistant/mcp_client.py` | `MCPClientManager`: connects to every server, discovers tools, routes tool calls |
| `assistant/display.py` | Rich rendering of servers, tools, tool calls and results |
| `mcp_demo.py` | Milestone 1 demo CLI |
| `sandbox/` | Small sample codebase the filesystem server is allowed to touch |
