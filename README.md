# Forge

**Forge** is an autonomous, MCP-powered coding agent for your terminal. Give it a task in plain English. It
reads and edits your files, runs commands, looks up documentation, and keeps going until the job is done.
Every tool call is shown as it happens.

> Status: agent core complete (M2). The custom RAG server (M3) and the evaluation (M4) are in progress.
> Planning documents: [docs/planning](docs/planning).

## Features
- **Agentic loop**: reason → call tools → observe → repeat, with a step budget and recovery from errors
- **MCP client**: launches every server in `mcp_servers.json` and discovers their tools at runtime
  - filesystem (`@modelcontextprotocol/server-filesystem`), sandboxed to the workspace
  - Context7 (`@upstash/context7-mcp`), current docs for third-party libraries
- **Native shell tool**: `run_command` for tests, scripts, git, and so on
- **Providers**: Ollama (local) and any OpenAI-compatible llama.cpp server, switchable mid-session with `/model`
- **Confirm or auto mode**: in confirm mode, tools that change things show a preview first (a real **diff** for
  edits) and ask `[Y]es / [n]o / [a]lways`
- **Streaming terminal UI**: live Markdown output, a thinking indicator, tool-call and result panels, Ctrl+C to cancel

## Setup

Requirements: Python 3.11+, Node.js 18+ (the MCP servers run via `npx`), and [Ollama](https://ollama.com) for local models.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # optional: adds the `forge` command
copy .env.example .env            # then fill in the values you need
ollama pull gemma4:e2b            # or set OLLAMA_MODEL to a model you already have
```

To use the hosted llama.cpp model, set `LLAMACPP_BASE_URL` (and `LLAMACPP_API_KEY`) in `.env`, then run `/model llamacpp`.

## Usage

```bash
forge -w path/to/project                 # interactive REPL (or: python -m forge)
forge -w sandbox --auto "fix the bug in calculator.py and run the tests"   # one-shot task
forge -p llamacpp                        # choose the provider at startup
```

| Command | |
|---|---|
| `/mode [confirm\|auto]` | ask before mutating tools, or run everything |
| `/model [provider[:model]]` | list providers or switch, e.g. `/model ollama:qwen3:8b` |
| `/tools [server]`, `/servers` | what the agent can call |
| `/clear`, `/help`, `/exit` | |

`sandbox/` is a small sample project for trying Forge safely. `python mcp_demo.py --demo` runs the
milestone-1 MCP walkthrough without an LLM.

## Project layout

| Path | Purpose |
|---|---|
| `forge/cli.py` | REPL, slash commands, streaming output, confirmation prompts |
| `forge/agent.py` | the agentic loop |
| `forge/providers.py` | provider abstraction (LangChain chat models) |
| `forge/tools.py` | tool registry: MCP + native tools, permissions, previews, truncation |
| `forge/mcp_client.py` | MCP client manager (stdio sessions, tool discovery, routing) |
| `forge/prompts.py` | system prompt with a workspace snapshot |
| `mcp_servers.json` | MCP server definitions |
| `docs/planning/` | plan, architecture, state and sequence diagrams |
