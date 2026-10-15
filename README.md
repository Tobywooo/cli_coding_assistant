# Forge

**Forge** is an autonomous, MCP-powered coding agent for your terminal. Give it a task in plain English. It
reads and edits your files, runs commands, looks up documentation, and keeps going until the job is done.
Every tool call is shown as it happens.

> Planning documents: [docs/planning](docs/planning) · as-built architecture: [docs/architecture_as_built.md](docs/architecture_as_built.md)
> · evaluation: [docs/evaluation](docs/evaluation) · report draft: [docs/report_draft.md](docs/report_draft.md)

## Features
- **Agentic loop**: reason → call tools → observe → repeat, with a step budget and recovery from errors
- **MCP client**: launches every server in `mcp_servers.json` and discovers their tools at runtime
  - filesystem (`@modelcontextprotocol/server-filesystem`), sandboxed to the workspace
  - Context7 (`@upstash/context7-mcp`), current docs for third-party libraries
  - **Forge RAG server** (`rag_server/`, ours): **fusion retrieval** (BM25 + vector search) over the RAGAS docs
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
python -m rag_server.ingest       # one-time: fetch RAGAS docs, chunk, embed, store (~30 s)
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

`sandbox/` is a small sample project for trying Forge safely. `python mcp_demo.py --demo` walks through all
three MCP servers' tools without an LLM.

## Evaluation

```bash
python -m rag_server.evaluate                          # fusion vs vector vs BM25 vs RRF -> docs/evaluation/rag_retrieval.md
python scripts/compare_models.py -p ollama llamacpp -n 3   # same tasks per model -> docs/evaluation/model_comparison.md
```

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
| `rag_server/` | custom RAG MCP server: `ingest.py` (one-time), `retrieval.py` (fusion), `server.py`, `evaluate.py` |
| `scripts/compare_models.py` | model comparison benchmark with automatic checks |
| `docs/planning/` | plan, architecture, state and sequence diagrams |
