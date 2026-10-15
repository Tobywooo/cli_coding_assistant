# Forge: Project Plan

**Forge** is an autonomous command-line coding agent. Given a task in plain English, it reasons about what to
do, acts on the local codebase through tools, observes the results, and iterates until the job is done.

## 1. Goals

| # | Requirement (from the brief) | How Forge meets it |
|---|---|---|
| 1 | Autonomous agentic loop | `forge/agent.py`: an LLM → tool calls → observations → LLM loop with a step budget |
| 2 | LLM tool calling (files, shell, search, retrieval) | MCP tools (filesystem, Context7, RAG) plus a native `run_command` tool, all exposed as function-calling schemas |
| 3 | MCP client, multiple servers, dynamic tool loading | `forge/mcp_client.py` launches every server in `mcp_servers.json` and discovers tools with `list_tools` at startup |
| 4 | Custom RAG MCP server | `rag_server/`: an MCP server over the **RAGAS** documentation, backed by a persistent Chroma vector store |
| 5 | Advanced RAG technique | **Fusion retrieval** (NirDiamant/RAG_Techniques): BM25 keyword search + dense vector search, scores normalised and fused |
| 6 | Provider abstraction: Ollama + a cloud/hosted provider | `forge/providers.py`: LangChain chat models behind one factory: **Ollama** (local) and a **llama.cpp** OpenAI-compatible endpoint (Qwen3.8-27B) |
| 7 | Interactive CLI: streaming, visible tool calls, confirm/auto modes | `forge/cli.py`: a prompt_toolkit REPL with Rich rendering, a streaming answer, tool-call and result panels, and `/mode confirm\|auto` |
| 8 | Architecture documentation | This folder: architecture, state and sequence diagrams (Mermaid, rendered on GitHub) |
| 9 | Evaluation | `rag_server/evaluate.py` (retrieval hit-rate/MRR: vector vs BM25 vs fusion) and `scripts/compare_models.py` (same coding task, each provider) |
| 10 | Collaboration | Git history: planning first, then one commit per milestone |

## 2. Design decisions

### 2.1 Build our own agent loop, but use LangChain for model abstraction
LangChain chat models (`ChatOllama`, `ChatOpenAI`) give us one interface for streaming and `bind_tools` across
providers, which is the hard, provider-specific part. We deliberately do **not** use a prebuilt agent executor:
writing the loop ourselves lets us add confirmation prompts, diff previews, step limits and tool-call
visibility exactly where we want them, and it is the learning objective.

### 2.2 Write our own MCP client instead of using an adapter library
`langchain-mcp-adapters` would hide the protocol. Our `MCPClientManager` uses the official MCP Python SDK directly.
It launches the servers over stdio, runs the `initialize` handshake, calls `list_tools`, routes `call_tool`, and
gives every tool a qualified name (`server__tool`) so tools from different servers can't collide.
A small converter turns each MCP JSON Schema into an OpenAI-style function definition for the LLM.

### 2.3 Servers are configuration, not code
`mcp_servers.json` lists each server's launch command, arguments, environment and trust level. Adding a server
(e.g. the RAG server) needs no Python changes. `${VAR}` placeholders are filled from `.env`.

### 2.4 Permission model
Every tool is classified as **read-only** or **mutating**:
- MCP servers that publish `annotations.readOnlyHint` (the filesystem server does) are classified by that hint.
- Servers that are read-only by nature (Context7, RAG) are marked `"read_only": true` in the config.
- The native `run_command` tool is always mutating.

| Mode | Read-only tools | Mutating tools |
|---|---|---|
| `confirm` (default) | run immediately | ask first: **y**es / **n**o / **a**lways allow this tool this session |
| `auto` | run immediately | run immediately |

Before asking about `edit_file`, Forge calls it with `dryRun: true` and shows the **diff**. A denied call is
returned to the LLM as a tool result ("user declined"), so the model can adapt instead of crashing.

### 2.5 Shell commands are a native tool
None of the three required servers runs shell commands, so Forge has one built-in tool, `run_command`. It runs in
the workspace directory, has a timeout, and returns exit code + stdout/stderr, truncated.

### 2.6 RAG server: RAGAS docs with fusion retrieval
- **Corpus:** the Markdown docs from the `explodinggradients/ragas` repo (sparse git checkout of `docs/`). The
  library is small enough for quick ingestion, has a real API that LLMs often get wrong, and doesn't overlap with Context7's
  LangChain coverage in our demos.
- **One-time ingestion** (`python -m rag_server.ingest`): load → split on Markdown headers → recursive split to
  about 1,000 characters with overlap → embed (Chroma's built-in ONNX `all-MiniLM-L6-v2`, fully local, no PyTorch)
  → persist in `rag_server/data/chroma`. Later runs reuse the store.
- **Advanced technique: fusion retrieval.** Dense embeddings capture meaning but miss exact identifiers
  (`LLMContextPrecisionWithReference`, `SingleTurnSample`). BM25 catches exact terms but misses paraphrases. The server
  scores every candidate with both, min-max normalises each score, and ranks by
  `alpha * vector + (1 - alpha) * bm25` (alpha = 0.5). A `method` argument (`fusion` | `vector` | `bm25`) makes
  the comparison visible live and in the evaluation.
- **Exposed as an MCP tool**, `search_ragas_docs(query, k, method)`, returning chunks with source file, section
  heading and scores. The agent decides when to call it.

### 2.7 Context management
Tool outputs (especially docs) can be thousands of characters. Results are truncated before going back to the LLM
(the CLI still shows a preview), and Ollama runs with a larger context window (`num_ctx`), because the default is
too small for multi-step tasks.

## 3. Components

```
forge/
  __main__.py      entry point (`python -m forge`)
  cli.py           REPL, slash commands, streaming output, confirmation prompts
  agent.py         agentic loop; emits events (text, tool call, tool result) to the UI
  providers.py     provider registry and chat-model factory (Ollama, llama.cpp)
  tools.py         ToolRegistry: MCP tools + native tools, schemas, permissions, execution
  mcp_client.py    MCP client manager (from milestone 1)
  config.py        settings and mcp_servers.json loading
  prompts.py       system prompt
  display.py       Rich rendering helpers
rag_server/
  ingest.py        one-time ingestion pipeline
  retrieval.py     vector, BM25 and fusion retrievers
  server.py        MCP server exposing search_ragas_docs
  evaluate.py      retrieval evaluation (hit@k, MRR)
scripts/
  compare_models.py  runs one coding task against each provider and records metrics
```

## 4. Milestones

| Milestone | Scope | Exit criteria |
|---|---|---|
| M0 Planning | this document, architecture/state/sequence diagrams | committed before any implementation |
| M1 MCP integration | MCP client, filesystem + Context7 servers, demo CLI | tools discovered dynamically and invoked from the CLI (presented) |
| M2 Agent core | providers, tool registry, agentic loop, permission modes | Forge completes "fix the bug and run the tests" with Ollama |
| M3 RAG server | ingestion, fusion retrieval, MCP server, wired into Forge | agent calls `search_ragas_docs` unprompted for a RAGAS question |
| M4 Evaluation | retrieval evaluation, model comparison harness | results tables for the report |
| M5 Polish and deliverables | README, diagrams for the report, demo video, PDF report | submitted |

## 5. Risks

| Risk | Mitigation |
|---|---|
| Small local models produce malformed tool calls or loop forever | step budget, schema-validated arguments, errors returned to the model as observations |
| Local inference is slow (first call ~60 s while the model loads) | keep the model loaded; compare against the hosted Qwen endpoint |
| MCP SDK v2 differs from tutorials | pin versions; check behaviour against the installed SDK |
| Destructive edits | confirm mode by default, filesystem server sandboxed to the workspace, diff preview |
| Large tool outputs overflow the context | truncate observations; tune `num_ctx` |

## 6. Evaluation plan

1. **Retrieval:** about 20 RAGAS questions, each labelled with the doc page that answers it. Report hit@1, hit@5 and MRR for
   vector-only, BM25-only and fusion retrieval.
2. **Models:** the same multi-step coding task (fix a bug, add tests, run them) in `auto` mode for each provider.
   Record success, number of steps and tool calls, wall-clock time, tokens, and notable failure modes.
