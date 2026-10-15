# Forge: Project Report (draft)

> Draft for the PDF report. Every number comes from `docs/evaluation/`. Sections marked **TODO (team)**
> need the llama.cpp results or the team's own reflections before submission.

## 1. System overview

Forge is an autonomous command-line coding agent. The user types a task. Forge sends it, with the schemas of every
available tool, to an LLM, executes the tool calls the model makes, feeds the results back, and repeats until the
model gives a final answer (or a 25-step budget runs out). Every tool call is shown in the terminal. In `confirm`
mode, any tool that changes something shows a preview first (a real diff for edits) and asks y / n / always.

**Components:** CLI (prompt_toolkit + Rich) → agentic loop → provider abstraction (LangChain) and tool registry →
MCP client manager → three MCP servers (filesystem, Context7, our RAG server) plus one native `run_command` tool.
Diagrams: [planning/02_architecture.md](planning/02_architecture.md) (original),
[architecture_as_built.md](architecture_as_built.md) (final, with changes),
[planning/03_state_diagram.md](planning/03_state_diagram.md),
[planning/04_sequence_diagrams.md](planning/04_sequence_diagrams.md) (4 end-to-end operations).

## 2. Design decisions

1. **Our own agent loop; LangChain only for models.** LangChain's chat models give one streaming + `bind_tools`
   interface for Ollama and OpenAI-compatible servers. We wrote the loop ourselves so confirmation, previews,
   cancellation and tool visibility sit exactly where we need them.
2. **Our own MCP client on the official SDK** (not an adapter library): stdio sessions, `list_tools` discovery,
   qualified tool names (`server__tool`), JSON Schema → function-calling schema conversion.
3. **Servers are configuration.** `mcp_servers.json` defines all three servers; the RAG server was added without
   code changes.
4. **Permissions come from MCP annotations.** The filesystem server marks each tool `readOnlyHint`. Docs servers
   are trusted as read-only in the config. `run_command` always counts as mutating.
5. **RAG: RAGAS docs + fusion retrieval.** 133 pages → 1,585 chunks (Markdown-header split, then 1,200 chars with
   150 overlap), MiniLM ONNX embeddings in a persistent Chroma store built once by `python -m rag_server.ingest`.
   Fusion = 0.5 × normalised vector score + 0.5 × normalised BM25 score.
6. **The loop is UI-agnostic.** It reports through callbacks, so the same loop runs the CLI and the headless
   benchmark.

## 3. LLM comparison

Same three tasks, fresh fixture workspace per run, auto mode, all MCP servers connected, scored by **automatic
checks** (hidden tests, file inspection, tool traces), not by the model's summary. Full table and per-run traces:
[evaluation/model_comparison.md](evaluation/model_comparison.md).

| Task | gemma4:e2b (Ollama, local) | Qwen3.8-27B (llama.cpp) |
|---|---|---|
| bugfix: fix divide-by-zero, write and run tests | 3/3 success · 5 steps · 11 s | **TODO (team)** |
| rag_codegen: RAGAS Faithfulness script from docs | 3/3 success · 3 steps · 13 s | **TODO (team)** |
| feature: stricter slugify, tests in new tests/ folder | 0/3 in the recorded batch (2/6 in two earlier batches) · ~11 steps · 66 s | **TODO (team)** |

To fill in the Qwen column: set `LLAMACPP_BASE_URL` / `LLAMACPP_API_KEY` in `.env`, then run
`python scripts/compare_models.py -p llamacpp -n 3`.

**Observations so far (Gemma 4 E2B):**
- It reliably handles short, single-file tasks and uses tools correctly: it reads before editing, creates
  folders before writing, and runs the tests itself.
- **The typical failure is getting stuck in edit_file retries.** On the feature task, the first edit works, but
  when the new tests fail it sends `edit_file` calls whose `oldText` no longer matches the file
  ("Could not find exact match"). It retries, sometimes re-reading the file first, but the replacement text still
  doesn't match, and it stops with its tests still failing. This is why the benchmark scores runs with hidden tests
  instead of trusting the model's own report.
- Results vary from run to run on the hard task (0/3 vs 2/6 across batches), so we report several runs.
- The workspace snapshot in the system prompt mattered most: without it, Gemma invented file paths.

**TODO (team):** add the Qwen observations, e.g. success rate, whether it re-reads files after failed edits,
speed and token use versus Gemma, and the quality of the final summaries.

## 4. Advanced RAG: fusion retrieval insights

32 hand-labelled questions (15 keyword-style naming exact classes, 17 plain-English paraphrases); a hit is a
chunk from a page that answers the question. Full results: [evaluation/rag_retrieval.md](evaluation/rag_retrieval.md).

| Method | hit@1 | hit@5 | MRR@10 |
|---|---|---|---|
| vector only | 0.72 | 0.88 | 0.80 |
| BM25 only | 0.69 | 0.84 | 0.75 |
| **fusion (α = 0.5)** | 0.75 | **0.94** | 0.82 |
| RRF (rank fusion) | **0.78** | **0.94** | **0.85** |

**Insights**
1. **Fusion beats both parts.** hit@5 went from 0.88 (vector) and 0.84 (BM25) to 0.94. The two methods fail on
   different questions. Vector search ranked the `ContextEntityRecall` page 8th, while BM25 put it 1st. BM25 missed
   the summarisation page for a paraphrased question, while vector search put it 1st. Fusion recovers most of these.
2. **On keyword questions fusion is nearly perfect** (hit@5 1.00, MRR 0.96). Class names like
   `LLMContextRecall` are exactly what BM25 is good at.
3. **Paraphrases are the hard part for all methods** (hit@1 0.59 for vector, BM25 and score fusion). Rank-based
   fusion (RRF) did better there (0.71), probably because min-max normalisation lets one very confident method
   dominate the fused score.
4. **The weight matters.** α = 0.75 scored best (MRR 0.90), but it was tuned on the same 32 questions, so we keep
   the default α = 0.5 and treat 0.75 as an optimistic upper bound. A held-out question set would be needed to choose α properly.
5. **A bug in the reference implementation.** The NirDiamant notebook adds vector scores (in similarity order)
   to BM25 scores (in corpus order) position by position, so a chunk's vector score can be added to another chunk's
   BM25 score. Our version keys both scores by chunk id before fusing.
6. **Remaining failures are vocabulary gaps.** "Did the retriever find all the information needed?" misses the
   Context Recall page with every method, because the page talks about "reference claims attributed to the context".
   Query transformation (e.g. HyDE) would be the next technique to try.
7. **Retrieval quality depends on the agent's query.** In the RAG-grounded coding task, Gemma searched for
   "...SingleTurnSample". That correctly returned the docs' *legacy* API section, so the generated script used the
   older API. Retrieval worked, but the model's query steered it.

## 5. Reflection

**What worked**
- The MCP abstraction: three very different servers (Node filesystem, remote Context7, our Python RAG server)
  behind one client. Adding the RAG server was a config change.
- Visible tool calls and diff previews: the confirm-mode diff caught a model replacing `divide()` instead of adding
  a new function.
- Automatic checks in the benchmark, rather than trusting model summaries.

**What didn't**
- Small local models get stuck in retry loops on exact-match edits. Forge recovers (errors go back to the model),
  but the model doesn't always learn from them.
- MCP SDK v2 differs from most tutorials (snake_case attributes, FastMCP renamed to MCPServer). We had to read the
  installed SDK's code.
- Windows-specific work: `npx.cmd`, console encodings, killing process trees on timeout, Ctrl+C handling.

**What we'd improve**
- Line-range edits or automatic re-reads after a failed `edit_file`, to break retry loops.
- Query transformation (HyDE) or a re-ranker in the RAG server, plus a held-out set to tune α.
- Session persistence and an `/undo` for file edits (optional extensions in the brief).

**TODO (team):** personal reflections, team process, and anything you learn from the Qwen runs.
