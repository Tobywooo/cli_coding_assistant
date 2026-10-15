# Milestone 1: MCP Integration Progress

## What's working

- **MCP client layer** (`assistant/mcp_client.py`) built on the official MCP Python SDK (v2.3), using the stdio transport.
  - Servers are defined in `mcp_servers.json`, not in code. Adding a server needs no Python changes.
  - Tools are discovered at runtime with `list_tools`. Our code never hard-codes a server's tools.
  - Tools get a qualified name (`server__tool`), so two servers can't collide when we later hand all tools to the LLM.
  - A server that fails to start is reported, and the other servers stay usable.
- **Server 1: filesystem** (`@modelcontextprotocol/server-filesystem`), 14 tools. The demo shows directory listing,
  creating and writing files, an exact-match edit that returns a diff, reading, glob search, and a rejected
  attempt to read outside the sandbox.
- **Server 2: Context7** (`@upstash/context7-mcp`), 2 tools. The demo shows `resolve-library-id` and then
  `query-docs`, which fetches current LangChain documentation.
- **CLI visibility**: every call shows the server, tool, JSON arguments, the result (or error) and timing, styled with Rich.

## Challenges encountered

1. **The MCP SDK changed under us.** `pip install mcp` installed v2.3, while most tutorials and the DeepLearning.AI
   course use the 1.x API. Attribute names moved to snake_case (`input_schema`, `is_error`, `server_info`),
   so we had to read the installed SDK's own signatures instead of relying on examples.
2. **Windows process spawning.** `npx` is really `npx.cmd` on Windows, and server stderr (npm notices, banners)
   was mixed into the UI. We now send each server's stderr to `logs/<server>.log`.
3. **Async lifecycle.** The SDK uses anyio task groups, which must be closed from the task that opened them.
   Connecting to servers in parallel tasks would break teardown, so we connect them one at a time on a single
   `AsyncExitStack`. Each server also gets its own stack, so a server that fails halfway through connecting is cleaned up without affecting the others.
4. **Tool quirks the LLM will run into.** `write_file` does not create parent directories (our first demo run
   failed with `ENOENT`), `edit_file` needs an exact text match, and the server only accepts paths inside its
   allowed directories. These rules will need to go in the system prompt or be handled gracefully in the agent loop.
5. **Large tool outputs.** `query-docs` returns about 4,000 characters per call. That's fine for display (we
   truncate), but it will eat into the context window of small local models, so we'll need truncation or
   summarisation before results go back to the LLM.
6. **External service limits.** Context7 works without a key but is rate-limited, so the config passes an
   optional `CONTEXT7_API_KEY` through from `.env`.

## Next steps

1. **Provider abstraction**: one interface over Ollama (local, e.g. `gemma4:e2b`) and a hosted, OpenAI-compatible
   endpoint (I have a llama.cpp server running Qwen3.8-27B). Check that the self-hosted endpoint counts
   as the required cloud provider, or add Groq or OpenAI as well.
2. **Agentic loop**: send the task and the MCP tool schemas to the LLM, run the tool calls it returns through
   `MCPClientManager.call_tool`, feed the results back, and repeat until the model gives a final answer.
   Include a confirm or auto-execute mode for tools that write files or run shell commands.
3. **Custom RAG MCP server**: ingest one library's documentation once into a persistent Chroma store, then
   serve retrieval as an MCP tool. Pick and implement an advanced technique from NirDiamant/RAG_Techniques.
4. **Streaming CLI REPL** with status indicators, built on the display module from this milestone.
5. **Diagrams and report**: state and sequence diagrams, plus the LLM comparison.
