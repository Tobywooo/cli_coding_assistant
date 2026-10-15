# Forge: System Architecture (original, planning phase)

```mermaid
flowchart LR
    user([Developer]) <-->|task / answers,<br/>confirmations| cli

    subgraph forge["Forge process (Python)"]
        cli["CLI Interface<br/>prompt_toolkit + Rich<br/>streaming, slash commands,<br/>confirm prompts"]
        agent["Agentic Loop<br/>messages, step budget,<br/>tool-call dispatch"]
        prov["Provider Abstraction<br/>LangChain chat models"]
        reg["Tool Registry<br/>schemas, permissions,<br/>truncation"]
        native["Native tool<br/>run_command"]
        mcpc["MCP Client Manager<br/>stdio sessions,<br/>tool discovery"]
        cfg[("mcp_servers.json<br/>.env")]

        cli <-->|events: text, tool call,<br/>tool result, confirm?| agent
        agent <-->|messages + tool schemas /<br/>streamed tokens + tool calls| prov
        agent <-->|execute tool / observation| reg
        reg --> native
        reg <--> mcpc
        cfg -.->|server definitions,<br/>API keys| mcpc
        cfg -.-> prov
    end

    subgraph llms["LLM providers"]
        ollama["Ollama (local)<br/>gemma4:e2b"]
        llamacpp["llama.cpp server (hosted)<br/>Qwen3.8-27B<br/>OpenAI-compatible API"]
    end
    prov <-->|HTTP| ollama
    prov <-->|HTTPS| llamacpp

    subgraph servers["MCP servers (child processes, stdio JSON-RPC)"]
        fs["Filesystem server<br/>@modelcontextprotocol/server-filesystem<br/>sandboxed to workspace"]
        c7["Context7 server<br/>@upstash/context7-mcp"]
        rag["Forge RAG server<br/>(custom, Python)<br/>search_ragas_docs"]
    end
    mcpc <--> fs
    mcpc <--> c7
    mcpc <--> rag

    fs <-->|read / write| ws[("Workspace<br/>(user's codebase)")]
    native -->|subprocess| ws
    c7 <-->|HTTPS| c7api(("Context7 API"))

    subgraph ragstore["RAG data (built once)"]
        chroma[("Chroma vector DB<br/>MiniLM embeddings")]
        bm25["BM25 index<br/>(built from stored chunks<br/>at server start)"]
    end
    rag <--> chroma
    rag <--> bm25
    ingest["rag_server.ingest<br/>(one-time)<br/>load, chunk, embed"] -->|persist| chroma
    docs[("RAGAS docs<br/>Markdown")] --> ingest
```

## Component responsibilities

| Component | Responsibility |
|---|---|
| **CLI Interface** | Reads tasks and slash commands; streams the model's text; renders each tool call (server, tool, arguments) and result; asks for confirmation in `confirm` mode |
| **Agentic Loop** | Holds the conversation; calls the LLM with the tool schemas; executes the returned tool calls in order; appends observations; repeats until the model answers without tool calls or the step budget runs out |
| **Provider Abstraction** | Builds a LangChain chat model for `ollama` or `llamacpp` from the config; `bind_tools` gives every provider the same tool interface |
| **Tool Registry** | Merges MCP tools and native tools; converts MCP JSON Schemas to function schemas; classifies tools as read-only or mutating; truncates outputs |
| **MCP Client Manager** | Starts each configured server, runs `initialize` + `list_tools`, routes `call_tool` to the owning session, and shuts the servers down cleanly |
| **Filesystem server** | Reads, writes, edits (with diff/dry-run) and searches files, restricted to the workspace |
| **Context7 server** | Up-to-date documentation for public libraries |
| **Forge RAG server** | Fusion retrieval (BM25 + vectors) over the RAGAS documentation |
