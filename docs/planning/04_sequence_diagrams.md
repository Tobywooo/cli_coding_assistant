# Forge: Sequence Diagrams

Four end-to-end operations:

1. **Startup and dynamic tool discovery**
2. **Fixing a bug in confirm mode**: filesystem server, diff preview, confirmation prompts, native shell tool
3. **Writing code grounded in the RAG server** (auto mode): custom RAG server, fusion retrieval
4. **Looking up external docs, and a declined change**: Context7 server, user says "no", model adapts

Participants: **User**, **CLI Interface**, **Agentic Loop**, **Tools / MCP Client** (tool registry + MCP client
manager), the **MCP servers**, and the **LLM** (behind the provider abstraction).

---

## 1. Startup and dynamic tool discovery

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI Interface
    participant Agent as Agentic Loop
    participant Tools as Tools / MCP Client
    participant FS as Filesystem MCP Server
    participant C7 as Context7 MCP Server
    participant RAG as RAG MCP Server
    participant LLM

    User->>CLI: forge --provider ollama
    CLI->>Tools: load mcp_servers.json + .env
    loop for each configured server
        Tools->>FS: spawn process (stdio), initialize
        FS-->>Tools: server info + capabilities
        Tools->>FS: list_tools
        FS-->>Tools: 14 tools with JSON schemas + readOnlyHint
    end
    Note over Tools,RAG: same handshake for Context7 (2 tools) and RAG (search_ragas_docs)
    Tools->>Tools: add native run_command, qualify names (server__tool), classify read-only vs mutating
    Tools-->>CLI: registry ready (17+ tools)
    CLI->>Agent: create loop with provider + registry + system prompt
    Agent->>LLM: build chat model (no request yet)
    CLI-->>User: banner, provider/model, mode=confirm, servers and tool counts
```

---

## 2. Fixing a bug in confirm mode

Task: *"divide() in calculator.py crashes on zero. Fix it to raise a ValueError and run the tests."*

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI Interface
    participant Agent as Agentic Loop
    participant Tools as Tools / MCP Client
    participant FS as Filesystem MCP Server
    participant LLM

    User->>CLI: task (natural language)
    CLI->>Agent: run_turn(task)
    Agent->>LLM: system prompt + history + tool schemas
    LLM-->>Agent: tool_call filesystem__read_text_file(calculator.py)
    Agent->>CLI: event ToolCall (shown as panel)
    Agent->>Tools: execute read_text_file
    Note over Tools: read-only, so no confirmation
    Tools->>FS: call_tool read_text_file
    FS-->>Tools: file contents
    Tools-->>Agent: observation (truncated if long)
    Agent->>CLI: event ToolResult
    Agent->>LLM: history + ToolMessage(file contents)
    LLM-->>Agent: tool_call filesystem__edit_file(oldText, newText)
    Agent->>Tools: execute edit_file
    Note over Tools: mutating tool and mode=confirm
    Tools->>FS: call_tool edit_file with dryRun=true
    FS-->>Tools: unified diff (no change on disk)
    Tools->>CLI: confirm? (diff preview)
    CLI-->>User: show diff, prompt y / n / a
    User->>CLI: y
    CLI-->>Tools: approved
    Tools->>FS: call_tool edit_file (real)
    FS-->>Tools: diff of applied edit
    Tools-->>Agent: observation
    Agent->>LLM: history + ToolMessage(diff)
    LLM-->>Agent: tool_call run_command("python -m pytest -q")
    Agent->>Tools: execute run_command
    Tools->>CLI: confirm? (command preview)
    CLI-->>User: prompt y / n / a
    User->>CLI: y
    Tools->>Tools: subprocess in workspace (timeout 120 s)
    Tools-->>Agent: exit code 0 + output
    Agent->>LLM: history + ToolMessage(test output)
    LLM-->>Agent: streamed final answer (no tool calls)
    Agent->>CLI: text deltas
    CLI-->>User: "Fixed divide() ... all tests pass"
```

---

## 3. Writing code grounded in the RAG server (auto mode)

Task: *"Write eval_faithfulness.py that scores one sample with RAGAS' Faithfulness metric."*

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI Interface
    participant Agent as Agentic Loop
    participant Tools as Tools / MCP Client
    participant RAG as RAG MCP Server
    participant FS as Filesystem MCP Server
    participant LLM

    User->>CLI: /mode auto
    CLI-->>User: mutating tools now run without prompts
    User->>CLI: task (natural language)
    CLI->>Agent: run_turn(task)
    Agent->>LLM: system prompt + history + tool schemas
    LLM-->>Agent: tool_call rag__search_ragas_docs("Faithfulness metric SingleTurnSample", k=5)
    Agent->>CLI: event ToolCall
    Agent->>Tools: execute search_ragas_docs
    Tools->>RAG: call_tool search_ragas_docs
    activate RAG
    RAG->>RAG: BM25 scores over all chunks
    RAG->>RAG: vector similarity (Chroma, MiniLM embedding of query)
    RAG->>RAG: min-max normalise, fused = 0.5*vector + 0.5*bm25, top-k
    RAG-->>Tools: 5 chunks with source file, section, scores
    deactivate RAG
    Tools-->>Agent: observation
    Agent->>CLI: event ToolResult (sources shown)
    Agent->>LLM: history + ToolMessage(retrieved chunks)
    LLM-->>Agent: tool_call filesystem__write_file(eval_faithfulness.py, code)
    Note over Tools: mode=auto, so no prompt
    Agent->>Tools: execute write_file
    Tools->>FS: call_tool write_file
    FS-->>Tools: success
    Tools-->>Agent: observation
    Agent->>LLM: history + ToolMessage
    LLM-->>Agent: streamed final answer citing the docs used
    Agent->>CLI: text deltas
    CLI-->>User: summary + how to run the script
```

---

## 4. Looking up external docs, and a declined change

Task: *"Add a LangChain @tool wrapper around slugify() using the current LangChain API."*

```mermaid
sequenceDiagram
    actor User
    participant CLI as CLI Interface
    participant Agent as Agentic Loop
    participant Tools as Tools / MCP Client
    participant C7 as Context7 MCP Server
    participant FS as Filesystem MCP Server
    participant LLM

    User->>CLI: task (natural language)
    CLI->>Agent: run_turn(task)
    Agent->>LLM: system prompt + history + tool schemas
    LLM-->>Agent: tool_call context7__resolve-library-id("langchain")
    Agent->>Tools: execute (read-only)
    Tools->>C7: call_tool resolve-library-id
    C7-->>Tools: matching library IDs
    Tools-->>Agent: observation
    Agent->>LLM: history + ToolMessage
    LLM-->>Agent: tool_call context7__query-docs("/websites/langchain", "@tool decorator")
    Agent->>Tools: execute (read-only)
    Tools->>C7: call_tool query-docs
    C7-->>Tools: current docs + code snippets
    Tools-->>Agent: observation (truncated)
    Agent->>LLM: history + ToolMessage(docs)
    LLM-->>Agent: tool_call filesystem__write_file(utils/strings.py, whole file)
    Agent->>Tools: execute write_file
    Tools->>CLI: confirm? (content preview)
    CLI-->>User: preview of new file, prompt y / n / a
    User->>CLI: n (I'd rather keep the original file)
    CLI-->>Tools: declined
    Tools-->>Agent: observation "User declined this tool call"
    Agent->>LLM: history + ToolMessage(declined)
    LLM-->>Agent: tool_call filesystem__write_file(utils/tools.py, new file)
    Agent->>Tools: execute write_file
    Tools->>CLI: confirm? (content preview)
    CLI-->>User: prompt y / n / a
    User->>CLI: y
    Tools->>FS: call_tool write_file
    FS-->>Tools: success
    Tools-->>Agent: observation
    Agent->>LLM: history + ToolMessage
    LLM-->>Agent: streamed final answer
    Agent->>CLI: text deltas
    CLI-->>User: "Added utils/tools.py wrapping slugify as a LangChain tool"
```
