# Forge: State Diagram

This shows how information moves through Forge, from startup through one task (an "agent turn") and back to idle.

```mermaid
stateDiagram-v2
    [*] --> Startup

    state Startup {
        [*] --> LoadConfig
        LoadConfig --> ConnectServers : .env + mcp_servers.json
        ConnectServers --> DiscoverTools : initialize handshake per server
        DiscoverTools --> BuildRegistry : list_tools results
        BuildRegistry --> [*] : MCP + native tools, permission levels
    }

    Startup --> Idle : banner (provider, mode, servers, tool counts)

    Idle --> ParseInput : user submits a line
    ParseInput --> SlashCommand : starts with "/"
    SlashCommand --> Idle : /mode, /model, /tools, /clear, /help
    SlashCommand --> Shutdown : /exit
    ParseInput --> AgentTurn : natural-language task

    state AgentTurn {
        [*] --> CallLLM : history + system prompt + tool schemas
        CallLLM --> Streaming : provider request
        Streaming --> InspectResponse : text tokens shown live
        state has_tools <<choice>>
        InspectResponse --> has_tools
        has_tools --> Finished : no tool calls (final answer)
        has_tools --> SelectTool : one or more tool calls

        SelectTool --> PermissionCheck : next pending call
        state needs_ok <<choice>>
        PermissionCheck --> needs_ok
        needs_ok --> ExecuteTool : read-only tool, auto mode, or "always" granted
        needs_ok --> PreviewChange : mutating tool in confirm mode
        PreviewChange --> AwaitConfirm : diff (edit_file dry run) or content / command preview
        AwaitConfirm --> ExecuteTool : yes / always
        AwaitConfirm --> RecordObservation : no, "user declined" result

        ExecuteTool --> RecordObservation : MCP call_tool or native run_command
        state more <<choice>>
        RecordObservation --> more : truncated result appended as ToolMessage
        more --> SelectTool : calls remaining in this response
        more --> CallLLM : all calls done, steps remaining
        more --> StepLimit : step budget exhausted
        StepLimit --> Finished
    }

    AgentTurn --> Idle : final answer rendered
    AgentTurn --> Idle : Ctrl+C (turn cancelled) or provider error shown

    Shutdown --> [*] : close MCP sessions, stop server processes
```

## Notes
- **Observations always go back to the model**, including errors and declined confirmations. The model sees what
  happened and can choose a different approach (for example, `create_directory` after a failed `write_file`).
- The **step budget** (default 25 LLM calls per task) stops runaway loops with small models.
- `/clear` empties the conversation history but keeps MCP connections; `/model` swaps the provider without
  restarting the servers.
