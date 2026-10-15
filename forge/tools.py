"""Tool registry: every tool the agent can call, from MCP servers or built in.

Responsibilities:
  * merge MCP tools (discovered at runtime) with Forge's native tools
  * turn each tool's JSON Schema into an OpenAI-style function definition for the LLM
  * apply the permission policy (confirm mode asks before any mutating tool)
  * build previews for confirmation (a real diff for edit_file via its dryRun option)
  * execute the call and truncate the output before it goes back to the LLM
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol

from .config import Settings
from .mcp_client import MCPClientManager, ToolInfo

Decision = Literal["yes", "no", "always"]


@dataclass
class ToolOutput:
    text: str
    is_error: bool = False
    duration_s: float = 0.0
    declined: bool = False


@dataclass
class Tool:
    name: str                 # name the LLM sees, e.g. filesystem__read_text_file
    short_name: str           # name shown in the UI, e.g. read_text_file
    source: str               # MCP server name, or "forge" for native tools
    description: str
    parameters: dict[str, Any]
    read_only: bool
    handler: Callable[[dict[str, Any]], Awaitable[ToolOutput]]

    def schema(self) -> dict[str, Any]:
        """OpenAI function-calling format, accepted by LangChain's bind_tools for every provider."""
        params = {k: v for k, v in self.parameters.items() if k != "$schema"}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return {"type": "function",
                "function": {"name": self.name, "description": self.description[:1024], "parameters": params}}


@dataclass
class ToolCall:
    id: str
    tool: Tool
    args: dict[str, Any]


@dataclass
class Preview:
    """What to show the user before they approve a mutating call."""

    kind: Literal["diff", "code", "command", "args"]
    text: str
    language: str = "text"
    note: str = ""


class ToolUI(Protocol):
    """The parts of the user interface the registry needs."""

    def on_tool_call(self, call: ToolCall) -> None: ...
    def on_tool_start(self, call: ToolCall) -> None: ...
    def on_tool_result(self, call: ToolCall, output: ToolOutput) -> None: ...
    async def confirm(self, call: ToolCall, preview: Preview) -> Decision: ...


@dataclass
class Permissions:
    mode: Literal["confirm", "auto"] = "confirm"
    always_allowed: set[str] = field(default_factory=set)   # tools approved with "always" this session

    def needs_confirmation(self, tool: Tool) -> bool:
        return self.mode == "confirm" and not tool.read_only and tool.name not in self.always_allowed


class ToolRegistry:
    def __init__(self, manager: MCPClientManager, settings: Settings):
        self.manager = manager
        self.settings = settings
        self.permissions = Permissions(mode=settings.mode)  # type: ignore[arg-type]
        self.tools: dict[str, Tool] = {}
        for info in manager.tools:
            self._add(self._wrap_mcp_tool(info))
        self._add(self._run_command_tool())

    # ------------------------------------------------------------------ registration
    def _add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def _wrap_mcp_tool(self, info: ToolInfo) -> Tool:
        async def handler(args: dict[str, Any]) -> ToolOutput:
            result = await self.manager.call_tool(info.qualified_name, args)
            return ToolOutput(result.text, result.is_error, result.duration_s)

        return Tool(name=info.qualified_name, short_name=info.name, source=info.server,
                    description=info.description, parameters=info.input_schema,
                    read_only=info.read_only, handler=handler)

    def _run_command_tool(self) -> Tool:
        shell = "cmd.exe" if sys.platform == "win32" else "/bin/sh"
        return Tool(
            name="run_command", short_name="run_command", source="forge",
            description=(f"Run a shell command ({shell}) in the workspace directory and return its exit code "
                         "and combined stdout/stderr. Use it to run tests, scripts, linters, git, pip, etc. "
                         "Non-interactive only: commands that wait for input will time out."),
            parameters={"type": "object", "properties": {
                "command": {"type": "string", "description": "The command line to execute"},
                "timeout_seconds": {"type": "integer",
                                    "description": f"Kill the command after this many seconds "
                                                   f"(default {self.settings.command_timeout_s})"},
            }, "required": ["command"]},
            read_only=False, handler=self._run_command,
        )

    # ------------------------------------------------------------------ lookup
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def resolve(self, name: str) -> Tool | None:
        """Exact name first; small models sometimes drop the `server__` prefix, so
        fall back to a unique match on the short name."""
        if name in self.tools:
            return self.tools[name]
        matches = [t for t in self.tools.values() if t.short_name == name or t.name.endswith(f"__{name}")]
        return matches[0] if len(matches) == 1 else None

    # ------------------------------------------------------------------ execution
    async def execute(self, call_id: str, name: str, args: Any, ui: ToolUI) -> tuple[ToolCall | None, ToolOutput]:
        """Run one tool call end to end. Always returns an output the LLM can learn from."""
        tool = self.resolve(name)
        if tool is None:
            return None, ToolOutput(f"Error: unknown tool '{name}'. Available tools: {', '.join(self.tools)}",
                                    is_error=True)
        if not isinstance(args, dict):
            return None, ToolOutput(f"Error: arguments for {name} must be a JSON object.", is_error=True)

        call = ToolCall(call_id, tool, args)
        ui.on_tool_call(call)

        if self.permissions.needs_confirmation(tool):
            preview, early_output = await self._preview(tool, args)
            if early_output is not None:
                # The dry run already failed (e.g. edit text not found), so the real call
                # would too. Report that to the model without asking the user.
                ui.on_tool_result(call, early_output)
                return call, early_output
            decision = await ui.confirm(call, preview)
            if decision == "no":
                output = ToolOutput("The user declined this tool call. Do not retry it unchanged: "
                                    "try a different approach or ask the user what they want.",
                                    is_error=True, declined=True)
                ui.on_tool_result(call, output)
                return call, output
            if decision == "always":
                self.permissions.always_allowed.add(tool.name)

        ui.on_tool_start(call)
        start = time.perf_counter()
        try:
            output = await tool.handler(args)
        except Exception as exc:  # a broken tool must never crash the agent loop
            output = ToolOutput(f"Error: {type(exc).__name__}: {exc}", is_error=True)
        output.duration_s = output.duration_s or (time.perf_counter() - start)
        ui.on_tool_result(call, output)
        return call, output

    def for_llm(self, output: ToolOutput) -> str:
        """Truncate long outputs, keeping the head and the tail (test failures are usually at the end)."""
        text, limit = output.text or "(no output)", self.settings.max_result_chars
        if len(text) <= limit:
            return text
        head, tail = int(limit * 0.6), int(limit * 0.4)
        return f"{text[:head]}\n\n[... {len(text) - head - tail:,} characters truncated ...]\n\n{text[-tail:]}"

    # ------------------------------------------------------------------ previews
    async def _preview(self, tool: Tool, args: dict[str, Any]) -> tuple[Preview, ToolOutput | None]:
        if tool.short_name == "edit_file" and tool.source == "filesystem":
            dry = await tool.handler({**args, "dryRun": True})
            if dry.is_error:
                return Preview("args", ""), dry
            return Preview("diff", dry.text.replace("```diff", "").replace("```", "").strip(), "diff"), None
        if tool.short_name == "write_file":
            path = str(args.get("path", ""))
            target = Path(path) if Path(path).is_absolute() else self.settings.workspace / path
            note = "overwrites existing file" if target.exists() else "new file"
            return Preview("code", str(args.get("content", "")), _lexer_for(path), note), None
        if tool.name == "run_command":
            return Preview("command", str(args.get("command", "")), "bat" if sys.platform == "win32" else "bash",
                           f"cwd: {self.settings.workspace}"), None
        return Preview("args", json.dumps(args, indent=2, ensure_ascii=False), "json"), None

    # ------------------------------------------------------------------ native tools
    async def _run_command(self, args: dict[str, Any]) -> ToolOutput:
        command = str(args.get("command", "")).strip()
        if not command:
            return ToolOutput("Error: 'command' is required.", is_error=True)
        timeout = int(args.get("timeout_seconds") or self.settings.command_timeout_s)
        # Make Python children print UTF-8 so their output decodes cleanly on Windows.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
                 else {"start_new_session": True})
        start = time.perf_counter()
        proc = await asyncio.create_subprocess_shell(
            command, cwd=self.settings.workspace, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **extra)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            await _kill_tree(proc)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return ToolOutput(f"Error: command timed out after {timeout}s and was killed.", True,
                              time.perf_counter() - start)
        text = stdout.decode("utf-8", errors="replace").strip()
        return ToolOutput(f"exit code: {proc.returncode}\n{text}" if text else f"exit code: {proc.returncode}",
                          is_error=proc.returncode != 0, duration_s=time.perf_counter() - start)


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a shell and everything it started (killing just the shell leaves children running)."""
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        killer = await asyncio.create_subprocess_exec("taskkill", "/F", "/T", "/PID", str(proc.pid),
                                                      stdout=asyncio.subprocess.DEVNULL,
                                                      stderr=asyncio.subprocess.DEVNULL)
        await killer.wait()
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


_LEXERS = {".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json", ".md": "markdown",
           ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".html": "html", ".css": "css", ".sh": "bash"}


def _lexer_for(path: str) -> str:
    return _LEXERS.get(Path(path).suffix.lower(), "text")
