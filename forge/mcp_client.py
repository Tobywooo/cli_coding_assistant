"""MCP client: connects to several MCP servers and exposes their tools in one registry.

Each server is launched as a subprocess and spoken to over stdio using the
official MCP Python SDK. Tools are discovered at runtime with `list_tools`,
so nothing about a server's tools is hard-coded here. The tool registry
(forge/tools.py) wraps these tools for the LLM and routes its tool calls
back through `call_tool`.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ServerConfig

# npx may need to download a package on first run, so allow a generous startup window.
CONNECT_TIMEOUT_S = 120
CALL_TIMEOUT_S = 120


@dataclass
class ToolInfo:
    """A tool advertised by an MCP server."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    # Whether the tool only reads state. Drives the confirmation policy.
    read_only: bool = False

    @property
    def qualified_name(self) -> str:
        """Globally unique name (`server__tool`) so two servers can't collide."""
        return f"{self.server}__{self.name}"


@dataclass
class ToolResult:
    """The outcome of one tool call, flattened to text for display / the LLM."""

    tool: ToolInfo
    arguments: dict[str, Any]
    text: str
    is_error: bool
    duration_s: float


@dataclass
class ServerConnection:
    """Live state for one connected (or failed) server."""

    config: ServerConfig
    session: ClientSession | None = None
    server_name: str = ""
    server_version: str = ""
    tools: list[ToolInfo] = field(default_factory=list)
    error: str | None = None
    connect_time_s: float = 0.0

    @property
    def connected(self) -> bool:
        return self.session is not None


def _is_read_only(tool: Any, cfg: ServerConfig) -> bool:
    """A tool is read-only if its server is trusted as read-only in the config,
    or if the server annotates the tool with readOnlyHint=True. Anything
    unannotated is treated as mutating, the safe default."""
    if cfg.read_only:
        return True
    annotations = getattr(tool, "annotations", None)
    return bool(annotations and getattr(annotations, "read_only_hint", False))


def _describe_exception(exc: BaseException) -> str:
    """anyio wraps failures in ExceptionGroups; dig out the first real error."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


class MCPClientManager:
    """Owns the connections to every configured MCP server."""

    def __init__(self, configs: list[ServerConfig], log_dir: Path):
        self.configs = configs
        self.log_dir = log_dir
        self.connections: dict[str, ServerConnection] = {}
        # One exit stack holds every subprocess/session so `close()` tears
        # them all down in reverse order.
        self._stack = AsyncExitStack()

    # ------------------------------------------------------------------ lifecycle
    async def connect_all(self, on_connected=None) -> None:
        """Connect to each server in turn.

        Connections are made sequentially in the same task on purpose: the SDK
        uses anyio task groups, which must be exited from the task that
        entered them.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for cfg in self.configs:
            conn = await self._connect(cfg)
            if on_connected:
                on_connected(conn)

    async def _connect(self, cfg: ServerConfig) -> ServerConnection:
        conn = ServerConnection(config=cfg)
        self.connections[cfg.name] = conn
        start = time.perf_counter()

        # Each server gets its own stack so a failure halfway through
        # connecting can be cleaned up without touching the others.
        server_stack = AsyncExitStack()
        try:
            # Server stderr (banners, npm notices, debug output) goes to a log
            # file instead of cluttering the terminal UI.
            errlog = server_stack.enter_context(
                open(self.log_dir / f"{cfg.name}.log", "w", encoding="utf-8")
            )
            params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None, cwd=cfg.cwd)
            read, write = await server_stack.enter_async_context(stdio_client(params, errlog=errlog))
            session = await server_stack.enter_async_context(ClientSession(read, write))

            async with asyncio.timeout(CONNECT_TIMEOUT_S):
                init = await session.initialize()
                listed = await session.list_tools()

            conn.session = session
            conn.server_name = init.server_info.name
            conn.server_version = init.server_info.version
            conn.tools = [
                ToolInfo(
                    server=cfg.name,
                    name=tool.name,
                    description=(tool.description or "").strip(),
                    input_schema=tool.input_schema or {},
                    read_only=_is_read_only(tool, cfg),
                )
                for tool in listed.tools
            ]
            await self._stack.enter_async_context(server_stack)
        except Exception as exc:  # report the failure but keep the other servers usable
            conn.error = _describe_exception(exc)
            await server_stack.aclose()
        conn.connect_time_s = time.perf_counter() - start
        return conn

    async def close(self) -> None:
        await self._stack.aclose()

    # ------------------------------------------------------------------ tools
    @property
    def tools(self) -> list[ToolInfo]:
        """All tools from all connected servers."""
        return [tool for conn in self.connections.values() for tool in conn.tools]

    def find_tool(self, name: str) -> ToolInfo:
        """Look a tool up by qualified name (`server__tool`), `server/tool`, or bare name if unique."""
        name = name.replace("/", "__").replace(".", "__")
        matches = [t for t in self.tools if name in (t.qualified_name, t.name)]
        if not matches:
            raise KeyError(f"Unknown tool '{name}'")
        if len(matches) > 1:
            options = ", ".join(t.qualified_name for t in matches)
            raise KeyError(f"'{name}' is ambiguous; use one of: {options}")
        return matches[0]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run a tool on the server that owns it and return its text output."""
        tool = self.find_tool(name)
        session = self.connections[tool.server].session
        arguments = arguments or {}
        start = time.perf_counter()
        try:
            result = await session.call_tool(tool.name, arguments, read_timeout_seconds=CALL_TIMEOUT_S)
            text = self._content_to_text(getattr(result, "content", None) or [])
            is_error = bool(getattr(result, "is_error", False))
        except Exception as exc:
            text, is_error = _describe_exception(exc), True
        return ToolResult(tool, arguments, text, is_error, time.perf_counter() - start)

    @staticmethod
    def _content_to_text(content: list[Any]) -> str:
        """Flatten MCP content blocks (text, images, resources) into one string."""
        parts = []
        for block in content:
            kind = getattr(block, "type", "")
            if kind == "text":
                parts.append(block.text)
            elif kind == "resource" and hasattr(block.resource, "text"):
                parts.append(block.resource.text)
            else:
                parts.append(f"[{kind or type(block).__name__} content]")
        return "\n".join(parts)
