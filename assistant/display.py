"""Rich-based rendering for servers, tools and tool calls.

Making every tool call visible (which server, which tool, what arguments,
what came back, how long it took) is a core requirement, so all of that
formatting lives here and can be reused by the agent loop later.
"""
from __future__ import annotations

import json
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .mcp_client import MCPClientManager, ServerConnection, ToolInfo, ToolResult

console = Console()

# One accent colour per server so it's obvious at a glance which server a call hit.
SERVER_COLORS = ["cyan", "magenta", "yellow", "green", "blue"]
_color_cache: dict[str, str] = {}


def server_color(server: str) -> str:
    if server not in _color_cache:
        _color_cache[server] = SERVER_COLORS[len(_color_cache) % len(SERVER_COLORS)]
    return _color_cache[server]


def first_line(text: str, limit: int = 90) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def render_connection(conn: ServerConnection) -> None:
    """One status line per server as it comes up during startup."""
    name = (f"{conn.config.name:<12}", f"bold {server_color(conn.config.name)}")
    if conn.connected:
        console.print(Text.assemble(
            ("  ✔ ", "green"), name,
            (f"{conn.server_name} v{conn.server_version}", "dim"),
            (f"  {len(conn.tools)} tools", "bold"),
            (f"  ({conn.connect_time_s:.1f}s)", "dim"),
        ))
    else:
        console.print(Text.assemble(("  ✘ ", "red"), name, (conn.error or "", "red")))


def render_servers(manager: MCPClientManager) -> None:
    table = Table(title="MCP servers", title_style="bold", header_style="bold")
    table.add_column("Server")
    table.add_column("Status")
    table.add_column("Reports as")
    table.add_column("Tools", justify="right")
    table.add_column("Launch command", style="dim", overflow="fold")
    for conn in manager.connections.values():
        status = Text("connected", style="green") if conn.connected else Text("failed", style="red")
        command = " ".join([conn.config.command, *conn.config.args])
        reports_as = f"{conn.server_name} v{conn.server_version}" if conn.connected else conn.error or ""
        table.add_row(
            Text(conn.config.name, style=f"bold {server_color(conn.config.name)}"),
            status, reports_as, str(len(conn.tools)), command,
        )
    console.print(table)


def render_tools(manager: MCPClientManager, server: str | None = None) -> None:
    table = Table(title="Tools loaded from MCP servers", title_style="bold", header_style="bold", expand=True)
    table.add_column("Server", no_wrap=True)
    table.add_column("Tool", style="bold", no_wrap=True)
    table.add_column("Parameters", style="dim")
    # Single-line descriptions keep the table readable; `schema <tool>` shows the full text.
    table.add_column("Description", ratio=1, no_wrap=True, overflow="ellipsis")
    for tool in manager.tools:
        if server and tool.server != server:
            continue
        table.add_row(
            Text(tool.server, style=server_color(tool.server)),
            tool.name, ", ".join(_param_labels(tool)), first_line(tool.description),
        )
    console.print(table)


def _param_labels(tool: ToolInfo) -> list[str]:
    required = set(tool.input_schema.get("required", []))
    return [f"{p}*" if p in required else p for p in tool.input_schema.get("properties", {})]


def render_schema(tool: ToolInfo) -> None:
    body = Group(
        Text(tool.description or "(no description)"),
        Text(""),
        Syntax(json.dumps(tool.input_schema, indent=2), "json", theme="ansi_dark", word_wrap=True),
    )
    console.print(Panel(body, title=f"[bold]{tool.qualified_name}[/]", border_style=server_color(tool.server)))


def render_tool_call(tool: ToolInfo, arguments: dict[str, Any]) -> None:
    """Shown *before* a tool runs: which server and tool, with what arguments."""
    color = server_color(tool.server)
    title = Text.assemble(("⚙ tool call  ", "bold"), (tool.server, f"bold {color}"), (" › ", "dim"), (tool.name, "bold"))
    args = Syntax(json.dumps(arguments, indent=2, ensure_ascii=False), "json", theme="ansi_dark", word_wrap=True)
    console.print(Panel(args, title=title, title_align="left", border_style=color))


def render_tool_result(result: ToolResult, max_lines: int = 30) -> None:
    """Shown *after* a tool runs: its output (truncated) and timing."""
    lines = result.text.splitlines() or ["(empty result)"]
    shown = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        shown += f"\n… ({len(lines) - max_lines} more lines, {len(result.text):,} chars total)"
    status = ("✘ error", "bold red") if result.is_error else ("✔ result", "bold green")
    title = Text.assemble(status, (f"  {result.tool.name}", "bold"), (f"  {result.duration_s:.2f}s", "dim"))
    console.print(Panel(Text(shown), title=title, title_align="left",
                        border_style="red" if result.is_error else "green"))
