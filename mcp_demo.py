"""MCP integration demo (milestone 1).

Connects to every server in mcp_servers.json, lists the tools they expose,
and lets you invoke those tools, either through a guided scripted demo or
an interactive shell. There is no LLM in the loop yet; this exercises the
MCP client layer the agent will be built on.

Usage:
    python mcp_demo.py                # interactive shell
    python mcp_demo.py --demo         # guided walkthrough of both servers
    python mcp_demo.py --demo --no-pause
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from forge.config import load_server_configs
from forge.display import (
    console, render_connection, render_schema, render_servers,
    render_tool_call, render_tool_result, render_tools,
)
from forge.mcp_client import MCPClientManager, ToolInfo

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "mcp_servers.json"
LOG_DIR = ROOT / "logs"

HELP = """[bold]Commands[/]
  [cyan]servers[/]                       connection status of each MCP server
  [cyan]tools[/] [dim]\\[server][/]                 list tools (optionally for one server)
  [cyan]schema[/] <tool>                 show a tool's description and input schema
  [cyan]call[/] <tool> [dim]\\[json-args][/]        run a tool; omit args to be prompted for each one
  [cyan]demo[/]                          guided walkthrough of both servers
  [cyan]help[/] / [cyan]quit[/]
[dim]Tools can be named `read_text_file`, `filesystem/read_text_file` or `filesystem__read_text_file`.[/]"""


# --------------------------------------------------------------------------- tool execution
async def run_tool(manager: MCPClientManager, name: str, arguments: dict[str, Any], max_lines: int = 30) -> str:
    """Show the call, execute it on the owning server, show the result."""
    tool = manager.find_tool(name)
    render_tool_call(tool, arguments)
    with console.status(f"[dim]{tool.server} is running {tool.name}…[/]", spinner="dots"):
        result = await manager.call_tool(tool.qualified_name, arguments)
    render_tool_result(result, max_lines=max_lines)
    return result.text


def prompt_arguments(tool: ToolInfo) -> dict[str, Any]:
    """Ask for each parameter in the tool's JSON schema. Non-string values are parsed as JSON."""
    schema = tool.input_schema
    required = set(schema.get("required", []))
    arguments: dict[str, Any] = {}
    for name, prop in schema.get("properties", {}).items():
        kind = prop.get("type", "any")
        label = f"  [bold]{name}[/] [dim]({kind}{', required' if name in required else ', optional'})[/]"
        raw = Prompt.ask(label, default="" if name not in required else None, show_default=False)
        if raw == "":
            continue
        if kind == "string":
            arguments[name] = raw
        else:
            try:
                arguments[name] = json.loads(raw)
            except json.JSONDecodeError:
                arguments[name] = raw
    return arguments


# --------------------------------------------------------------------------- scripted demo
def _pause(enabled: bool) -> None:
    if enabled:
        console.input("[dim]  ↵ press Enter for the next step…[/]")


def _step(number: int, title: str, why: str) -> None:
    console.print()
    console.print(Rule(Text.assemble((f"Step {number}  ", "bold"), (title, "bold white")), align="left"))
    console.print(f"[dim]{why}[/]")


async def run_demo(manager: MCPClientManager, workspace: Path, pause: bool) -> None:
    """Walk through representative tools from each connected server."""
    connected = {name for name, conn in manager.connections.items() if conn.connected}
    demo_file = workspace / "demo" / "greeter.py"
    step = 0

    if "filesystem" in connected:
        console.print(Panel("[bold]Server 1: filesystem[/]  (@modelcontextprotocol/server-filesystem)\n"
                            f"Sandboxed to: [cyan]{workspace}[/]", border_style="cyan"))
        fs_steps = [
            ("Which directories may the agent touch?",
             "The server refuses any path outside the directories it was launched with.",
             "list_allowed_directories", {}),
            ("Explore the workspace", "How the agent will orient itself in an unfamiliar codebase.",
             "directory_tree", {"path": str(workspace)}),
            ("Create a folder", "write_file won't create missing parent directories, so make one first.",
             "create_directory", {"path": str(demo_file.parent)}),
            ("Create a new file", "write_file creates (or overwrites) a file with the content we give it.",
             "write_file", {"path": str(demo_file), "content":
                 '"""Tiny module written through the MCP filesystem server."""\n\n\n'
                 'def greet(name):\n    return "Hello " + name\n'}),
            ("Make a targeted edit", "edit_file replaces exact text and returns a git-style diff of the change.",
             "edit_file", {"path": str(demo_file), "edits": [{
                 "oldText": 'def greet(name):\n    return "Hello " + name',
                 "newText": 'def greet(name: str, excited: bool = False) -> str:\n'
                            '    """Return a greeting for `name`."""\n'
                            '    return f"Hello, {name}{\'!\' if excited else \'.\'}"'}]}),
            ("Read the result back", "Confirms the edit landed on disk.",
             "read_text_file", {"path": str(demo_file)}),
            ("Search the codebase", "Glob search across the workspace, e.g. to find every Python file.",
             "search_files", {"path": str(workspace), "pattern": "**/*.py"}),
            ("Try to escape the sandbox", "A path outside the allowed directory should be rejected.",
             "read_text_file", {"path": str(ROOT / "mcp_servers.json")}),
        ]
        for title, why, tool, args in fs_steps:
            step += 1
            _step(step, title, why)
            await run_tool(manager, tool, args)
            _pause(pause)

    if "context7" in connected:
        console.print()
        console.print(Panel("[bold]Server 2: context7[/]  (@upstash/context7-mcp)\n"
                            "External resource server: live, version-specific library docs.", border_style="magenta"))
        step += 1
        _step(step, "Resolve a library name to a Context7 ID",
              "The agent first looks up which documentation set matches the library it needs.")
        listing = await run_tool(manager, "resolve-library-id",
                                 {"libraryName": "langchain", "query": "define a custom tool for an agent"},
                                 max_lines=20)
        _pause(pause)

        match = re.search(r"Context7-compatible library ID:\s*(\S+)", listing)
        library_id = match.group(1) if match else "/websites/langchain"
        step += 1
        _step(step, "Pull up-to-date docs for that library",
              f"Using the ID returned above ({library_id}) to fetch focused documentation snippets.")
        await run_tool(manager, "query-docs",
                       {"libraryId": library_id, "query": "How do I define a custom tool with the @tool decorator?"},
                       max_lines=40)

    console.print()
    console.print(Panel("[bold green]Demo complete.[/] Every call above went through our MCP client: "
                        "tools were discovered at runtime, routed to the owning server over stdio, "
                        "and their results returned to the CLI.", border_style="green"))


# --------------------------------------------------------------------------- interactive shell
async def repl(manager: MCPClientManager, workspace: Path) -> None:
    console.print(HELP)
    while True:
        try:
            line = console.input("\n[bold cyan]mcp ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        command, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if command in ("quit", "exit", "q"):
                break
            elif command == "help":
                console.print(HELP)
            elif command == "servers":
                render_servers(manager)
            elif command == "tools":
                render_tools(manager, rest or None)
            elif command == "schema":
                render_schema(manager.find_tool(rest))
            elif command == "call":
                name, _, raw_args = rest.partition(" ")
                tool = manager.find_tool(name)
                arguments = json.loads(raw_args) if raw_args.strip() else prompt_arguments(tool)
                await run_tool(manager, tool.qualified_name, arguments)
            elif command == "demo":
                await run_demo(manager, workspace, pause=True)
            else:
                console.print(f"[red]Unknown command '{command}'.[/] Type [cyan]help[/].")
        except KeyError as exc:
            console.print(f"[red]{exc.args[0]}[/]")
        except json.JSONDecodeError as exc:
            console.print(f"[red]Arguments must be valid JSON:[/] {exc}")
        except KeyboardInterrupt:
            console.print("[dim]cancelled[/]")


# --------------------------------------------------------------------------- entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo of the assistant's MCP client layer.")
    parser.add_argument("--workspace", default=str(ROOT / "sandbox"),
                        help="directory the filesystem server is allowed to access (default: ./sandbox)")
    parser.add_argument("--servers", nargs="+", help="only connect to these servers from mcp_servers.json")
    parser.add_argument("--demo", action="store_true", help="run the guided walkthrough, then exit")
    parser.add_argument("--no-pause", action="store_true", help="don't wait for Enter between demo steps")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    configs = load_server_configs(CONFIG_PATH, {"WORKSPACE": str(workspace)})
    if args.servers:
        configs = [c for c in configs if c.name in args.servers]

    console.print(Panel.fit(
        Text.assemble(("MCP Integration Demo\n", "bold"),
                      ("CLI coding assistant · milestone 1 · tools discovered at runtime from MCP servers", "dim")),
        border_style="bright_blue"))
    console.print(f"[dim]Config:[/] {CONFIG_PATH.name}   [dim]Workspace:[/] {workspace}   "
                  f"[dim]Server logs:[/] {LOG_DIR.name}/\n")

    manager = MCPClientManager(configs, LOG_DIR)
    try:
        with console.status("Connecting to MCP servers (first run may download packages)…"):
            await manager.connect_all(on_connected=render_connection)
        console.print()
        render_tools(manager)

        if args.demo:
            await run_demo(manager, workspace, pause=not args.no_pause)
        else:
            await repl(manager, workspace)
    finally:
        with console.status("Shutting down MCP servers…"):
            await manager.close()


if __name__ == "__main__":
    # The Windows console defaults to a legacy code page; force UTF-8 so ✔/⚙/› render.
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
