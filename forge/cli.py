"""Forge's interactive terminal interface.

  * prompt_toolkit REPL with history and a status toolbar
  * streaming output: finished Markdown blocks are printed permanently while
    the block still being written is shown in a live region
  * every tool call is shown (server › tool + arguments) with its result
  * confirm mode: mutating tools show a preview (diff / file / command) and
    ask y / n / a(lways)
  * Ctrl+C cancels the running task without leaving Forge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.markdown import Markdown
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import Agent, TurnStats
from .config import LOG_DIR, PROJECT_ROOT, Settings, load_server_configs
from .display import BRAND, EMBER_GRADIENT, console, render_connection, render_servers, render_tools, server_color
from .mcp_client import MCPClientManager
from .providers import PROVIDERS, ProviderError, parse_model_ref
from .tools import Decision, Preview, ToolCall, ToolOutput, ToolRegistry

LOGO = [
    "███████╗ ██████╗ ██████╗  ██████╗ ███████╗",
    "██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝",
    "█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  ",
    "██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ",
    "██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗",
    "╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
]

HELP = f"""[bold]Just type a task[/], e.g. [italic]"add input validation to divide() and run the tests"[/].

[bold]Commands[/]
  [{BRAND}]/mode[/] [dim]\\[confirm|auto][/]       confirm: ask before tools that change things · auto: never ask
  [{BRAND}]/model[/] [dim]\\[provider[:model]][/]   show providers, or switch, e.g. /model llamacpp
  [{BRAND}]/tools[/] [dim]\\[server][/]             list the tools the agent can call
  [{BRAND}]/servers[/]                    MCP server status
  [{BRAND}]/clear[/]                      start a fresh conversation
  [{BRAND}]/help[/], [{BRAND}]/exit[/]
[dim]Ctrl+C cancels a running task · Ctrl+D exits · ↑ recalls previous prompts[/]"""


def _short(value: Any, workspace: Path, limit: int = 70) -> str:
    """Compact, single-line rendering of one argument value."""
    if isinstance(value, str):
        text = value.replace(str(workspace) + "\\", "").replace(str(workspace) + "/", "")
        text = text.replace("\n", "⏎ ")
    else:
        text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clean_diff(text: str) -> str:
    """Strip the code fence and the long Index/---/+++ path header from the filesystem server's diffs."""
    lines = text.replace("```diff", "").replace("```", "").strip().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("@@")), 0)
    return "\n".join(lines[start:])


class ForgeUI:
    """Implements the AgentUI callbacks with Rich + prompt_toolkit."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        # prompt_toolkit sessions need a real console, so they are created on first use;
        # non-interactive runs (one-shot tasks in auto mode, benchmarks) never create one.
        self._confirm_session: PromptSession | None = None
        self._live: Live | None = None
        self._label = ""
        self._phase_start = 0.0
        self._buffer = ""
        self._reasoning_chars = 0
        self._reasoning_tail = ""
        self._reasoning_start = 0.0
        self._thought_reported = False

    # ------------------------------------------------------------------ live region
    def _start_live(self, label: str) -> None:
        self._stop_live()
        self._label, self._phase_start = label, time.perf_counter()
        self._live = Live(get_renderable=self._render_live, console=console, refresh_per_second=10, transient=True)
        self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render_live(self):
        if self._buffer:  # answer text still being written: show its tail as plain text
            lines = self._buffer.splitlines()[-max(3, console.height - 4):]
            return Text("\n".join(lines))
        label = Text.assemble((self._label, f"bold {BRAND}"),
                              (f"  {time.perf_counter() - self._phase_start:.0f}s", "dim"))
        tail = self._reasoning_tail.strip().splitlines()
        if tail:
            label.append(f"   {tail[-1][-90:]}", style="dim italic")
        return Spinner("dots", text=label, style=BRAND)

    def reset(self) -> None:
        """Clean up after an interrupted turn."""
        self._stop_live()
        self._buffer = ""

    # ------------------------------------------------------------------ LLM streaming
    def on_llm_start(self, step: int) -> None:
        self._buffer, self._reasoning_chars, self._reasoning_tail = "", 0, ""
        self._thought_reported = False
        self._start_live("thinking")

    def on_reasoning(self, text: str) -> None:
        if not self._reasoning_chars:
            self._reasoning_start = time.perf_counter()
        self._reasoning_chars += len(text)
        self._reasoning_tail = (self._reasoning_tail + text)[-400:]

    def on_text(self, text: str) -> None:
        self._report_thought()
        self._buffer += text
        # Print every complete Markdown block (ending in a blank line, not inside a code fence).
        split = self._buffer.rfind("\n\n")
        while split > 0 and self._buffer[:split].count("```") % 2 == 1:
            split = self._buffer.rfind("\n\n", 0, split)
        if split > 0:
            stable, self._buffer = self._buffer[:split], self._buffer[split + 2:]
            console.print(Markdown(stable))

    def on_llm_end(self) -> None:
        self._stop_live()
        self._report_thought()
        if self._buffer.strip():
            console.print(Markdown(self._buffer))
        self._buffer = ""

    def _report_thought(self) -> None:
        if self._reasoning_chars and not self._thought_reported:
            self._thought_reported = True
            seconds = time.perf_counter() - self._reasoning_start
            console.print(Text(f"✻ thought for {seconds:.1f}s", style="dim italic"))

    def on_notice(self, text: str) -> None:
        self._stop_live()
        console.print(f"[yellow]⚠ {text}[/]")

    # ------------------------------------------------------------------ tools
    def on_tool_call(self, call: ToolCall) -> None:
        color = server_color(call.tool.source)
        line = Text.assemble(("● ", color), (call.tool.source, f"bold {color}"), (" › ", "dim"),
                             (call.tool.short_name, "bold"))
        for key, value in call.args.items():
            if key == "content":  # file bodies are shown in the confirmation preview, not here
                value = f"<{len(str(value).splitlines())} lines>"
            line.append(f"  {key}=", style="dim")
            line.append(_short(value, self.workspace))
        console.print(line)

    def on_tool_start(self, call: ToolCall) -> None:
        self._start_live(f"running {call.tool.short_name}")

    def on_tool_result(self, call: ToolCall | None, output: ToolOutput) -> None:
        self._stop_live()
        lines = output.text.strip().splitlines()
        if output.declined:
            console.print(Text("  ⎿ ✘ declined by user", style="yellow"))
            return
        status = ("✘ error", "red") if output.is_error else ("✔", "green")
        summary = Text.assemble("  ⎿ ", status, (f"  {output.duration_s:.2f}s · {len(lines)} lines", "dim"))
        console.print(summary)
        if not lines:
            return
        name = call.tool.short_name if call else ""
        if name == "edit_file" and not output.is_error:
            diff = "\n".join(_clean_diff(output.text).splitlines()[:40])
            console.print(Padding(Syntax(diff, "diff", theme="ansi_dark"), (0, 0, 0, 5)))
            return
        shown = lines[-10:] if name == "run_command" else lines[:6]
        hidden = len(lines) - len(shown)
        body = Text("\n".join(line[:160] for line in shown), style="red" if output.is_error else "dim")
        if hidden:
            body.append(f"\n… {hidden} more lines", style="dim italic")
        console.print(Padding(body, (0, 0, 0, 5)))

    async def confirm(self, call: ToolCall, preview: Preview) -> Decision:
        self._stop_live()
        if preview.kind == "diff":
            body = Syntax(_clean_diff(preview.text), "diff", theme="ansi_dark")
        elif preview.kind == "code":
            lines = preview.text.splitlines()
            body = Syntax("\n".join(lines[:40]) + (f"\n… ({len(lines) - 40} more lines)" if len(lines) > 40 else ""),
                          preview.language, theme="ansi_dark", line_numbers=True)
        else:
            body = Syntax(preview.text, preview.language, theme="ansi_dark", word_wrap=True)
        title = Text.assemble(("Allow ", "bold"), (call.tool.source, f"bold {server_color(call.tool.source)}"),
                              (" › ", "dim"), (call.tool.short_name, "bold"), ("?", "bold"))
        console.print(Panel(body, title=title, title_align="left", subtitle=preview.note or None,
                            subtitle_align="left", border_style="yellow", padding=(0, 1)))
        prompt = HTML(f"  <b>[Y]</b>es  <b>[n]</b>o  <b>[a]</b>lways allow {call.tool.short_name} this session "
                      f"<style fg='{BRAND}'>›</style> ")
        self._confirm_session = self._confirm_session or PromptSession()
        while True:
            try:
                answer = (await self._confirm_session.prompt_async(prompt)).strip().lower()
            except (KeyboardInterrupt, EOFError):
                return "no"
            if answer in ("", "y", "yes"):
                return "yes"
            if answer in ("n", "no"):
                return "no"
            if answer in ("a", "always"):
                return "always"


class ForgeCLI:
    def __init__(self, settings: Settings, manager: MCPClientManager, registry: ToolRegistry, agent: Agent):
        self.settings, self.manager, self.registry, self.agent = settings, manager, registry, agent
        self.ui = ForgeUI(settings.workspace)
        self._turn: asyncio.Task | None = None

    # ------------------------------------------------------------------ REPL
    async def repl(self) -> None:
        self._install_interrupt_handler()
        LOG_DIR.mkdir(exist_ok=True)
        session: PromptSession = PromptSession(history=FileHistory(str(LOG_DIR / "prompt_history.txt")))
        prompt = HTML(f"\n<style fg='{BRAND}'><b>forge</b></style> <style fg='#888888'>›</style> ")
        while True:
            try:
                line = (await session.prompt_async(prompt, bottom_toolbar=self._toolbar)).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not await self.handle_command(line):
                    break
                continue
            await self.run_task(line)

    def _toolbar(self) -> HTML:
        mode_color = "ansigreen" if self.registry.permissions.mode == "confirm" else "ansiyellow"
        return HTML(f" <b>{self.agent.provider}</b> · {self.agent.model_name}   │   mode: "
                    f"<style fg='{mode_color}'><b>{self.registry.permissions.mode}</b></style>   │   "
                    f"{len(self.registry.tools)} tools   │   /help ")

    def _install_interrupt_handler(self) -> None:
        """Ctrl+C while a task runs cancels that task only (at the prompt, prompt_toolkit handles it)."""
        loop = asyncio.get_running_loop()

        def on_sigint(signum, frame):
            if self._turn is not None and not self._turn.done():
                loop.call_soon_threadsafe(self._turn.cancel)

        signal.signal(signal.SIGINT, on_sigint)

    async def run_task(self, task: str) -> TurnStats | None:
        self._turn = asyncio.create_task(self.agent.run_turn(task, self.ui))
        try:
            stats = await self._turn
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():   # we are being cancelled ourselves: propagate
                raise
            self.ui.reset()
            console.print("[yellow]■ interrupted[/] [dim](conversation kept; type a new task or /clear)[/]")
            return None
        except Exception as exc:
            self.ui.reset()
            console.print(Panel(f"{type(exc).__name__}: {exc}", title="[bold red]Provider / agent error[/]",
                                border_style="red"))
            if "ConnectError" in type(exc).__name__ or "Connection" in str(exc):
                console.print(f"[dim]Is the {self.agent.provider} server running and reachable? "
                              "Switch with /model.[/]")
            return None
        finally:
            self._turn = None
        self._print_footer(stats)
        return stats

    @staticmethod
    def _print_footer(stats: TurnStats) -> None:
        mark = Text("✔ done", style="green") if stats.finished else Text("■ stopped", style="yellow")
        details = (f"  {stats.steps} step{'s' * (stats.steps != 1)} · {stats.tool_calls} tool call"
                   f"{'s' * (stats.tool_calls != 1)} · {stats.input_tokens:,} in / {stats.output_tokens:,} out tokens"
                   f" · {stats.duration_s:.1f}s")
        console.print(Text.assemble(mark, (details, "dim")))

    # ------------------------------------------------------------------ slash commands
    async def handle_command(self, line: str) -> bool:
        """Returns False when Forge should exit."""
        command, _, arg = line.partition(" ")
        arg = arg.strip()
        if command in ("/exit", "/quit", "/q"):
            return False
        if command == "/help":
            console.print(HELP)
        elif command == "/mode":
            permissions = self.registry.permissions
            new_mode = arg or ("auto" if permissions.mode == "confirm" else "confirm")
            if new_mode not in ("confirm", "auto"):
                console.print("[red]Mode must be 'confirm' or 'auto'.[/]")
            else:
                permissions.mode = new_mode  # type: ignore[assignment]
                note = "mutating tools ask first" if new_mode == "confirm" else "all tools run without asking"
                console.print(f"Mode: [bold]{new_mode}[/] [dim]({note})[/]")
        elif command == "/model":
            self._model_command(arg)
        elif command == "/tools":
            render_tools(self.manager, arg or None)
            console.print(f"[dim]+ built-in: [/][{BRAND}]run_command[/] [dim](shell, always asks in confirm mode)[/]")
        elif command == "/servers":
            render_servers(self.manager)
        elif command == "/clear":
            self.agent.reset()
            console.print("[dim]Conversation cleared.[/]")
        else:
            console.print(f"[red]Unknown command {command}.[/] Type /help.")
        return True

    def _model_command(self, arg: str) -> None:
        if not arg:
            table = Table(title="Providers", header_style="bold", title_style="bold")
            for column in ("", "Provider", "Description", "Default model", "Status"):
                table.add_column(column)
            for spec in PROVIDERS.values():
                missing = spec.missing_env()
                status = Text("ready", "green") if not missing else Text(f"set {', '.join(missing)}", "red")
                current = "●" if spec.name == self.agent.provider else ""
                table.add_row(current, spec.name, spec.label, spec.configured_model, status)
            console.print(table)
            console.print("[dim]Switch with /model <provider> or /model <provider>:<model>[/]")
            return
        provider, model = parse_model_ref(arg)
        try:
            self.agent.set_model(provider, model)
        except ProviderError as exc:
            console.print(f"[red]{exc}[/]")
            return
        console.print(f"Now using [bold]{self.agent.provider}[/] · {self.agent.model_name} "
                      "[dim](conversation kept)[/]")


# ---------------------------------------------------------------------------- startup
def print_banner() -> None:
    console.print()
    for line, color in zip(LOGO, EMBER_GRADIENT):
        console.print(Text("  " + line, style=f"bold {color}"))
    console.print(Text.assemble(("  autonomous coding agent", "bold"), (f"  ·  v{__version__}  ·  ", "dim"),
                                ("MCP-powered: filesystem · docs · RAG", "dim")))
    console.print()


def print_session_info(agent: Agent, registry: ToolRegistry, manager: MCPClientManager) -> None:
    connected = [c for c in manager.connections.values() if c.connected]
    mode = registry.permissions.mode
    rows = [
        ("model", f"[bold]{agent.provider}[/] · {agent.model_name}"),
        ("mode", f"[bold]{mode}[/] [dim]({'mutating tools ask first' if mode == 'confirm' else 'no confirmations'}"
                 " · /mode to change)[/]"),
        ("workspace", str(registry.settings.workspace)),
        ("tools", f"{len(registry.tools)} [dim]({len(registry.tools) - 1} from {len(connected)} MCP servers"
                  " + run_command)[/]"),
    ]
    console.print()
    for key, value in rows:
        console.print(f"  [dim]{key:<10}[/] {value}")
    console.print(f"\n  [dim]Type a task, or[/] [{BRAND}]/help[/][dim].[/]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="forge", description="Forge: an autonomous coding agent for your terminal.")
    parser.add_argument("task", nargs="*", help="run this one task and exit (omit for the interactive REPL)")
    parser.add_argument("-p", "--provider", help=f"LLM provider ({', '.join(PROVIDERS)}), optionally provider:model")
    parser.add_argument("-m", "--model", help="model name for the provider")
    parser.add_argument("-w", "--workspace", default=".", help="project directory Forge works in (default: cwd)")
    parser.add_argument("--auto", action="store_true", help="auto mode: run every tool without asking")
    parser.add_argument("--servers", nargs="+", help="only connect these MCP servers from mcp_servers.json")
    return parser.parse_args()


async def amain(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        console.print(f"[red]Workspace {workspace} is not a directory.[/]")
        return 1

    settings = Settings.from_env(workspace)
    if args.provider:
        settings.provider, settings.model = parse_model_ref(args.provider)
    if args.model:
        settings.model = args.model
    if args.auto:
        settings.mode = "auto"

    print_banner()
    configs = load_server_configs(settings.servers_file, {
        "WORKSPACE": str(workspace), "PROJECT_ROOT": str(PROJECT_ROOT), "PYTHON": sys.executable})
    if args.servers:
        configs = [c for c in configs if c.name in args.servers]

    manager = MCPClientManager(configs, LOG_DIR)
    try:
        with console.status(f"[{BRAND}]Connecting to MCP servers…[/]", spinner="dots"):
            await manager.connect_all(on_connected=render_connection)
        registry = ToolRegistry(manager, settings)
        connected = {name for name, conn in manager.connections.items() if conn.connected}
        try:
            agent = Agent(registry, settings, connected)
        except ProviderError as exc:
            console.print(f"\n[red]{exc}[/]")
            return 1
        print_session_info(agent, registry, manager)

        cli = ForgeCLI(settings, manager, registry, agent)
        if args.task:
            stats = await cli.run_task(" ".join(args.task))
            return 0 if stats and stats.finished else 1
        await cli.repl()
        return 0
    finally:
        with console.status("[dim]Shutting down MCP servers…[/]"):
            await manager.close()


def main() -> None:
    # The Windows console defaults to a legacy code page; force UTF-8 so the logo and icons render.
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(asyncio.run(amain(parse_args())))
    except KeyboardInterrupt:
        pass
