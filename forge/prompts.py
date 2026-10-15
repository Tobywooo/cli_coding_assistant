"""System prompt for the agent. Built at runtime so it only mentions servers that actually connected."""
from __future__ import annotations

import datetime as dt
import platform
import sys
from pathlib import Path

_BASE = """You are Forge, an autonomous coding agent running in the user's terminal. You complete software tasks \
in the user's local codebase by calling tools, observing the results, and continuing until the task is done.

# Environment
- Workspace (project root): {workspace}
- OS: {os}. run_command uses {shell}.
- Today's date: {date}

# Workspace files (snapshot at startup)
{tree}

# How to work
- Act, don't just describe: inspect the code with tools before changing it, then make the change yourself.
- Only use paths that exist (see the snapshot above, or explore with filesystem__directory_tree / \
filesystem__search_files). Never guess paths. Paths are relative to the workspace or absolute inside it.
- Read a file before editing it. For small changes use filesystem__edit_file; its oldText must match the file \
exactly, including indentation. Use filesystem__write_file for new files or full rewrites.
- filesystem__write_file does not create folders: call filesystem__create_directory first when needed.
- After changing code, verify it: run the code or its tests with run_command when possible, and fix failures.
- If a tool fails or the user declines a call, read the message and adapt. Never repeat an identical failing call.
- Don't ask the user for permission in plain text. Just call the tool; Forge asks the user when needed.
{server_hints}
# Final answer
When the task is complete, reply with a brief summary: what you changed (files), how you verified it, and \
anything the user should know. Use Markdown. Be concise."""

_SERVER_HINTS = {
    "rag": "- For anything about the RAGAS library (evaluation metrics, SingleTurnSample/EvaluationDataset, "
           "evaluate(), test-set generation), call rag__search_ragas_docs FIRST and base your code on the "
           "retrieved docs. Your memory of the RAGAS API is likely out of date.",
    "context7": "- For other third-party libraries, get current docs with context7__resolve-library-id, then "
                "context7__query-docs, instead of relying on memory.",
}


_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".idea",
              ".vscode", "dist", "build", ".tox"}


def workspace_snapshot(workspace: Path, max_depth: int = 3, max_entries: int = 60) -> str:
    """A short indented file tree, so the model starts with real paths instead of guessing."""
    lines: list[str] = []

    def walk(directory: Path, depth: int) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if len(lines) >= max_entries:
                return
            if entry.name in _SKIP_DIRS or entry.name.startswith(".") and entry.is_dir():
                continue
            lines.append("  " * depth + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir() and depth + 1 < max_depth:
                walk(entry, depth + 1)

    walk(workspace, 0)
    if len(lines) >= max_entries:
        lines.append("... (truncated, explore with tools)")
    return "\n".join(lines) if lines else "(empty directory)"


def build_system_prompt(workspace: Path, connected_servers: set[str]) -> str:
    hints = [hint for server, hint in _SERVER_HINTS.items() if server in connected_servers]
    return _BASE.format(
        workspace=workspace,
        os=f"{platform.system()} {platform.release()}",
        shell="cmd.exe (Windows): use `python`, `dir`, `type`; chain with &&" if sys.platform == "win32" else "/bin/sh",
        date=dt.date.today().isoformat(),
        tree=workspace_snapshot(workspace),
        server_hints=("\n# Documentation tools\n" + "\n".join(hints) + "\n") if hints else "",
    )
