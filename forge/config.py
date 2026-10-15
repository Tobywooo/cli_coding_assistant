"""Configuration: Forge settings (from the environment / .env) and MCP server definitions.

MCP servers are defined in mcp_servers.json rather than in code, so adding a
server (like our custom RAG server) needs no Python changes. Placeholders of
the form ${VAR} in that file are expanded from the environment plus a few
values Forge provides itself (WORKSPACE, PROJECT_ROOT, PYTHON).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVERS_FILE = PROJECT_ROOT / "mcp_servers.json"
LOG_DIR = PROJECT_ROOT / "logs"

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ServerConfig:
    """Everything needed to launch one MCP server over stdio."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    # True for servers whose tools never modify anything (docs lookups), so
    # their tools never need user confirmation. Servers that publish MCP
    # readOnlyHint annotations (e.g. filesystem) are classified per tool instead.
    read_only: bool = False
    cwd: str | None = None


@dataclass
class Settings:
    """Runtime settings, mostly read from environment variables (see .env.example)."""

    workspace: Path
    provider: str = "ollama"
    model: str | None = None          # None = the provider's default model
    mode: str = "confirm"             # "confirm" or "auto"
    max_steps: int = 25               # LLM calls allowed per task
    max_result_chars: int = 8000      # tool output sent back to the LLM is truncated to this
    command_timeout_s: int = 120      # default timeout for run_command
    servers_file: Path = DEFAULT_SERVERS_FILE

    @classmethod
    def from_env(cls, workspace: Path) -> "Settings":
        return cls(
            workspace=workspace,
            provider=os.getenv("FORGE_PROVIDER", "ollama"),
            mode=os.getenv("FORGE_MODE", "confirm"),
            max_steps=int(os.getenv("FORGE_MAX_STEPS", "25")),
            max_result_chars=int(os.getenv("FORGE_MAX_RESULT_CHARS", "8000")),
        )


def _expand(value: str, variables: dict[str, str]) -> str:
    """Replace every ${VAR} in `value`; unknown variables become ''."""
    return _PLACEHOLDER.sub(lambda m: variables.get(m.group(1), ""), value)


def load_server_configs(path: Path, extra_vars: dict[str, str] | None = None) -> list[ServerConfig]:
    """Parse the config file and return the enabled servers, placeholders expanded."""
    variables = {**os.environ, **(extra_vars or {})}
    raw = json.loads(path.read_text(encoding="utf-8"))

    configs: list[ServerConfig] = []
    for name, spec in raw.get("servers", {}).items():
        if not spec.get("enabled", True):
            continue
        env = {key: _expand(val, variables) for key, val in spec.get("env", {}).items()}
        configs.append(
            ServerConfig(
                name=name,
                command=_expand(spec["command"], variables),
                args=[_expand(arg, variables) for arg in spec.get("args", [])],
                # Drop env vars that expanded to nothing (e.g. an optional API key
                # that isn't set) so the server falls back to its own defaults.
                env={key: val for key, val in env.items() if val},
                description=spec.get("description", ""),
                read_only=bool(spec.get("read_only", False)),
                cwd=_expand(spec["cwd"], variables) if spec.get("cwd") else None,
            )
        )
    return configs
