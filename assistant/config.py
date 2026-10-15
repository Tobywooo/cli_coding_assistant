"""Loads MCP server definitions from mcp_servers.json.

Placeholders of the form ${VAR} are expanded from the process environment
(plus any extra variables passed in, e.g. WORKSPACE). Keeping server
definitions in a JSON file means new servers (like our custom RAG server)
can be added without touching any Python code.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ServerConfig:
    """Everything needed to launch one MCP server over stdio."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""


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
                command=spec["command"],
                args=[_expand(arg, variables) for arg in spec.get("args", [])],
                # Drop env vars that expanded to nothing (e.g. an optional API key
                # that isn't set) so the server falls back to its own defaults.
                env={key: val for key, val in env.items() if val},
                description=spec.get("description", ""),
            )
        )
    return configs
