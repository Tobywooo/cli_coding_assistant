"""Compare LLM providers on the same coding tasks, scored by automatic checks rather than the model's own claims.

    python scripts/compare_models.py                          # every provider that is configured, 1 run each
    python scripts/compare_models.py -p ollama llamacpp -n 3  # chosen providers, 3 runs per task
    python scripts/compare_models.py --report-only            # rebuild the report from saved runs

Each run gets a fresh copy of a small fixture project, starts Forge's MCP servers
for it, and runs the full agent loop in auto mode. Results accumulate in
docs/evaluation/model_runs.json, and docs/evaluation/model_comparison.md is
regenerated after every run.

Tasks
  bugfix       fix divide-by-zero, write unittest tests, run them   (filesystem + run_command)
  rag_codegen  write a RAGAS Faithfulness script from the docs       (RAG server + filesystem)
  feature      stricter slugify() + tests in a new tests/ folder     (multi-file, folder creation, discovery)
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from forge.agent import Agent  # noqa: E402
from forge.config import LOG_DIR, Settings, load_server_configs  # noqa: E402
from forge.mcp_client import MCPClientManager  # noqa: E402
from forge.providers import PROVIDERS, ProviderError  # noqa: E402
from forge.tools import ToolRegistry  # noqa: E402

RESULTS_PATH = ROOT / "docs" / "evaluation" / "model_runs.json"
REPORT_PATH = ROOT / "docs" / "evaluation" / "model_comparison.md"

FIXTURE = {
    "calculator.py": '"""Sample code for the assistant to work on during demos."""\n\n\n'
                     "def add(a, b):\n    return a + b\n\n\n"
                     "def divide(a, b):\n    # TODO: handle division by zero\n    return a / b\n",
    "utils/strings.py": '"""String helpers used by the sample project."""\n\n\n'
                        'def slugify(text):\n    return text.lower().strip().replace(" ", "-")\n',
}


# ---------------------------------------------------------------------------- checks
def _python(workspace: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=workspace, capture_output=True, text=True, timeout=60)


def check_bugfix(workspace: Path, tools_used: list[str]) -> dict[str, bool]:
    hidden = ("import sys; sys.path.insert(0, '.'); import calculator as c\n"
              "assert c.add(2, 3) == 5 and c.divide(6, 3) == 2\n"
              "try:\n    c.divide(1, 0)\nexcept ValueError:\n    pass\nelse:\n    raise SystemExit('no ValueError')\n")
    tests = workspace / "test_calculator.py"
    unittest_run = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=workspace,
                                  capture_output=True, text=True, timeout=120)
    test_source = tests.read_text(encoding="utf-8") if tests.exists() else ""
    return {
        "hidden tests pass": _python(workspace, hidden).returncode == 0,
        "add() untouched": "def add(a, b):\n    return a + b" in (workspace / "calculator.py").read_text(encoding="utf-8"),
        "tests cover add, divide, zero case": all(s in test_source for s in ("add(", "divide(", "ValueError")),
        "its tests pass": unittest_run.returncode == 0 and "Ran 0 tests" not in unittest_run.stderr,
        "ran the tests itself": "run_command" in tools_used,
    }


def check_rag(workspace: Path, tools_used: list[str]) -> dict[str, bool]:
    script = workspace / "eval_faithfulness.py"
    source = script.read_text(encoding="utf-8") if script.exists() else ""
    try:
        ast.parse(source)
        parses = bool(source)
    except SyntaxError:
        parses = False
    return {
        "used RAG docs": "rag__search_ragas_docs" in tools_used,
        "script exists and parses": parses,
        "imports ragas": "ragas" in source,
        "uses Faithfulness": "Faithfulness" in source,
        "passes question/answer/context": all(k in source for k in ("user_input", "response", "retrieved_contexts")),
    }


def check_feature(workspace: Path, tools_used: list[str]) -> dict[str, bool]:
    hidden = ("import sys; sys.path.insert(0, '.'); from utils.strings import slugify as s\n"
              "assert s('Hello, World!') == 'hello-world', s('Hello, World!')\n"
              "assert s('  Many   spaces--and---dashes ') == 'many-spaces-and-dashes'\n"
              "assert s('Already-slugged') == 'already-slugged'\n")
    tests_dir = workspace / "tests"
    test_files = list(tests_dir.glob("test*.py")) if tests_dir.is_dir() else []
    # Accept any reasonable way of running the new tests.
    commands = [["-m", "unittest", "discover", "-s", "tests", "-t", "."], ["-m", "unittest", "discover", "-s", "tests"]]
    commands += [["-m", "unittest", f"tests.{f.stem}"] for f in test_files]
    tests_pass = False
    for args in commands:
        run = subprocess.run([sys.executable, *args], cwd=workspace, capture_output=True, text=True, timeout=120)
        if run.returncode == 0 and "Ran 0 tests" not in run.stderr and "Ran " in run.stderr:
            tests_pass = True
            break
    return {
        "hidden tests pass": _python(workspace, hidden).returncode == 0,
        "tests/ folder with a test file": bool(test_files),
        "its tests pass": tests_pass,
        "ran the tests itself": "run_command" in tools_used,
    }


@dataclass
class Task:
    prompt: str
    check: Callable[[Path, list[str]], dict[str, bool]]


TASKS = {
    "bugfix": Task(
        "divide() in calculator.py crashes when b is 0. Make it raise a ValueError with a clear message instead. "
        "Then create test_calculator.py using unittest with tests for add, divide and the zero case, run the "
        "tests, and fix anything that fails.", check_bugfix),
    "rag_codegen": Task(
        "Create eval_faithfulness.py that uses the RAGAS library to compute the Faithfulness score for one "
        "example (a question, an answer and one retrieved context). Use the current RAGAS API. Do not run it.",
        check_rag),
    "feature": Task(
        "Improve slugify() in utils/strings.py: it must lowercase the text, remove punctuation, and turn any run "
        "of spaces and/or hyphens into a single hyphen, with no leading or trailing hyphens. Add unittest tests in "
        "a new tests/ folder (tests/test_strings.py), run them, and fix anything that fails.", check_feature),
}


# ---------------------------------------------------------------------------- headless UI
class HeadlessUI:
    """Records tool calls; prints one line per call so long runs show progress."""

    def __init__(self):
        self.tools: list[str] = []
        self.errors = 0
        self.failures: list[str] = []    # "tool: first line of the error", for the failure analysis

    def on_llm_start(self, step): pass
    def on_reasoning(self, text): pass
    def on_text(self, text): pass
    def on_llm_end(self): pass
    def on_notice(self, text): print(f"      ! {text}")
    def on_tool_start(self, call): pass

    def on_tool_call(self, call):
        self.tools.append(call.tool.name)

    def on_tool_result(self, call, output):
        self.errors += output.is_error
        if output.is_error:
            first = next((line for line in output.text.splitlines() if line.strip()), "")[:160]
            self.failures.append(f"{call.tool.short_name if call else '?'}: {first}")
        print(f"      {'✘' if output.is_error else '✔'} {call.tool.name if call else '?'}")

    async def confirm(self, call, preview):
        return "yes"   # auto mode never asks, but be safe


@dataclass
class RunResult:
    provider: str
    model: str
    task: str
    started: str
    success: bool
    checks: dict[str, bool]
    finished: bool
    steps: int
    tool_calls: int
    failed_tool_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float
    tool_sequence: list[str] = field(default_factory=list)
    tool_failures: list[str] = field(default_factory=list)
    error: str | None = None


async def run_once(provider: str, model: str | None, task_name: str) -> RunResult:
    task = TASKS[task_name]
    workspace = Path(tempfile.mkdtemp(prefix=f"forge_bench_{task_name}_"))
    for rel, content in FIXTURE.items():
        (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
        (workspace / rel).write_text(content, encoding="utf-8")

    settings = Settings.from_env(workspace)
    settings.provider, settings.model, settings.mode = provider, model, "auto"
    configs = load_server_configs(settings.servers_file, {"WORKSPACE": str(workspace), "PROJECT_ROOT": str(ROOT),
                                                          "PYTHON": sys.executable})
    manager = MCPClientManager(configs, LOG_DIR / "bench")
    ui = HeadlessUI()
    started = dt.datetime.now().isoformat(timespec="seconds")
    try:
        await manager.connect_all()
        registry = ToolRegistry(manager, settings)
        agent = Agent(registry, settings, {n for n, c in manager.connections.items() if c.connected})
        start = time.perf_counter()
        error = None
        try:
            stats = await agent.run_turn(task.prompt, ui)
        except Exception as exc:  # provider errors count as a failed run
            stats, error = None, f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start
        checks = task.check(workspace, ui.tools)
        return RunResult(
            provider=provider, model=agent.model_name, task=task_name, started=started,
            success=all(checks.values()) and error is None, checks=checks,
            finished=bool(stats and stats.finished), steps=stats.steps if stats else 0,
            tool_calls=len(ui.tools), failed_tool_calls=ui.errors,
            input_tokens=stats.input_tokens if stats else 0, output_tokens=stats.output_tokens if stats else 0,
            seconds=round(seconds, 1), tool_sequence=ui.tools, tool_failures=ui.failures, error=error)
    finally:
        await manager.close()
        shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------- report
def write_report(runs: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["provider"], run["model"], run["task"]), []).append(run)

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    summary = ["| Task | Provider · model | Runs | Success | Checks passed | Steps | Tool calls (failed) | "
               "Tokens in / out | Time (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for (provider, model, task), rows in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][0])):
        checks = sum(sum(r["checks"].values()) for r in rows)
        total = sum(len(r["checks"]) for r in rows)
        summary.append(
            f"| {task} | {provider} · {model} | {len(rows)} | {sum(r['success'] for r in rows)}/{len(rows)} | "
            f"{checks}/{total} | {mean(rows, 'steps'):.1f} | {mean(rows, 'tool_calls'):.1f} "
            f"({mean(rows, 'failed_tool_calls'):.1f}) | {mean(rows, 'input_tokens'):,.0f} / "
            f"{mean(rows, 'output_tokens'):,.0f} | {mean(rows, 'seconds'):.1f} |")

    details = []
    for run in runs:
        failed = [name for name, ok in run["checks"].items() if not ok]
        details.append(f"- **{run['task']}** · {run['provider']} · {run['model']} · {run['started']}: "
                       f"{'✅ success' if run['success'] else '❌ failed: ' + (', '.join(failed) or run['error'] or '')}"
                       f"  \n  `{' → '.join(t.split('__')[-1] for t in run['tool_sequence']) or '(no tool calls)'}`"
                       + "".join(f"  \n  ↳ {f}" for f in sorted(set(run.get("tool_failures", [])))[:4]))

    REPORT_PATH.write_text(f"""# Model comparison

Generated by `python scripts/compare_models.py` (last updated {dt.date.today().isoformat()}).

Every run: a fresh copy of the fixture project, all MCP servers connected, the full Forge agent loop in **auto**
mode, and scoring by automatic checks (hidden tests, file inspection, tool trace), not by the model's own summary.
Averages are per run.

{chr(10).join(summary)}

## Tasks and checks

**bugfix**: {TASKS['bugfix'].prompt}
Checks: hidden tests pass (add works, divide works, divide by zero raises ValueError) · add() left untouched ·
tests cover add, divide and the ValueError case · the model's own tests pass · the model ran the tests itself.

**rag_codegen**: {TASKS['rag_codegen'].prompt}
Checks: called the RAG server · script exists and parses · imports ragas · uses Faithfulness ·
passes user_input / response / retrieved_contexts.

**feature**: {TASKS['feature'].prompt}
Checks: hidden tests pass (punctuation, repeated spaces/hyphens, already-slugged input) · tests/ folder with a
test file · the model's tests pass · the model ran the tests itself.

## Individual runs (tool sequence)

{chr(10).join(details)}
""", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--providers", nargs="+", help="providers (provider or provider:model)")
    parser.add_argument("-t", "--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("-n", "--runs", type=int, default=1, help="runs per provider per task")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    runs = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else []
    if not args.report_only:
        refs = args.providers or [name for name, spec in PROVIDERS.items() if not spec.missing_env()]
        for ref in refs:
            provider, _, model = ref.partition(":")
            for task_name in args.tasks:
                for i in range(args.runs):
                    print(f"[{provider}{':' + model if model else ''}] {task_name} run {i + 1}/{args.runs}")
                    try:
                        result = await run_once(provider, model or None, task_name)
                    except ProviderError as exc:
                        print(f"  skipped: {exc}")
                        break
                    verdict = "SUCCESS" if result.success else f"FAILED {result.checks} {result.error or ''}"
                    print(f"  -> {verdict} in {result.seconds}s, {result.steps} steps")
                    runs.append(asdict(result))
                    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    RESULTS_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    write_report(runs)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
