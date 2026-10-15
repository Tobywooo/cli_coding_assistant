"""The agentic loop: reason -> act -> observe -> repeat.

One call to `run_turn(task)`:
  1. appends the user's task to the conversation
  2. streams an LLM response (with every tool's schema bound to the model)
  3. if the response contains tool calls, executes each one through the
     ToolRegistry (which handles confirmation) and appends the results
  4. goes back to 2, until the model answers without tool calls or the
     step budget runs out

The loop knows nothing about terminals: it reports progress through the
AgentUI callbacks, so the same loop powers the CLI and the benchmark script.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import (
    AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.messages.utils import message_chunk_to_message

from .config import Settings
from .prompts import build_system_prompt
from .providers import create_chat_model
from .tools import ToolRegistry, ToolUI


class AgentUI(ToolUI, Protocol):
    """Callbacks the loop uses to report progress (implemented by the CLI)."""

    def on_llm_start(self, step: int) -> None: ...
    def on_reasoning(self, text: str) -> None: ...
    def on_text(self, text: str) -> None: ...
    def on_llm_end(self) -> None: ...
    def on_notice(self, text: str) -> None: ...


@dataclass
class TurnStats:
    steps: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    finished: bool = False      # True if the model gave a final answer within the step budget


class Agent:
    def __init__(self, registry: ToolRegistry, settings: Settings, connected_servers: set[str]):
        self.registry = registry
        self.settings = settings
        self.system_prompt = build_system_prompt(settings.workspace, connected_servers)
        self.history: list[BaseMessage] = []
        self.set_model(settings.provider, settings.model)

    # ------------------------------------------------------------------ configuration
    def set_model(self, provider: str, model: str | None = None) -> None:
        """Switch provider/model. History is kept, so you can change models mid-conversation."""
        chat_model, resolved = create_chat_model(provider, model)
        self.provider, self.model_name = provider, resolved
        self._model = chat_model.bind_tools(self.registry.schemas())

    def reset(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------ the loop
    async def run_turn(self, task: str, ui: AgentUI) -> TurnStats:
        stats = TurnStats()
        start = time.perf_counter()
        self.history.append(HumanMessage(task))
        try:
            for step in range(1, self.settings.max_steps + 1):
                stats.steps = step
                ai = await self._call_llm(step, ui, stats)
                self.history.append(ai)

                if not ai.tool_calls and not ai.invalid_tool_calls:
                    stats.finished = True        # no tool calls means this is the final answer
                    break

                # Act + observe: run every requested tool call, in order.
                for tool_call in ai.tool_calls:
                    _, output = await self.registry.execute(tool_call["id"], tool_call["name"], tool_call["args"], ui)
                    stats.tool_calls += 1
                    stats.failed_tool_calls += output.is_error
                    self.history.append(ToolMessage(
                        content=self.registry.for_llm(output), tool_call_id=tool_call["id"],
                        name=tool_call["name"], status="error" if output.is_error else "success"))

                # Malformed calls (unparseable JSON arguments) get an error result so the model can retry.
                for bad in ai.invalid_tool_calls:
                    stats.failed_tool_calls += 1
                    ui.on_notice(f"Model produced malformed arguments for {bad.get('name')}; asking it to retry.")
                    self.history.append(ToolMessage(
                        content=f"Error: could not parse the arguments for {bad.get('name')}: {bad.get('error')}. "
                                "Call the tool again with valid JSON arguments.",
                        tool_call_id=bad["id"], name=bad.get("name") or "unknown", status="error"))
            else:
                ui.on_notice(f"Stopped after {self.settings.max_steps} steps (FORGE_MAX_STEPS). "
                             "Say 'continue' to let Forge keep going.")
        finally:
            # Runs on Ctrl+C too: keep the history valid for the next turn.
            self._close_dangling_tool_calls()
            stats.duration_s = time.perf_counter() - start
        return stats

    async def _call_llm(self, step: int, ui: AgentUI, stats: TurnStats) -> AIMessage:
        """Stream one model response, forwarding reasoning and text to the UI as they arrive."""
        messages = [SystemMessage(self.system_prompt), *self.history]
        ui.on_llm_start(step)
        full: AIMessageChunk | None = None
        try:
            async for chunk in self._model.astream(messages):
                if reasoning := chunk.additional_kwargs.get("reasoning_content"):
                    ui.on_reasoning(reasoning)
                if text := chunk.text:
                    ui.on_text(text)
                full = chunk if full is None else full + chunk
        finally:
            ui.on_llm_end()

        if full is None:
            return AIMessage(content="")
        if full.usage_metadata:
            stats.input_tokens += full.usage_metadata.get("input_tokens", 0)
            stats.output_tokens += full.usage_metadata.get("output_tokens", 0)
        message = message_chunk_to_message(full)
        # Every tool call needs an id so its result can be matched to it; some servers omit them.
        for call in [*message.tool_calls, *message.invalid_tool_calls]:
            call["id"] = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        # Don't send reasoning back to the model on later steps; it only wastes context.
        message.additional_kwargs.pop("reasoning_content", None)
        return message

    def _close_dangling_tool_calls(self) -> None:
        """If a turn was interrupted between a tool request and its result, add placeholder
        results: providers reject a history with unanswered tool calls."""
        for index in range(len(self.history) - 1, -1, -1):
            message = self.history[index]
            if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
                answered = {m.tool_call_id for m in self.history[index + 1:] if isinstance(m, ToolMessage)}
                for call in [*message.tool_calls, *message.invalid_tool_calls]:
                    if call["id"] not in answered:
                        self.history.append(ToolMessage(content="Cancelled by the user before it ran.",
                                                        tool_call_id=call["id"], name=call.get("name") or "unknown",
                                                        status="error"))
                return
            if isinstance(message, HumanMessage):
                return
