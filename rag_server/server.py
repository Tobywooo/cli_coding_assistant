"""Forge RAG MCP server: exposes fusion retrieval over the RAGAS docs as an MCP tool.

Run by Forge's MCP client over stdio (see mcp_servers.json):
    python -m rag_server.server

Requires the one-time ingestion first: python -m rag_server.ingest
stdout is reserved for MCP JSON-RPC messages, so all logging goes to stderr.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .retrieval import FusionRetriever
from .store import MANIFEST_PATH

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forge-rag")

mcp = MCPServer(
    "forge-rag",
    version="0.1.0",
    instructions="Search the official RAGAS documentation (LLM/RAG evaluation library) with fusion retrieval.",
)

_retriever: FusionRetriever | None = None
_load_error: str | None = None


def _get_retriever() -> FusionRetriever | None:
    """Load the vector store and build BM25 once, on first use."""
    global _retriever, _load_error
    if _retriever is None and _load_error is None:
        try:
            _retriever = FusionRetriever()
            log.info("Loaded %d chunks; BM25 index built", len(_retriever))
        except Exception as exc:  # most likely: ingestion hasn't been run yet
            _load_error = f"{type(exc).__name__}: {exc}"
            log.error("Could not load the vector store: %s", _load_error)
    return _retriever


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def search_ragas_docs(query: str, k: int = 5,
                      method: Literal["fusion", "vector", "bm25"] = "fusion") -> str:
    """Search the official RAGAS documentation (the Python library for evaluating LLM and RAG applications:
    metrics such as Faithfulness or ContextPrecision, SingleTurnSample / EvaluationDataset, evaluate(),
    test-set generation, integrations). Returns the most relevant documentation chunks with their source page and
    section. Use specific queries that include class or function names when you know them.

    Args:
        query: what to look up, e.g. "Faithfulness metric example with SingleTurnSample"
        k: number of chunks to return (1-10)
        method: "fusion" (default: BM25 keyword + vector search combined), "vector" or "bm25" only
    """
    retriever = _get_retriever()
    if retriever is None:
        return (f"Error: the RAGAS vector store is not available ({_load_error}). "
                "Build it once with: python -m rag_server.ingest")
    k = max(1, min(int(k), 10))
    hits = retriever.search(query, k=k, method=method)

    explain = {"fusion": "0.5 × vector + 0.5 × BM25", "vector": "vector similarity only", "bm25": "BM25 only"}
    parts = [f'Top {len(hits)} RAGAS doc chunks for "{query}" (method: {method}, {explain[method]})']
    for rank, hit in enumerate(hits, 1):
        parts.append(f"\n[{rank}] {hit.source} › {hit.section}\n"
                     f"    score {hit.score:.2f} (vector {hit.vector:.2f}, bm25 {hit.bm25:.2f})\n"
                     f"{hit.text.split(chr(10), 1)[-1].strip()}")   # drop the "[page › section]" label line
    return "\n".join(parts)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def ragas_docs_info() -> str:
    """Describe the indexed RAGAS documentation: source commit, number of pages and chunks, embedding model."""
    if not MANIFEST_PATH.exists():
        return "The RAGAS docs have not been ingested yet. Run: python -m rag_server.ingest"
    return json.dumps(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), indent=2)


if __name__ == "__main__":
    _get_retriever()   # warm up before the first request so the first search is fast
    mcp.run("stdio")
