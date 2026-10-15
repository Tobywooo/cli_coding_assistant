"""Retrievers over the stored RAGAS chunks: vector, BM25, and fusion (the advanced technique).

Fusion retrieval (NirDiamant/RAG_Techniques, "Fusion Retrieval"):
  * vector search is good at meaning ("how do I check the answer is grounded?")
    but weak on exact identifiers (`LLMContextPrecisionWithReference`)
  * BM25 is the opposite: exact terms score highly, paraphrases miss
  * so every chunk gets both scores, each min-max normalised to [0, 1], and
        fused = alpha * vector + (1 - alpha) * bm25        (alpha = 0.5 by default)

Implementation note: the reference notebook takes vector results in similarity order
and BM25 scores in corpus order, then adds the two arrays position by position, so a
chunk's vector score can be added to a different chunk's BM25 score. Here both scores
are keyed by chunk id before fusing. Reciprocal Rank Fusion (`rrf`) is included as
a rank-based alternative for the evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from .store import open_collection

Method = Literal["fusion", "vector", "bm25", "rrf"]

_STOPWORDS = set("""a an and are as at be by can do does for from how i if in into is it of on or so that the
their then there these this to use used using what when where which who why will with you your""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps identifiers like single_turn_sample / SingleTurnSample intact."""
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _min_max(scores: np.ndarray) -> np.ndarray:
    low, high = float(scores.min()), float(scores.max())
    return np.zeros_like(scores) if high - low < 1e-12 else (scores - low) / (high - low)


@dataclass
class Hit:
    id: str
    text: str
    source: str
    section: str
    score: float           # score for the chosen method
    vector: float          # normalised vector similarity (0-1)
    bm25: float            # normalised BM25 score (0-1)


class FusionRetriever:
    def __init__(self):
        _, self.collection = open_collection()
        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = stored["ids"]
        self.docs: list[str] = stored["documents"]
        self.metas: list[dict] = stored["metadatas"]
        self.index = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        # BM25 is cheap to build (1.6k chunks in well under a second), so it isn't persisted.
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.docs])

    def __len__(self) -> int:
        return len(self.ids)

    # ------------------------------------------------------------------ raw scores (corpus order)
    def vector_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of every chunk to the query, aligned to self.ids by chunk id."""
        result = self.collection.query(query_texts=[query], n_results=len(self.ids), include=["distances"])
        scores = np.zeros(len(self.ids))
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            scores[self.index[chunk_id]] = 1.0 - distance        # cosine distance -> similarity
        return scores

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 5, method: Method = "fusion", alpha: float = 0.5) -> list[Hit]:
        vector = _min_max(self.vector_scores(query))
        bm25 = _min_max(self.bm25_scores(query))

        if method == "vector":
            final = vector
        elif method == "bm25":
            final = bm25
        elif method == "rrf":
            final = self._rrf(vector, bm25)
        else:
            final = alpha * vector + (1 - alpha) * bm25

        top = np.argsort(-final)[:k]
        return [Hit(id=self.ids[i], text=self.docs[i], source=self.metas[i].get("source", ""),
                    section=self.metas[i].get("section", ""), score=float(final[i]),
                    vector=float(vector[i]), bm25=float(bm25[i])) for i in top]

    @staticmethod
    def _rrf(*score_lists: np.ndarray, k: int = 60) -> np.ndarray:
        """Reciprocal Rank Fusion: sum of 1 / (k + rank) across the rankings."""
        fused = np.zeros_like(score_lists[0])
        for scores in score_lists:
            ranks = np.empty(len(scores))
            ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
            fused += 1.0 / (k + ranks)
        return fused
