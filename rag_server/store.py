"""Shared locations and the Chroma collection used by ingestion, retrieval and evaluation."""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_DIR = DATA_DIR / "ragas_repo"          # sparse checkout of the RAGAS repo (docs only)
DOCS_DIR = REPO_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"            # persistent vector database
MANIFEST_PATH = DATA_DIR / "manifest.json"  # what was ingested, when, from which commit

REPO_URL = "https://github.com/explodinggradients/ragas.git"
COLLECTION_NAME = "ragas_docs"
# Chroma's built-in embedding model: all-MiniLM-L6-v2 run through ONNX Runtime.
# Fully local, no PyTorch needed, downloaded once (~80 MB) to ~/.cache/chroma.
EMBEDDING_MODEL = "all-MiniLM-L6-v2 (ONNX, chromadb default)"


def open_collection(create: bool = False):
    """Open (or create) the persistent Chroma collection with cosine distance."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False))
    embedding = DefaultEmbeddingFunction()
    if create:
        return client, client.get_or_create_collection(
            COLLECTION_NAME, embedding_function=embedding, metadata={"hnsw:space": "cosine"})
    return client, client.get_collection(COLLECTION_NAME, embedding_function=embedding)
