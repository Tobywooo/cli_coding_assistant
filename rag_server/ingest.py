"""One-time ingestion: RAGAS docs -> chunks -> embeddings -> persistent Chroma collection.

    python -m rag_server.ingest            # builds the store (skips if it already exists)
    python -m rag_server.ingest --rebuild  # re-fetch nothing, re-chunk and re-embed everything

Pipeline
  1. Load:   sparse `git clone` of the RAGAS repo, Markdown under docs/ only
  2. Clean:  drop mkdocstrings directives (`::: ragas.x`), which are empty outside the docs site
  3. Split:  by Markdown headers (#, ##, ###) so chunks follow the doc structure, then a
             recursive character splitter (~1200 chars, 150 overlap) for long sections
  4. Label:  each chunk starts with "[page › section path]" so both BM25 and the embedding
             see where it came from (helps short chunks like code examples)
  5. Embed + store: Chroma computes MiniLM embeddings and persists them to rag_server/data/chroma

The BM25 index isn't persisted: the server rebuilds it from the stored chunks
at startup, which takes well under a second for this corpus.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .store import (CHROMA_DIR, COLLECTION_NAME, DATA_DIR, DOCS_DIR, EMBEDDING_MODEL, MANIFEST_PATH, REPO_DIR,
                    REPO_URL, open_collection)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 80      # skip fragments with almost no content (e.g. a lone heading)
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def log(message: str) -> None:
    print(message, flush=True)


def fetch_docs() -> str:
    """Sparse-clone just the Markdown docs of the RAGAS repo; return the commit hash."""
    if not (REPO_DIR / ".git").exists():
        log(f"Cloning RAGAS docs from {REPO_URL} (sparse, Markdown only)...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none", "--sparse",
                        REPO_URL, str(REPO_DIR)], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "sparse-checkout", "set", "--no-cone", "/docs/**/*.md"],
                       check=True)
    else:
        log(f"Using existing checkout at {REPO_DIR}")
    return subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def clean_markdown(text: str) -> str:
    text = re.sub(r"^:::.*$", "", text, flags=re.MULTILINE)         # mkdocstrings directives
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)          # HTML comments
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_document(source: str, text: str) -> list[tuple[str, str, dict]]:
    """Split one Markdown page into (id, labelled text, metadata) chunks."""
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n```\n", "\n\n", "\n", " ", ""])   # prefer breaking between code blocks / paragraphs

    page_title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
                      source.rsplit("/", 1)[-1].removesuffix(".md"))
    chunks = []
    for section in header_splitter.split_text(text):
        section_path = " > ".join(section.metadata[key] for _, key in HEADERS if key in section.metadata)
        for piece in char_splitter.split_text(section.page_content):
            if len(re.sub(r"\s+", "", piece)) < MIN_CHUNK_CHARS:
                continue
            index = len(chunks)
            label = f"[{source} › {section_path or page_title}]"
            chunks.append((f"{source}::{index}", f"{label}\n{piece}",
                           {"source": source, "section": section_path or page_title, "title": page_title,
                            "chunk": index}))
    return chunks


def build(rebuild: bool = False) -> None:
    if MANIFEST_PATH.exists() and CHROMA_DIR.exists() and not rebuild:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        log(f"Vector store already built ({manifest['chunks']} chunks from {manifest['documents']} pages, "
            f"{manifest['built_at']}). Use --rebuild to rebuild it.")
        return

    start = time.perf_counter()
    commit = fetch_docs()

    pages = sorted(DOCS_DIR.rglob("*.md"))
    all_chunks: list[tuple[str, str, dict]] = []
    used_pages = 0
    for path in pages:
        source = path.relative_to(REPO_DIR).as_posix()
        text = clean_markdown(path.read_text(encoding="utf-8", errors="replace"))
        chunks = chunk_document(source, text)
        used_pages += bool(chunks)
        all_chunks.extend(chunks)
    log(f"Loaded {len(pages)} pages -> {len(all_chunks)} chunks ({used_pages} pages had content)")

    client, _ = open_collection(create=True)
    if rebuild:
        client.delete_collection(COLLECTION_NAME)
    _, collection = open_collection(create=True)

    log(f"Embedding with {EMBEDDING_MODEL} (first run downloads the model)...")
    batch = 128
    for i in range(0, len(all_chunks), batch):
        ids, docs, metas = zip(*all_chunks[i:i + batch])
        collection.upsert(ids=list(ids), documents=list(docs), metadatas=list(metas))
        log(f"  {min(i + batch, len(all_chunks))}/{len(all_chunks)}")

    manifest = {
        "collection": COLLECTION_NAME, "source_repo": REPO_URL, "source_commit": commit,
        "documents": used_pages, "chunks": len(all_chunks), "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP, "embedding_model": EMBEDDING_MODEL,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "build_seconds": round(time.perf_counter() - start, 1),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Done in {manifest['build_seconds']}s. Store: {CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RAGAS docs vector store (one-time).")
    parser.add_argument("--rebuild", action="store_true", help="drop and rebuild the collection")
    build(parser.parse_args().rebuild)
    sys.exit(0)
