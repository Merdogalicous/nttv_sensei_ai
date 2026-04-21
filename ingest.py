"""
Ingest content files, chunk them, embed them, and build a FAISS index.

Usage (locally):
    python ingest.py

On Render:
    Make sure this runs in the build command so the index exists
    before the app starts.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer

from nttv_chatbot.chunking import ChunkingSettings, build_chunks
from nttv_chatbot.document_parsing import (
    DocumentParseSkipped,
    ParserSettings,
    SUPPORTED_INGEST_EXTENSIONS,
    parse_file,
)


# ---------------------------
# Paths & constants
# ---------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Index directory:
# - Locally: defaults to <repo>/index
# - On Render (or other hosts): set INDEX_DIR env var, e.g. /var/data/index
DEFAULT_INDEX_DIR = ROOT / "index"
INDEX_DIR = Path(os.getenv("INDEX_DIR") or DEFAULT_INDEX_DIR)

INDEX_DIR.mkdir(exist_ok=True, parents=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CONFIG_PATH = INDEX_DIR / "config.json"
META_PATH = INDEX_DIR / "meta.pkl"
FAISS_PATH = INDEX_DIR / "index.faiss"
FAISS_PATH_LEGACY = INDEX_DIR / "faiss.index"

# ---------------------------
# Utilities
# ---------------------------

def read_text_file(path: Path, parser_settings: Optional[ParserSettings] = None) -> str:
    parsed = parse_file(path, settings=parser_settings)
    return "\n\n".join(element.text for element in parsed.elements if (element.text or "").strip())


def iter_source_files() -> List[Path]:
    files: List[Path] = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_INGEST_EXTENSIONS:
            files.append(path)
    files.sort()
    return files

def parse_and_chunk_files(
    files: Iterable[Path],
    *,
    root: Path = ROOT,
    parser_settings: Optional[ParserSettings] = None,
    chunking_settings: Optional[ChunkingSettings] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, str]]]:
    parser_settings = parser_settings or ParserSettings.from_env()
    chunking_settings = chunking_settings or ChunkingSettings.from_env()

    all_chunks: List[Dict[str, Any]] = []
    ingested_files: List[str] = []
    skipped_files: List[Dict[str, str]] = []

    for file_path in files:
        source = _relative_source(file_path, root)
        print(f"\nReading {file_path} ...")

        try:
            parsed_document = parse_file(file_path, settings=parser_settings)
        except DocumentParseSkipped as exc:
            print(f"  -> skipped: {exc}")
            skipped_files.append({"path": source, "reason": str(exc)})
            continue
        except Exception as exc:
            print(f"  -> warning: failed to parse {source}: {exc}")
            skipped_files.append({"path": source, "reason": str(exc)})
            continue

        parsed_length = sum(len(element.text or "") for element in parsed_document.elements)
        print(f"  Parser: {parsed_document.parser_name}")
        print(f"  Elements: {len(parsed_document.elements)}")
        print(f"  Length: {parsed_length} characters")

        file_chunks = build_chunks(
            parsed_document,
            source_file=source,
            settings=chunking_settings,
        )
        print(f"  -> {len(file_chunks)} chunks")

        if file_chunks:
            all_chunks.extend(file_chunks)
            ingested_files.append(source)
        else:
            skipped_files.append({"path": source, "reason": "Parser returned no chunkable text."})
            print("  -> skipped: parser returned no chunkable text")

    return all_chunks, ingested_files, skipped_files


def _relative_source(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------
# Embeddings & index build
# ---------------------------

def embed_chunks(model: SentenceTransformer, chunks: List[Dict[str, Any]]) -> np.ndarray:
    texts = [(chunk.get("text") or "") for chunk in chunks]
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype="float32")
    return emb


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main() -> None:
    parser_settings = ParserSettings.from_env()
    chunking_settings = ChunkingSettings.from_env()

    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INDEX_DIR: {INDEX_DIR}")
    print(
        "PDF settings: "
        f"enabled={parser_settings.enable_pdf_ingest}, "
        f"parser={parser_settings.pdf_parser}, "
        f"max_pages={parser_settings.pdf_parse_max_pages}, "
        f"fail_open={parser_settings.pdf_fail_open}"
    )
    print(
        "Chunk settings: "
        f"target_tokens={chunking_settings.target_tokens}, "
        f"max_tokens={chunking_settings.max_tokens}, "
        f"overlap_tokens={chunking_settings.overlap_tokens}, "
        f"min_tokens={chunking_settings.min_tokens}"
    )

    files = iter_source_files()
    if not files:
        raise RuntimeError(f"No source files found in {DATA_DIR}")

    print("Found source files:")
    for file_path in files:
        print(" -", file_path.relative_to(ROOT))

    all_chunks, ingested_files, skipped_files = parse_and_chunk_files(
        files,
        root=ROOT,
        parser_settings=parser_settings,
        chunking_settings=chunking_settings,
    )

    print(f"\nTotal chunks (pre-filter): {len(all_chunks)}")

    filtered_chunks: List[Dict[str, Any]] = []
    dropped_empty = 0
    for chunk in all_chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            dropped_empty += 1
            continue
        filtered_chunks.append(chunk)

    if dropped_empty:
        print(f"Dropped {dropped_empty} empty chunks")

    all_chunks = filtered_chunks
    print(f"Total chunks (post-filter): {len(all_chunks)}")

    if not all_chunks:
        raise RuntimeError(
            "No chunks were created from the available source files. "
            "Check parser settings and source documents."
        )

    print("\nLoading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding chunks...")
    emb = embed_chunks(model, all_chunks)
    print("Embeddings shape:", emb.shape)

    if emb.shape[0] != len(all_chunks):
        raise RuntimeError(
            f"BUG: embeddings/chunks mismatch before index build: "
            f"embeddings={emb.shape[0]} chunks={len(all_chunks)}"
        )

    print("Building FAISS index...")
    index = build_faiss_index(emb)

    if int(index.ntotal) != len(all_chunks):
        raise RuntimeError(
            f"BUG: faiss.ntotal != chunks after add: ntotal={int(index.ntotal)} chunks={len(all_chunks)}"
        )

    print(f"Saving FAISS index to {FAISS_PATH}")
    faiss.write_index(index, str(FAISS_PATH))

    print(f"Saving legacy FAISS index to {FAISS_PATH_LEGACY}")
    faiss.write_index(index, str(FAISS_PATH_LEGACY))

    print(f"Saving metadata to {META_PATH}")
    with META_PATH.open("wb") as handle:
        pickle.dump(all_chunks, handle)

    config = {
        "embedding_model": EMBED_MODEL_NAME,
        "embed_model": EMBED_MODEL_NAME,
        "faiss_path": str(FAISS_PATH),
        "top_k": 6,
        "chunk_size": chunking_settings.max_tokens * 4,
        "chunk_overlap": chunking_settings.overlap_tokens * 4,
        "chunk_target_tokens": chunking_settings.target_tokens,
        "chunk_max_tokens": chunking_settings.max_tokens,
        "chunk_overlap_tokens": chunking_settings.overlap_tokens,
        "chunk_min_tokens": chunking_settings.min_tokens,
        "files": ingested_files,
        "skipped_files": skipped_files,
        "parsing": {
            "enable_pdf_ingest": parser_settings.enable_pdf_ingest,
            "pdf_parser": parser_settings.pdf_parser,
            "pdf_parse_max_pages": parser_settings.pdf_parse_max_pages,
            "pdf_fail_open": parser_settings.pdf_fail_open,
        },
        "chunking": {
            "strategy": "section-aware",
            "target_tokens": chunking_settings.target_tokens,
            "max_tokens": chunking_settings.max_tokens,
            "overlap_tokens": chunking_settings.overlap_tokens,
            "min_tokens": chunking_settings.min_tokens,
        },
        "num_chunks": len(all_chunks),
        "faiss_ntotal": int(index.ntotal),
    }

    print(f"Saving config to {CONFIG_PATH}")
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nIngest complete.")
    print(f"   Chunks written: {len(all_chunks)}")
    print(f"   FAISS ntotal:  {int(index.ntotal)}")
    print(f"   Files parsed:  {len(ingested_files)}")
    print(f"   Files skipped: {len(skipped_files)}")
    print(f"   Index path:    {FAISS_PATH}")
    print(f"   Legacy path:   {FAISS_PATH_LEGACY}")
    print(f"   Meta path:     {META_PATH}")
    print(f"   Config path:   {CONFIG_PATH}")


if __name__ == "__main__":
    main()
