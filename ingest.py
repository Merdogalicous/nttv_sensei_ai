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

from nttv_chatbot.document_parsing import (
    DocumentParseSkipped,
    ParserSettings,
    SUPPORTED_INGEST_EXTENSIONS,
    StructuredDocument,
    StructuredDocumentElement,
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

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


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


def simple_chunk_text(
    text: str,
    source: str,
    meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Naive character-based chunking with overlap."""

    base_meta = {"priority": _priority_for_source(source), "source": source}
    if meta:
        base_meta.update(meta)

    chunks: List[Dict[str, Any]] = []
    start = 0
    text = text or ""
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "meta": dict(base_meta),
                }
            )

        if end == n:
            break
        start = end - CHUNK_OVERLAP

    return chunks


def simple_chunk_document(document: StructuredDocument, source: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    buffered_elements: List[StructuredDocumentElement] = []
    buffered_parts: List[str] = []
    buffered_chars = 0

    def flush_buffer() -> None:
        nonlocal buffered_elements, buffered_parts, buffered_chars
        if not buffered_parts:
            return

        block_text = "\n\n".join(buffered_parts).strip()
        if block_text:
            chunks.extend(
                simple_chunk_text(
                    block_text,
                    source=source,
                    meta=_merge_element_metadata(source, buffered_elements, document.parser_name, document.file_type),
                )
            )

        buffered_elements = []
        buffered_parts = []
        buffered_chars = 0

    for element in document.elements:
        block_text = _element_block_text(element)
        if not block_text:
            continue

        if len(block_text) > CHUNK_SIZE:
            flush_buffer()
            chunks.extend(
                simple_chunk_text(
                    block_text,
                    source=source,
                    meta=_merge_element_metadata(source, [element], document.parser_name, document.file_type),
                )
            )
            continue

        projected_chars = buffered_chars + len(block_text) + (2 if buffered_parts else 0)
        if buffered_parts and projected_chars > CHUNK_SIZE:
            flush_buffer()

        buffered_elements.append(element)
        buffered_parts.append(block_text)
        buffered_chars += len(block_text) + (2 if len(buffered_parts) > 1 else 0)

    flush_buffer()
    return chunks


def parse_and_chunk_files(
    files: Iterable[Path],
    *,
    root: Path = ROOT,
    parser_settings: Optional[ParserSettings] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, str]]]:
    parser_settings = parser_settings or ParserSettings.from_env()

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

        file_chunks = simple_chunk_document(parsed_document, source=source)
        print(f"  -> {len(file_chunks)} chunks")

        if file_chunks:
            all_chunks.extend(file_chunks)
            ingested_files.append(source)
        else:
            skipped_files.append({"path": source, "reason": "Parser returned no chunkable text."})
            print("  -> skipped: parser returned no chunkable text")

    return all_chunks, ingested_files, skipped_files


def _priority_for_source(source: str) -> int:
    lower_source = source.lower()
    if "glossary" in lower_source:
        return 3
    if "rank" in lower_source:
        return 3
    if "technique description" in lower_source or "technique_descriptions" in lower_source:
        return 3
    if "kihon" in lower_source or "sanshin" in lower_source:
        return 2
    return 1


def _element_block_text(element: StructuredDocumentElement) -> str:
    text = (element.text or "").strip()
    if not text:
        return ""

    if element.element_type == "heading":
        return text

    if element.heading_path:
        heading_context = " > ".join(element.heading_path).strip()
        if heading_context:
            return f"{heading_context}\n\n{text}".strip()

    return text


def _merge_element_metadata(
    source: str,
    elements: List[StructuredDocumentElement],
    parser_name: str,
    file_type: str,
) -> Dict[str, Any]:
    pages = [page for element in elements for page in (element.page_start, element.page_end) if page is not None]
    heading_path = _common_heading_path([element.heading_path for element in elements])
    element_types = [element.element_type for element in elements if element.element_type]
    unique_element_types = list(dict.fromkeys(element_types))

    raw_metadata: Dict[str, Any]
    if len(elements) == 1:
        raw_metadata = dict(elements[0].raw_metadata)
    else:
        raw_metadata = {
            "element_count": len(elements),
            "element_types": unique_element_types,
            "pages": sorted(set(pages)),
        }

    page_start = min(pages) if pages else None
    page_end = max(pages) if pages else None

    return {
        "priority": _priority_for_source(source),
        "source": source,
        "parser": parser_name,
        "file_type": file_type,
        "page": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "heading_path": heading_path,
        "element_type": unique_element_types[0] if len(unique_element_types) == 1 else "mixed",
        "raw_metadata": raw_metadata,
    }


def _common_heading_path(paths: List[List[str]]) -> List[str]:
    non_empty_paths = [path for path in paths if path]
    if not non_empty_paths:
        return []

    prefix = list(non_empty_paths[0])
    for path in non_empty_paths[1:]:
        match_len = 0
        for left, right in zip(prefix, path):
            if left != right:
                break
            match_len += 1
        prefix = prefix[:match_len]
        if not prefix:
            break

    return prefix or list(non_empty_paths[-1])


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

    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INDEX_DIR: {INDEX_DIR}")
    print(
        "PDF settings: "
        f"enabled={parser_settings.enable_pdf_ingest}, "
        f"parser={parser_settings.pdf_parser}, "
        f"max_pages={parser_settings.pdf_parse_max_pages}, "
        f"fail_open={parser_settings.pdf_fail_open}"
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
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": ingested_files,
        "skipped_files": skipped_files,
        "parsing": {
            "enable_pdf_ingest": parser_settings.enable_pdf_ingest,
            "pdf_parser": parser_settings.pdf_parser,
            "pdf_parse_max_pages": parser_settings.pdf_parse_max_pages,
            "pdf_fail_open": parser_settings.pdf_fail_open,
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
