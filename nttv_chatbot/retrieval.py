from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - safe fallback if dependency is missing
    BM25Okapi = None

from extractors.schools import SCHOOL_ALIASES


LOGGER = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
RRF_K = 60
DEFAULT_JINA_API_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_JINA_MODEL = "jina-reranker-v2-base-multilingual"


@dataclass(frozen=True)
class RetrievalSettings:
    use_hybrid_retrieval: bool
    dense_top_k: int
    lexical_top_k: int
    fused_top_k: int
    reranker_backend: str
    jina_api_key: str
    jina_api_url: str
    jina_model: str
    context_max_chars: int = 6000

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        settings = cls(
            use_hybrid_retrieval=_parse_bool_env("USE_HYBRID_RETRIEVAL", True),
            dense_top_k=_int_env("DENSE_TOP_K", 12),
            lexical_top_k=_int_env("LEXICAL_TOP_K", 12),
            fused_top_k=_int_env("FUSED_TOP_K", 10),
            reranker_backend=(os.getenv("RERANKER_BACKEND") or "none").strip().lower() or "none",
            jina_api_key=(os.getenv("JINA_API_KEY") or "").strip(),
            jina_api_url=(os.getenv("JINA_API_URL") or DEFAULT_JINA_API_URL).strip(),
            jina_model=(os.getenv("JINA_RERANK_MODEL") or DEFAULT_JINA_MODEL).strip(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.dense_top_k <= 0:
            raise ValueError("DENSE_TOP_K must be > 0.")
        if self.lexical_top_k <= 0:
            raise ValueError("LEXICAL_TOP_K must be > 0.")
        if self.fused_top_k <= 0:
            raise ValueError("FUSED_TOP_K must be > 0.")
        if self.reranker_backend not in {"none", "heuristic_only", "jina_api"}:
            raise ValueError(
                "RERANKER_BACKEND must be one of: none, heuristic_only, jina_api."
            )


@dataclass
class RetrievalResult:
    dense_candidates: list[dict[str, Any]]
    lexical_candidates: list[dict[str, Any]]
    fused_candidates: list[dict[str, Any]]
    reranked_candidates: list[dict[str, Any]]
    final_candidates: list[dict[str, Any]]
    context: str
    reranker_backend_requested: str
    reranker_backend_used: str
    reranker_fallback_reason: Optional[str] = None

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "dense_candidates": self.dense_candidates,
            "lexical_candidates": self.lexical_candidates,
            "fused_candidates": self.fused_candidates,
            "reranked_candidates": self.reranked_candidates,
            "final_candidates": self.final_candidates,
            "reranker_backend_requested": self.reranker_backend_requested,
            "reranker_backend_used": self.reranker_backend_used,
            "reranker_fallback_reason": self.reranker_fallback_reason,
        }


class LexicalRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = chunks
        self._documents = [_lexical_document_text(chunk) for chunk in chunks]
        self._tokenized_documents = [_tokenize(document) for document in self._documents]
        self._bm25 = None

        if BM25Okapi is not None and self._tokenized_documents:
            self._bm25 = BM25Okapi(self._tokenized_documents)

    @property
    def backend_name(self) -> str:
        return "bm25" if self._bm25 is not None else "token_overlap"

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        query_tokens = _tokenize(question)
        if not query_tokens:
            return []

        if self._bm25 is not None:
            raw_scores = list(self._bm25.get_scores(query_tokens))
        else:
            raw_scores = [
                _token_overlap_score(tokens, query_tokens)
                for tokens in self._tokenized_documents
            ]

        ranked = sorted(
            enumerate(raw_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        candidates: list[dict[str, Any]] = []
        for chunk_index, raw_score in ranked:
            score = float(raw_score)
            if score <= 0:
                continue

            candidate = _candidate_from_chunk(self._chunks[chunk_index], chunk_index)
            candidate["lexical_score"] = score
            candidate["lexical_rank"] = len(candidates) + 1
            candidate["matched_stages"] = ["lexical"]
            candidate["score"] = score
            candidates.append(candidate)

            if len(candidates) >= top_k:
                break

        return candidates


def search(
    question: str,
    *,
    index: Any,
    chunks: list[dict[str, Any]],
    embed_query: Callable[[str], Any],
    final_top_k: int,
    settings: Optional[RetrievalSettings] = None,
    lexical_retriever: Optional[LexicalRetriever] = None,
) -> RetrievalResult:
    settings = settings or RetrievalSettings.from_env()
    settings.validate()

    dense_candidates = dense_retrieve(
        question,
        index=index,
        chunks=chunks,
        embed_query=embed_query,
        top_k=settings.dense_top_k,
    )

    lexical_candidates: list[dict[str, Any]] = []
    if settings.use_hybrid_retrieval:
        lexical = lexical_retriever or LexicalRetriever(chunks)
        lexical_candidates = lexical.search(question, top_k=settings.lexical_top_k)

    fused_candidates = fuse_candidate_rankings(
        dense_candidates,
        lexical_candidates,
        top_k=settings.fused_top_k,
    )

    reranked_by_heuristics = apply_priority_heuristics(question, fused_candidates)
    reranked_candidates, backend_used, fallback_reason = apply_optional_reranker(
        question,
        reranked_by_heuristics,
        settings=settings,
    )

    final_candidates = reranked_candidates[: max(1, final_top_k)]
    context = assemble_context(final_candidates, max_chars=settings.context_max_chars)

    return RetrievalResult(
        dense_candidates=dense_candidates,
        lexical_candidates=lexical_candidates,
        fused_candidates=fused_candidates,
        reranked_candidates=reranked_candidates,
        final_candidates=final_candidates,
        context=context,
        reranker_backend_requested=settings.reranker_backend,
        reranker_backend_used=backend_used,
        reranker_fallback_reason=fallback_reason,
    )


def dense_retrieve(
    question: str,
    *,
    index: Any,
    chunks: list[dict[str, Any]],
    embed_query: Callable[[str], Any],
    top_k: int,
) -> list[dict[str, Any]]:
    ntotal = int(getattr(index, "ntotal", 0) or 0)
    if ntotal <= 0:
        raise RuntimeError("FAISS index is empty (ntotal=0). Re-run ingest.py.")

    if not chunks:
        raise RuntimeError("meta.pkl loaded but contains 0 chunks. Re-run ingest.py.")

    if abs(ntotal - len(chunks)) > 0:
        raise RuntimeError(
            f"Index/meta mismatch: faiss.ntotal={ntotal}, chunks={len(chunks)}.\n"
            "This usually means FAISS and meta.pkl are from different ingest runs."
        )

    want = min(max(1, top_k), ntotal)
    vector = embed_query(question)
    distances, indices = index.search(vector, want)

    candidates: list[dict[str, Any]] = []
    for rank, (chunk_index, raw_score) in enumerate(zip(indices[0], distances[0]), start=1):
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        candidate = _candidate_from_chunk(chunks[chunk_index], chunk_index)
        candidate["dense_score"] = float(raw_score)
        candidate["dense_rank"] = rank
        candidate["matched_stages"] = ["dense"]
        candidate["score"] = float(raw_score)
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "Retrieval returned 0 usable passages. This can happen if FAISS returned only -1 indices "
            "or if index/meta are mismatched.\n"
            "Try: delete index/ and re-run `python ingest.py`."
        )

    return candidates


def fuse_candidate_rankings(
    dense_candidates: list[dict[str, Any]],
    lexical_candidates: list[dict[str, Any]],
    *,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for stage_name, candidates in (
        ("dense", dense_candidates),
        ("lexical", lexical_candidates),
    ):
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate.get("chunk_id")
            if not chunk_id:
                continue

            if chunk_id not in merged:
                merged[chunk_id] = _clone_candidate(candidate)

            target = merged[chunk_id]
            target["matched_stages"] = sorted(
                set(target.get("matched_stages") or []).union(candidate.get("matched_stages") or [])
            )
            target["rrf_score"] = float(target.get("rrf_score") or 0.0) + (1.0 / (rrf_k + rank))

            if stage_name == "dense":
                target["dense_score"] = candidate.get("dense_score")
                target["dense_rank"] = candidate.get("dense_rank")
            else:
                target["lexical_score"] = candidate.get("lexical_score")
                target["lexical_rank"] = candidate.get("lexical_rank")

    fused = list(merged.values())
    fused.sort(
        key=lambda item: (
            float(item.get("rrf_score") or 0.0),
            1 if item.get("dense_rank") is not None else 0,
            _safe_score(item.get("dense_score")),
            _safe_score(item.get("lexical_score")),
        ),
        reverse=True,
    )

    for rank, candidate in enumerate(fused, start=1):
        candidate["fused_rank"] = rank
        candidate["score"] = float(candidate.get("rrf_score") or 0.0)

    return fused[: max(1, top_k)]


def apply_priority_heuristics(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []

    for candidate in candidates:
        heuristic_delta = _heuristic_delta(question, candidate)
        heuristic_score = float(candidate.get("rrf_score") or 0.0) + heuristic_delta

        updated = _clone_candidate(candidate)
        updated["heuristic_delta"] = heuristic_delta
        updated["heuristic_score"] = heuristic_score
        updated["rerank_score"] = heuristic_score
        updated["final_score"] = heuristic_score
        reranked.append(updated)

    reranked.sort(
        key=lambda item: (
            float(item.get("heuristic_score") or 0.0),
            float(item.get("rrf_score") or 0.0),
            _safe_score(item.get("dense_score")),
            _safe_score(item.get("lexical_score")),
        ),
        reverse=True,
    )

    for rank, candidate in enumerate(reranked, start=1):
        candidate["heuristic_rank"] = rank
        candidate["final_rank"] = rank

    return reranked


def apply_optional_reranker(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    settings: RetrievalSettings,
) -> tuple[list[dict[str, Any]], str, Optional[str]]:
    if not candidates:
        return [], settings.reranker_backend, None

    if settings.reranker_backend in {"none", "heuristic_only"}:
        return candidates, settings.reranker_backend, None

    if settings.reranker_backend != "jina_api":
        return candidates, "heuristic_only", "Unsupported reranker backend; used heuristic ordering."

    if not settings.jina_api_key:
        return candidates, "heuristic_only", "JINA_API_KEY is not set; used heuristic ordering."

    rerank_window = min(len(candidates), max(10, settings.fused_top_k))
    rerank_input = candidates[:rerank_window]

    payload = {
        "model": settings.jina_model,
        "query": question,
        "documents": [candidate.get("text") or "" for candidate in rerank_input],
        "top_n": rerank_window,
    }
    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            settings.jina_api_url,
            headers=headers,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        if not isinstance(results, list):
            raise ValueError("Jina reranker returned an unexpected response shape.")

        by_index = {}
        for row in results:
            if not isinstance(row, dict):
                continue
            raw_index = row.get("index")
            if not isinstance(raw_index, int):
                continue
            by_index[raw_index] = float(row.get("relevance_score") or 0.0)

        if not by_index:
            raise ValueError("Jina reranker returned no usable scores.")

        reranked_rows: list[tuple[float, dict[str, Any]]] = []
        for original_index, candidate in enumerate(rerank_input):
            updated = _clone_candidate(candidate)
            reranker_score = by_index.get(original_index, float("-inf"))
            if reranker_score != float("-inf"):
                updated["reranker_score"] = reranker_score
                updated["rerank_score"] = reranker_score
                updated["final_score"] = reranker_score
            reranked_rows.append((reranker_score, updated))

        reranked_rows.sort(
            key=lambda item: (
                item[0],
                float(item[1].get("heuristic_score") or 0.0),
            ),
            reverse=True,
        )

        reranked = [candidate for _, candidate in reranked_rows]
        for rank, candidate in enumerate(reranked, start=1):
            candidate["final_rank"] = rank

        tail = [_clone_candidate(candidate) for candidate in candidates[rerank_window:]]
        for offset, candidate in enumerate(tail, start=len(reranked) + 1):
            candidate["final_rank"] = offset

        return reranked + tail, "jina_api", None
    except Exception as exc:  # pragma: no cover - network path
        LOGGER.warning("Jina rerank failed; falling back to heuristic ordering: %s", exc)
        return candidates, "heuristic_only", f"Jina rerank failed: {exc}"


def assemble_context(snippets: list[dict[str, Any]], max_chars: int = 6000) -> str:
    lines: list[str] = []
    total = 0

    for index, snippet in enumerate(snippets, start=1):
        source = snippet.get("source") or (snippet.get("meta") or {}).get("source") or "unknown"
        tag = f"[{index}] {os.path.basename(source)}"
        page_start = snippet.get("page_start") or snippet.get("page")
        page_end = snippet.get("page_end") or page_start
        if page_start and page_end and page_start != page_end:
            tag += f" (pp. {page_start}-{page_end})"
        elif page_start:
            tag += f" (p. {page_start})"

        block = f"{tag}\n{snippet.get('text') or ''}\n\n---\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)

    return "".join(lines)


def _candidate_from_chunk(chunk: dict[str, Any], chunk_index: int) -> dict[str, Any]:
    meta = dict(chunk.get("meta") or {})
    source = meta.get("source") or chunk.get("source") or ""
    chunk_id = meta.get("chunk_id") or chunk.get("chunk_id") or f"chunk-{chunk_index}"

    return {
        "text": chunk.get("text", ""),
        "meta": meta,
        "source": source,
        "page": meta.get("page"),
        "page_start": meta.get("page_start", meta.get("page")),
        "page_end": meta.get("page_end", meta.get("page")),
        "heading_path": meta.get("heading_path") or [],
        "priority_bucket": meta.get("priority_bucket"),
        "chunk_id": chunk_id,
        "rank_tag": meta.get("rank_tag"),
        "school_tag": meta.get("school_tag"),
        "weapon_tag": meta.get("weapon_tag"),
        "technique_tag": meta.get("technique_tag"),
        "dense_score": None,
        "dense_rank": None,
        "lexical_score": None,
        "lexical_rank": None,
        "rrf_score": 0.0,
        "fused_rank": None,
        "heuristic_delta": 0.0,
        "heuristic_score": None,
        "heuristic_rank": None,
        "reranker_score": None,
        "final_score": None,
        "final_rank": None,
        "matched_stages": [],
        "score": 0.0,
        "rerank_score": None,
    }


def _clone_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(candidate)
    cloned["meta"] = dict(candidate.get("meta") or {})
    cloned["heading_path"] = list(candidate.get("heading_path") or [])
    cloned["matched_stages"] = list(candidate.get("matched_stages") or [])
    return cloned


def _lexical_document_text(chunk: dict[str, Any]) -> str:
    meta = chunk.get("meta") or {}
    parts = [
        " > ".join(meta.get("heading_path") or []),
        meta.get("section_title") or "",
        meta.get("content_type") or "",
        meta.get("rank_tag") or "",
        meta.get("school_tag") or "",
        meta.get("weapon_tag") or "",
        meta.get("technique_tag") or "",
        chunk.get("text") or "",
    ]
    return "\n".join(part for part in parts if part)


def _tokenize(text: str) -> list[str]:
    folded = _fold(text)
    return TOKEN_RE.findall(folded)


def _token_overlap_score(document_tokens: list[str], query_tokens: list[str]) -> float:
    if not document_tokens or not query_tokens:
        return 0.0

    doc_counts: dict[str, int] = {}
    for token in document_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    score = 0.0
    for token in query_tokens:
        if token in doc_counts:
            score += 1.0 + math.log1p(doc_counts[token])
    return score


def _heuristic_delta(question: str, candidate: dict[str, Any]) -> float:
    text = candidate.get("text", "") or ""
    meta = candidate.get("meta") or {}
    q_low = _fold(question)
    t_low = _fold(text)

    priority_boost = 0.0
    prio = int(meta.get("priority", 0) or 0)
    if prio:
        priority_boost = {1: 0.0, 2: 0.20, 3: 0.40}.get(prio, 0.0)
    else:
        fname_heur = os.path.basename(meta.get("source", "")).lower()
        if "nttv rank requirements" in fname_heur:
            priority_boost = 0.40
        elif "nttv training reference" in fname_heur or "technique descriptions" in fname_heur:
            priority_boost = 0.20

    keyword_boost = 0.0
    if "ryu" in t_low:
        keyword_boost += 0.10
    if "school" in t_low or "schools" in t_low:
        keyword_boost += 0.05
    if "bujinkan" in t_low:
        keyword_boost += 0.05

    qt_boost = 0.0
    if "kihon happo" in q_low and "kihon happo" in t_low:
        qt_boost += 0.60

    ask_sanshin = ("sanshin" in q_low) or ("san shin" in q_low)
    has_sanshin = ("sanshin" in t_low) or ("san shin" in t_low) or ("sanshin no kata" in t_low)
    if ask_sanshin and has_sanshin:
        qt_boost += 0.45

    if "kyusho" in q_low and "kyusho" in t_low:
        qt_boost += 0.25

    ask_boshi = ("boshi ken" in q_low) or ("shito ken" in q_low)
    has_boshi = ("boshi ken" in t_low) or ("shito ken" in t_low)
    if ask_boshi and has_boshi:
        qt_boost += 0.45

    weapon_terms = [
        "hanbo",
        "rokushakubo",
        "rokushaku",
        "katana",
        "tanto",
        "shoto",
        "kusari",
        "fundo",
        "kusari fundo",
        "kyoketsu",
        "shoge",
        "shuko",
        "jutte",
        "jitte",
        "tessen",
        "kunai",
        "shuriken",
        "senban",
        "shaken",
    ]
    ask_weapon = (
        any(term in q_low for term in weapon_terms)
        or ("weapon" in q_low)
        or ("weapons" in q_low)
        or ("what rank" in q_low)
        or ("introduced at" in q_low)
        or ("when do i learn" in q_low)
    )
    has_weaponish = any(term in t_low for term in weapon_terms) or ("[weapon]" in t_low)
    if ask_weapon and has_weaponish:
        qt_boost += 0.55

    fname = os.path.basename(meta.get("source") or "").lower()
    if ask_weapon and ("weapons reference" in fname or "glossary" in fname):
        qt_boost += 0.25

    school_alias_tokens: list[str] = []
    for canon, aliases in SCHOOL_ALIASES.items():
        school_alias_tokens.extend([_fold(canon)] + [_fold(alias) for alias in aliases])
    if any(token in q_low for token in school_alias_tokens) and any(token in t_low for token in school_alias_tokens):
        qt_boost += 0.45

    ask_soke = any(
        term in q_low for term in ["soke", "grandmaster", "headmaster", "current head", "current grandmaster"]
    )
    has_soke = ("[sokeship]" in t_low) or (" soke" in t_low)
    if ask_soke and (has_soke or "leadership" in fname):
        qt_boost += 0.60
        if "leadership" in fname:
            qt_boost += 0.20

    technique_terms = [
        "omote gyaku",
        "ura gyaku",
        "musha dori",
        "take ori",
        "hon gyaku jime",
        "oni kudaki",
        "ude garame",
        "ganseki otoshi",
        "juji gatame",
        "omoplata",
        "te hodoki",
        "tai hodoki",
    ]
    ask_tech = any(term in q_low for term in technique_terms) or ("what is" in q_low and len(q_low.split()) <= 6)
    has_tech = any(term in t_low for term in technique_terms) or ("technique descriptions" in fname)
    if ask_tech and has_tech:
        qt_boost += 0.55

    kata_boost = 0.0
    ask_kata = (" kata" in q_low) or ("no kata" in q_low) or (" kata?" in q_low)
    has_kata = (" kata" in t_low) or ("no kata" in t_low)
    if ask_kata and has_kata:
        kata_boost += 0.50

    offtopic_penalty = 0.0
    if "kihon happo" in q_low and "kyusho" in t_low:
        offtopic_penalty += 0.15
    if "kyusho" in q_low and "kihon happo" in t_low:
        offtopic_penalty += 0.15
    if ask_sanshin and "kyusho" in t_low:
        offtopic_penalty += 0.12

    lore_penalty = 0.0
    if any(term in t_low for term in ["sarutobi", "sasuke", "leaping from tree", "legend", "folklore"]):
        lore_penalty += 0.10

    length_penalty = min(len(text) / 2000.0, 0.3)

    rank_boost = 0.0
    for rank in [
        "10th kyu",
        "9th kyu",
        "8th kyu",
        "7th kyu",
        "6th kyu",
        "5th kyu",
        "4th kyu",
        "3rd kyu",
        "2nd kyu",
        "1st kyu",
    ]:
        if rank in q_low and rank in t_low:
            rank_boost += 0.50

    return (
        priority_boost
        + keyword_boost
        + qt_boost
        + rank_boost
        + kata_boost
        - length_penalty
        - offtopic_penalty
        - lore_penalty
    )


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold()
    normalized = normalized.replace("-", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _safe_score(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value.strip())


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return _parse_bool(value)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")
