# app.py (patched)
import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
import re
import unicodedata

import numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from nttv_chatbot.retrieval import (
    LexicalRetriever,
    RetrievalResult,
    RetrievalSettings,
    assemble_context as assemble_retrieval_context,
    search as search_retrieval_pipeline,
)


# Vector index
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# Embeddings
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

# Deterministic extractors (dispatcher + specific modules)
from extractors.kihon_happo import try_answer_kihon_happo
from extractors import try_extract_answer
from extractors.leadership import try_extract_answer as try_leadership
from extractors.weapons import try_answer_weapon_rank
from extractors.rank import try_answer_rank_requirements
from extractors.schools import (
    try_answer_school_profile,
    try_answer_schools_list,   # list extractor
    SCHOOL_ALIASES,
    is_school_list_query,
)

from extractors.technique_match import (
    is_single_technique_query as _is_single_technique_query,
    technique_name_variants as _tech_name_variants,
)

# --------------------------------------------------------------------
# Index / metadata lazy loader (Render-safe)
# --------------------------------------------------------------------
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")

# Globals that get populated once we actually load the index
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
CHUNKS: List[Dict[str, Any]] = []
INDEX = None

@st.cache_resource(show_spinner=False)
def _load_index_and_meta() -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Lazy, cached loader for FAISS index + meta.
    - Runs only when first needed.
    - Raises RuntimeError with a clear message if files are missing.
    """
    global EMBED_MODEL_NAME, TOP_K, CHUNKS, INDEX

    index_dir = os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)
    config_path = os.getenv("CONFIG_PATH", os.path.join(index_dir, "config.json"))
    meta_path = os.getenv("META_PATH", os.path.join(index_dir, "meta.pkl"))

    # ---- Basic existence checks for config/meta
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"Index config not found at {config_path}.\n"
            "Hints:\n"
            f"- INDEX_DIR is currently: {index_dir}\n"
            "- On Render, make sure ingest.py ran successfully in the build step.\n"
            "- Confirm config.json was written into that INDEX_DIR."
        )

    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Index metadata not found at {meta_path}.\n"
            "Hints:\n"
            f"- INDEX_DIR is currently: {index_dir}\n"
            "- On Render, confirm ingest.py wrote meta.pkl into the same directory as config.json.\n"
            "- If you changed INDEX_DIR in the environment, make sure ingest.py and app.py agree."
        )

    # ---- Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Support both 'embedding_model' (new) and 'embed_model' (old) keys
    EMBED_MODEL_NAME = (
        cfg.get("embedding_model")
        or cfg.get("embed_model")
        or EMBED_MODEL_NAME
    )

    # TOP_K is optional; default remains 6
    TOP_K_CFG = int(cfg.get("top_k", TOP_K))

    if faiss is None:
        raise RuntimeError(
            "faiss is not installed.\n"
            "Make sure `faiss-cpu==1.13.0` is present in requirements.txt and installed."
        )

    # ---- Resolve FAISS index path robustly
    faiss_candidates: list[str] = []

    # 1) Hard override via env
    idx_env = os.getenv("INDEX_PATH")
    if idx_env:
        faiss_candidates.append(idx_env)

    # 2) Config-specified path (absolute or relative to index_dir)
    cfg_faiss = cfg.get("faiss_path")
    if cfg_faiss:
        if os.path.isabs(cfg_faiss):
            faiss_candidates.append(cfg_faiss)
        else:
            faiss_candidates.append(os.path.join(index_dir, cfg_faiss))

    # 3) Backwards-compatible default names inside index_dir
    faiss_candidates.append(os.path.join(index_dir, "index.faiss"))  # what ingest.py writes
    faiss_candidates.append(os.path.join(index_dir, "faiss.index"))  # legacy name

    tried: list[str] = []
    def _try_load(fpath: str) -> Optional[Tuple[Any, List[Dict[str, Any]]]]:
        tried.append(fpath)
        if not (fpath and os.path.exists(fpath)):
            return None
        idx_local = faiss.read_index(fpath)
        with open(meta_path, "rb") as f:
            chunks_local: List[Dict[str, Any]] = pickle.load(f)
        # If obviously mismatched, signal caller to try next candidate
        ntotal_local = int(getattr(idx_local, "ntotal", 0) or 0)
        if ntotal_local <= 0 or len(chunks_local) <= 0:
            return None
        # If huge mismatch, don't accept this pair (likely stale env path)
        if abs(ntotal_local - len(chunks_local)) > 0:
            return None
        return (idx_local, chunks_local)

    idx = None
    chunks: List[Dict[str, Any]] = []

    # try candidates in order; skip pairs that mismatch counts
    for cand in faiss_candidates:
        pair = _try_load(cand)
        if pair is not None:
            idx, chunks = pair
            faiss_path = cand
            break

    if idx is None:
        tried_text = "\n".join(f"- {p}" for p in faiss_candidates if p)
        raise RuntimeError(
            "FAISS index file not found or unusable (mismatch with meta.pkl).\n"
            "Paths tried:\n"
            f"{tried_text}\n\n"
            "Hints:\n"
            "- Ensure ingest.py has recently rebuilt index.faiss + meta.pkl together.\n"
            "- If you set INDEX_PATH to a legacy name (faiss.index), remove it or point it to the new file."
        )

    # Update globals for any code that reads them
    INDEX = idx
    CHUNKS = chunks
    globals()["TOP_K"] = TOP_K_CFG

    return idx, chunks



# --------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------
_EMBED_MODEL = None

def get_embedder():
    global _EMBED_MODEL
    # Guarantee config/index has been loaded at least once so EMBED_MODEL_NAME is correct
    _load_index_and_meta()

    if _EMBED_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Add `sentence-transformers` to requirements.txt."
            )
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL

def embed_query(q: str) -> np.ndarray:
    model = get_embedder()
    v = model.encode([q], normalize_embeddings=True)
    return v.astype("float32")


# --------------------------------------------------------------------
# Retrieval & reranking
# --------------------------------------------------------------------
def retrieve(q: str, k: int | None = None) -> List[Dict[str, Any]]:
    """
    Search FAISS, then rerank with filename priority, query-aware boosts/penalties, and rank match.
    """
    idx, chunks = _load_index_and_meta()
    if k is None:
        k = TOP_K

    # --- Sanity checks to catch mismatched artifacts early
    ntotal = int(getattr(idx, "ntotal", 0) or 0)
    if ntotal <= 0:
        raise RuntimeError("FAISS index is empty (ntotal=0). Re-run ingest.py.")

    if len(chunks) == 0:
        raise RuntimeError("meta.pkl loaded but contains 0 chunks. Re-run ingest.py.")

    # If index/meta are out of sync, fail loudly (and print paths via sidebar)
    if abs(ntotal - len(chunks)) > 0:
        raise RuntimeError(
            f"Index/meta mismatch: faiss.ntotal={ntotal}, chunks={len(chunks)}.\n"
            "This usually means FAISS and meta.pkl are from different ingest runs."
        )

    v = embed_query(q)

    # Overfetch, but never ask for more than we actually have in the index
    want = min(max(k * 2, k), ntotal)
    D, I = idx.search(v, want)

    cand = []
    q_low = q.lower()

    for idx_i, score in zip(I[0], D[0]):
        # FAISS may return -1 for empty slots; also guard out-of-range indices
        if idx_i < 0 or idx_i >= len(chunks):
            continue

        c = chunks[idx_i]
        text = c.get("text", "")
        meta = c.get("meta", {}) or {}
        t_low = text.lower()

        # ---- Priority boost from ingest (preferred), else filename heuristic
        prio = int(meta.get("priority", 0))
        if prio:
            priority_boost = {1: 0.0, 2: 0.20, 3: 0.40}.get(prio, 0.0)
        else:
            fname_heur = os.path.basename(meta.get("source", "")).lower()
            if "nttv rank requirements" in fname_heur:
                priority_boost = 0.40
            elif "nttv training reference" in fname_heur or "technique descriptions" in fname_heur:
                priority_boost = 0.20
            else:
                priority_boost = 0.0

        # ---- Generic keyword nudges (small)
        keyword_boost = 0.0
        if "ryu" in t_low or "ryū" in t_low:
            keyword_boost += 0.10
        if "school" in t_low or "schools" in t_low:
            keyword_boost += 0.05
        if "bujinkan" in t_low:
            keyword_boost += 0.05

        # ---- Query-aware boosts/penalties (STRONG for core concepts)
        qt_boost = 0.0

        # Kihon Happo
        if "kihon happo" in q_low and "kihon happo" in t_low:
            qt_boost += 0.60

        # Sanshin
        ask_sanshin = ("sanshin" in q_low) or ("san shin" in q_low)
        has_sanshin = ("sanshin" in t_low) or ("san shin" in t_low) or ("sanshin no kata" in t_low)
        if ask_sanshin and has_sanshin:
            qt_boost += 0.45

        # Kyusho
        if "kyusho" in q_low and "kyusho" in t_low:
            qt_boost += 0.25

        # Boshi/Shito names
        ask_boshi = ("boshi ken" in q_low) or ("shito ken" in q_low)
        has_boshi = ("boshi ken" in t_low) or ("shito ken" in t_low)
        if ask_boshi and has_boshi:
            qt_boost += 0.45

        # Weapons cues
        weapon_terms = [
            "hanbo","hanbō","rokushakubo","rokushaku","katana","tanto","shoto","shōtō",
            "kusari","fundo","kusari fundo","kyoketsu","shoge","shōge","shuko","shukō",
            "jutte","jitte","tessen","kunai","shuriken","senban","shaken"
        ]
        ask_weapon = (
            any(w in q_low for w in weapon_terms)
            or ("weapon" in q_low) or ("weapons" in q_low)
            or ("what rank" in q_low) or ("introduced at" in q_low)
            or ("when do i learn" in q_low)
        )
        has_weaponish = any(w in t_low for w in weapon_terms) or ("[weapon]" in t_low) or ("weapons reference" in t_low)
        if ask_weapon and has_weaponish:
            qt_boost += 0.55

        # Filename heuristic: prefer Weapons Reference / Glossary for weapons Qs
        fname = os.path.basename(meta.get("source") or "").lower()
        if ask_weapon and ("weapons reference" in fname or "glossary" in fname):
            qt_boost += 0.25

        # Schools / ryū boost
        school_aliases = []
        for canon, aliases in SCHOOL_ALIASES.items():
            school_aliases.extend([canon.lower()] + [a.lower() for a in aliases])
        if any(a in q_low for a in school_aliases) and any(a in t_low for a in school_aliases):
            qt_boost += 0.45

        # Leadership boost
        ask_soke = any(t in q_low for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"])
        has_soke = ("[sokeship]" in t_low) or (" soke" in t_low) or (" sōke" in t_low)
        if ask_soke and (has_soke or "leadership" in fname):
            qt_boost += 0.60
            if "leadership" in fname:
                qt_boost += 0.20

        # Technique name nudge (from Technique Descriptions)
        technique_terms = [
            "omote gyaku","ura gyaku","musha dori","take ori","hon gyaku jime","oni kudaki",
            "ude garame","ganseki otoshi","juji gatame","omoplata","te hodoki","tai hodoki",
        ]
        ask_tech = any(t in q_low for t in technique_terms) or ("what is" in q_low and len(q_low.split()) <= 6)
        has_tech = any(t in t_low for t in technique_terms) or ("technique descriptions" in fname)
        if ask_tech and has_tech:
            qt_boost += 0.55

        # Kata boost
        kata_boost = 0.0
        ask_kata = (" kata" in q_low) or ("no kata" in q_low) or (" kata?" in q_low)
        has_kata = (" kata" in t_low) or ("no kata" in t_low)
        if ask_kata and has_kata:
            kata_boost += 0.50

        # Offtopic penalties / lore / length
        offtopic_penalty = 0.0
        if "kihon happo" in q_low and "kyusho" in t_low: offtopic_penalty += 0.15
        if "kyusho" in q_low and "kihon happo" in t_low: offtopic_penalty += 0.15
        if ask_sanshin and "kyusho" in t_low: offtopic_penalty += 0.12

        lore_penalty = 0.0
        if any(k in t_low for k in ["sarutobi", "sasuke", "leaping from tree", "legend", "folklore"]):
            lore_penalty += 0.10

        length_penalty = min(len(text) / 2000.0, 0.3)

        # Exact rank match
        rank_boost = 0.0
        for rank in ["10th kyu","9th kyu","8th kyu","7th kyu","6th kyu","5th kyu","4th kyu","3rd kyu","2nd kyu","1st kyu"]:
            if rank in q_low and rank in t_low:
                rank_boost += 0.50

        new_score = (
            float(score)
            + priority_boost
            + keyword_boost
            + qt_boost
            + rank_boost
            + kata_boost
            - length_penalty
            - offtopic_penalty
            - lore_penalty
        )

        cand.append(
            (
                new_score,
                {
                    "text": text,
                    "meta": meta,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "page_start": meta.get("page_start", meta.get("page")),
                    "page_end": meta.get("page_end", meta.get("page")),
                    "heading_path": meta.get("heading_path") or [],
                    "priority_bucket": meta.get("priority_bucket"),
                    "chunk_id": meta.get("chunk_id"),
                    "rank_tag": meta.get("rank_tag"),
                    "school_tag": meta.get("school_tag"),
                    "weapon_tag": meta.get("weapon_tag"),
                    "technique_tag": meta.get("technique_tag"),
                    "score": float(score),
                    "rerank_score": float(new_score),
                },
            )
        )

    cand.sort(key=lambda x: x[0], reverse=True)
    out = [c for _, c in cand[:k]]

    if not out:
        raise RuntimeError(
            "Retrieval returned 0 usable passages. This can happen if FAISS returned only -1 indices "
            "or if index/meta are mismatched.\n"
            "Try: delete index/ and re-run `python ingest.py`."
        )

    return out


def build_context(snippets: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Concatenate top-k snippets into a context block with a cap."""
    lines, total = [], 0
    for i, s in enumerate(snippets, 1):
        tag = f"[{i}] {os.path.basename(s['source'])}"
        page_start = s.get("page_start") or s.get("page")
        page_end = s.get("page_end") or page_start
        if page_start and page_end and page_start != page_end:
            tag += f" (pp. {page_start}-{page_end})"
        elif page_start:
            tag += f" (p. {page_start})"
        block = f"{tag}\n{s['text']}\n\n---\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "".join(lines)

def retrieval_quality(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return max(h.get("rerank_score", h.get("score", 0.0)) for h in hits)


# Hybrid retrieval overrides the legacy local-only retrieval helpers above.
_LEXICAL_RETRIEVER: Optional[LexicalRetriever] = None
_LAST_RETRIEVAL_RESULT: Optional[RetrievalResult] = None


def get_lexical_retriever() -> LexicalRetriever:
    global _LEXICAL_RETRIEVER
    if _LEXICAL_RETRIEVER is None:
        _, chunks = _load_index_and_meta()
        _LEXICAL_RETRIEVER = LexicalRetriever(chunks)
    return _LEXICAL_RETRIEVER


def _legacy_retrieve(q: str, k: int | None = None) -> List[Dict[str, Any]]:
    global _LAST_RETRIEVAL_RESULT

    idx, chunks = _load_index_and_meta()
    settings = RetrievalSettings.from_env()
    final_top_k = k if k is not None else TOP_K

    _LAST_RETRIEVAL_RESULT = search_retrieval_pipeline(
        q,
        index=idx,
        chunks=chunks,
        embed_query=embed_query,
        final_top_k=final_top_k,
        settings=settings,
        lexical_retriever=get_lexical_retriever(),
    )
    return _LAST_RETRIEVAL_RESULT.final_candidates


def get_last_retrieval_debug() -> Dict[str, Any]:
    if _LAST_RETRIEVAL_RESULT is None:
        return _empty_retrieval_debug("Retrieval was not run for this answer.")
    return _LAST_RETRIEVAL_RESULT.to_debug_payload()


def _legacy_build_context(snippets: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    return assemble_retrieval_context(snippets, max_chars=max_chars)


def _legacy_retrieval_quality(hits: List[Dict[str, Any]]) -> float:
    if not hits:
        return 0.0
    return max(
        h.get("final_score")
        or h.get("rerank_score")
        or h.get("heuristic_score")
        or h.get("score")
        or 0.0
        for h in hits
    )


retrieve = _legacy_retrieve
build_context = _legacy_build_context
retrieval_quality = _legacy_retrieval_quality


def _empty_retrieval_debug(reason: str) -> Dict[str, Any]:
    return {
        "dense_candidates": [],
        "lexical_candidates": [],
        "fused_candidates": [],
        "reranked_candidates": [],
        "final_candidates": [],
        "reranker_backend_requested": "none",
        "reranker_backend_used": "none",
        "reranker_fallback_reason": reason,
        "deterministic_short_circuit": True,
    }


def _format_page_range(meta: Dict[str, Any]) -> str:
    page_start = meta.get("page_start") or meta.get("page")
    page_end = meta.get("page_end") or page_start
    if page_start and page_end and page_start != page_end:
        return f"{page_start}-{page_end}"
    if page_start:
        return str(page_start)
    return ""


def _format_detected_tags(meta: Dict[str, Any]) -> str:
    labels = [
        ("rank", meta.get("rank_tag")),
        ("school", meta.get("school_tag")),
        ("weapon", meta.get("weapon_tag")),
        ("technique", meta.get("technique_tag")),
    ]
    parts = [f"{name}: {value}" for name, value in labels if value]
    return ", ".join(parts)


def _format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _render_stage_candidates(title: str, candidates: List[Dict[str, Any]]) -> None:
    with st.expander(title, expanded=False):
        if not candidates:
            st.caption("No candidates for this stage.")
            return

        for i, candidate in enumerate(candidates, 1):
            meta = candidate.get("meta") or {}
            source_file = meta.get("source_file") or candidate.get("source") or ""
            name = os.path.basename(source_file)
            page_range = _format_page_range(meta)
            heading_path = " > ".join(meta.get("heading_path") or [])
            tags = _format_detected_tags(meta)
            priority_bucket = meta.get("priority_bucket") or f"p{max(1, 4 - int(meta.get('priority', 1) or 1))}"
            score_line = (
                f"dense={_format_score(candidate.get('dense_score'))}, "
                f"lexical={_format_score(candidate.get('lexical_score'))}, "
                f"rrf={_format_score(candidate.get('rrf_score'))}, "
                f"heuristic={_format_score(candidate.get('heuristic_score'))}, "
                f"reranker={_format_score(candidate.get('reranker_score'))}, "
                f"final={_format_score(candidate.get('final_score'))}"
            )

            st.write(f"[{i}] {name}")
            st.caption(score_line)
            if page_range:
                st.caption(f"Pages: {page_range}")
            if heading_path:
                st.caption(f"Heading: {heading_path}")
            st.caption(f"Priority bucket: {priority_bucket}")
            if tags:
                st.caption(f"Tags: {tags}")


# --------------------------------------------------------------------
# Injectors & helpers
# --------------------------------------------------------------------
def _gather_full_text_for_source(name_contains: str) -> Tuple[str, Optional[str]]:
    _, chunks = _load_index_and_meta()
    name_low = (name_contains or "").lower()
    parts, path = [], None
    for c in chunks:
        src = (c["meta"].get("source") or "")
        if name_low in src.lower():
            parts.append(c["text"])
            path = src
    return ("\n\n".join(parts), path)


def inject_rank_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["kyu", "shodan", "rank requirement", "rank requirements"]):
        return hits
    txt, path = _gather_full_text_for_source("nttv rank requirements")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "nttv rank requirements.txt (synthetic)"},
        "source": path or "nttv rank requirements.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 997.0,
    }
    return [synth] + hits


def inject_leadership_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["soke","sōke","grandmaster","headmaster","current head","current grandmaster"]):
        return hits
    txt, path = _gather_full_text_for_source("bujinkan leadership and wisdom")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Bujinkan Leadership and Wisdom.txt (synthetic)"},
        "source": path or "Bujinkan Leadership and Wisdom.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 998.0,
    }
    return [synth] + hits


def inject_schools_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ql = question.lower()
    if not any(t in ql for t in ["school", "schools", "ryu", "ryū", "bujinkan"]):
        return hits
    txt, path = _gather_full_text_for_source("schools of the bujinkan summaries")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Schools of the Bujinkan Summaries.txt (synthetic)"},
        "source": path or "Schools of the Bujinkan Summaries.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 995.0,
    }
    return [synth] + hits


def inject_weapons_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepend the full NTTV Weapons Reference when the question mentions a weapon or 'rank/learn' for weapons."""
    ql = question.lower()
    weapon_triggers = [
        "hanbo","hanbō","rokushakubo","rokushaku","katana","tanto","shoto","shōtō",
        "kusari","fundo","kusari fundo","kyoketsu","shoge","shōge","shuko","shukō",
        "jutte","jitte","tessen","kunai","shuriken","senban","shaken","throwing star","throwing spike",
        "weapon","weapons","what rank","when do i learn","introduced at"
    ]
    if not any(t in ql for t in weapon_triggers):
        return hits
    txt, path = _gather_full_text_for_source("weapons reference")
    if not txt:
        return hits
    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "NTTV Weapons Reference.txt (synthetic)"},
        "source": path or "NTTV Weapons Reference.txt (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 996.0,
    }
    return [synth] + hits


def inject_techniques_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepend the full Technique Descriptions when a technique-style question is asked (but NOT for concepts)."""
    ql = question.lower()

    # 🚫 Do NOT inject techniques for concept queries (these have their own extractors)
    if any(b in ql for b in ["kihon happo", "sanshin", "school", "schools", "ryu", "ryū"]):
        return hits

    triggers = [
        "what is", "define", "explain",
        "gyaku", "kudaki", "dori", "gatame", "ganseki", "nage", "otoshi",
        "wrist lock", "shoulder lock", "armbar",
        "te hodoki", "tai hodoki",
        " no kata",
    ]
    if not any(t in ql for t in triggers):
        return hits

    txt, path = _gather_full_text_for_source("technique descriptions")
    if not txt:
        return hits

    synth = {
        "text": txt,
        "meta": {"priority": 1, "source": path or "Technique Descriptions.md (synthetic)"},
        "source": path or "Technique Descriptions.md (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 994.0,
    }
    return [synth] + hits


def inject_kihon_passage_if_needed(question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If the question is about Kihon Happo, synthesize a concise passage with the two subsets + items."""
    _, chunks = _load_index_and_meta()
    ql = question.lower()
    if "kihon happo" not in ql and "kihon happō" not in ql:
        return hits

    kosshi_lines, torite_lines, defs = [], [], []

    def push_lines_from(text: str):
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            low = ln.lower()
            if "kihon happo" in low and 10 < len(ln) < 220:
                defs.append(ln.rstrip(" ;,"))
            if "kosshi" in low and "sanpo" in low:
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
                parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail) if 2 <= len(p.strip()) <= 60]
                kosshi_lines.extend(parts)
            if "torite" in low and ("goho" in low or "gohō" in low):
                tail = ln.split(":", 1)[1].strip() if ":" in ln else ln
                parts = [p.strip(" -•\t") for p in re.split(r"[;,]", tail) if 2 <= len(p.strip()) <= 60]
                torite_lines.extend(parts)

    # scan top-N retrieved first, then a light scan across chunks if needed
    for p in hits[:8]:
        push_lines_from(p.get("text", ""))

    if (len(kosshi_lines) < 3 or len(torite_lines) < 5):
        for c in chunks[:1000]:  # bounded scan
            src = (c["meta"].get("source") or "").lower()
            if not any(tag in src for tag in ["training reference", "rank requirements", "schools", "glossary", "technique descriptions"]):
                continue
            push_lines_from(c["text"])
            if len(kosshi_lines) >= 3 and len(torite_lines) >= 5 and defs:
                break

    def dedupe(seq):
        seen = set(); out = []
        for x in seq:
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out

    kosshi = dedupe(kosshi_lines)[:3]
    torite = dedupe(torite_lines)[:5]

    if not (kosshi or torite or defs):
        return hits

    parts = ["Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."]
    if kosshi:
        parts.append("Kosshi Kihon Sanpo: " + ", ".join(kosshi) + ".")
    if torite:
        parts.append("Torite Goho: " + ", ".join(torite) + ".")
    if defs:
        parts.append(defs[0] if parts[-1].endswith(".") else (". " + defs[0]))

    body = " ".join(parts).strip()

    synth = {
        "text": body,
        "meta": {"priority": 1, "source": "Kihon Happo (synthetic)"},
        "source": "Kihon Happo (synthetic)",
        "page": None,
        "score": 1.0,
        "rerank_score": 998.0,
    }
    return [synth] + hits


# --------------------------------------------------------------------
# Single-technique CSV fast-path (parsing & render)
# --------------------------------------------------------------------
def _parse_tech_csv_line(line: str) -> Optional[Dict[str, str]]:
    """
    Parse a single technique CSV row from Technique Descriptions.md.

    Expected logical columns (min 12):
      0 name
      1 japanese
      2 english
      3 family (e.g., 'Kihon Happo - Kosshi')
      4 rank_intro (e.g., '7th Kyu')
      5 approved (✅/False/True)
      6 focus
      7 safety
      8 partner_required (True/False)
      9 solo (True/False)
      10 tags (pipe-separated)
      11+ definition (may include commas)
    """
    if not line or "," not in line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 12:
        return None
    head = parts[:11]
    definition = ",".join(parts[11:]).strip()

    name = head[0]
    japanese = head[1]
    english = head[2]
    family = head[3]
    rank_intro = head[4]
    approved = head[5]
    focus = head[6]
    safety = head[7]
    partner_required = head[8]
    solo = head[9]
    tags = head[10]

    return {
        "name": name,
        "japanese": japanese,
        "english": english,
        "family": family,
        "rank_intro": rank_intro,
        "approved": approved,
        "focus": focus,
        "safety": safety,
        "partner_required": partner_required,
        "solo": solo,
        "tags": tags,
        "definition": definition,
    }


def _render_single_technique(row: Dict[str, str], *, bullets: bool, tone: str, detail_mode: str) -> str:
    """
    Format a single technique into bullets or paragraph. 'detail_mode' in {"Brief","Standard","Full"}.
    """
    title  = row.get("name","Technique")
    jp     = row.get("japanese","")
    en     = row.get("english","")
    family = row.get("family","")
    rank   = row.get("rank_intro","")
    focus  = row.get("focus","")
    safety = row.get("safety","")
    part   = row.get("partner_required","")
    solo   = row.get("solo","")
    tags   = row.get("tags","")
    defin  = (row.get("definition") or "").strip()

    if detail_mode == "Brief":
        brief = [f"{title}:"]
        if en:  brief.append(f"- English: {en}")
        if jp:  brief.append(f"- Japanese: {jp}")
        if defin: brief.append(f"- Definition: {defin if defin.endswith('.') else defin + '.'}")
        body = "\n".join(brief)

    elif detail_mode == "Standard":
        std = [f"{title}:"]
        if en:      std.append(f"- English: {en}")
        if jp:      std.append(f"- Japanese: {jp}")
        if family:  std.append(f"- Family: {family}")
        if rank:    std.append(f"- Rank intro: {rank}")
        if focus:   std.append(f"- Focus: {focus}")
        if defin:   std.append(f"- Definition: {defin if defin.endswith('.') else defin + '.'}")
        body = "\n".join(std)

    else:  # Full
        full = [f"{title}:"]
        if jp:      full.append(f"- Japanese: {jp}")
        if en:      full.append(f"- English: {en}")
        if family:  full.append(f"- Family: {family}")
        if rank:    full.append(f"- Rank intro: {rank}")
        if focus:   full.append(f"- Focus: {focus}")
        if safety:  full.append(f"- Safety: {safety}")
        if part:    full.append(f"- Partner required: {part}")
        if solo:    full.append(f"- Solo: {solo}")
        if tags:    full.append(f"- Tags: {tags}")
        if defin:   full.append(f"- Definition: {defin if defin.endswith('.') else defin + '.'}")
        body = "\n".join(full)

    return _render_det(body, bullets=bullets, tone=tone)


def answer_single_technique_if_synthetic(
    passages: List[Dict[str, Any]],
    *,
    bullets: bool,
    tone: str,
    detail_mode: str
) -> Optional[str]:
    """
    If the first passage is our synthetic single-technique CSV line, parse & render it now.
    """
    if not passages:
        return None
    top = passages[0]
    src = (top.get("source") or "").lower()
    if "technique descriptions (synthetic line)" not in src:
        return None
    line = (top.get("text") or "").strip()
    row = _parse_tech_csv_line(line)
    if not row:
        return None
    return _render_single_technique(row, bullets=bullets, tone=tone, detail_mode=detail_mode)


# --- Technique CSV line injector (for single-technique queries) ----------------
import re as _re2  # avoid shadowing

def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


def _is_single_technique_query(q: str) -> Optional[str]:
    """
    Return a candidate technique name if the query looks like a single technique ask,
    else None. Handles 'explain/define/what is ... (no kata)?'
    """
    ql = (q or "").strip().lower()
    for ban in ("kihon happo", "kihon happō", "sanshin", "school", "schools", "ryu", "ryū"):
        if ban in ql:
            return None
    m = _re2.search(r"(?:what\s+is|define|explain)\s+(.+)$", q, flags=_re2.I)
    cand = (m.group(1) if m else q).strip().rstrip("?!.")
    cand = _re2.sub(r"\b(technique|in ninjutsu|in bujinkan)\b", "", cand, flags=_re2.I).strip()
    return cand if 2 <= len(cand) <= 80 else None


def _tech_name_variants(name: str) -> list[str]:
    v = [name.strip()]
    ln = name.strip().lower()
    if ln.endswith(" no kata"):
        v.append(name[:-8].strip())
    else:
        v.append(f"{name} no Kata")
    nh = name.replace("-", " ")
    if nh != name:
        v.append(nh)
        if nh.lower().endswith(" no kata"):
            v.append(nh[:-8].strip())
        else:
            v.append(f"{nh} no Kata")
    v.append(_fold(name))
    seen = set(); out = []
    for x in v:
        if x and x not in seen:
            out.append(x); seen.add(x)
    return out


def _find_tech_line_in_chunks(name_variants: list[str]) -> Optional[str]:
    """
    Scan all CHUNKS for lines from Technique Descriptions.md whose first CSV cell
    (technique name) matches any variant (macron-insensitive).
    Return the full CSV line if found.
    """
    _, chunks = _load_index_and_meta()
    folded_targets = {_fold(v) for v in name_variants}
    for c in chunks:
        src = (c["meta"].get("source") or "").lower()
        if "technique descriptions.md" not in src:
            continue
        for raw in c["text"].splitlines():
            line = raw.strip()
            if not line or "," not in line:
                continue
            first = line.split(",", 1)[0].strip()
            if _fold(first) in folded_targets:
                return line
    return None


def inject_specific_technique_line_if_needed(question: str, passages: list[dict]) -> list[dict]:
    cand = _is_single_technique_query(question)
    if not cand:
        return passages

    variants = _tech_name_variants(cand)
    line = _find_tech_line_in_chunks(variants)
    if not line:
        return passages

    synth = {
        "text": line,
        "meta": {"source": "Technique Descriptions (synthetic line)", "priority": 1},
        "source": "Technique Descriptions (synthetic line)",
        "page": None,
        "score": 1.0,
        "rerank_score": 1.0,
    }
    if not passages or passages[0].get("text") != line:
        return [synth] + passages
    return passages


# --------------------------------------------------------------------
# LLM backend (fallback)
# --------------------------------------------------------------------
def call_llm(
    prompt: str,
    system: str = "You are a precise assistant. Use only the provided context."
) -> Tuple[str, str]:
    import requests
    model = os.environ.get("MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_API_BASE")
        or os.environ.get("LM_STUDIO_BASE_URL")
        or "http://localhost:1234/v1"
    )

    headers = {"Content-Type": "application/json"}
    if "openai" in base or "openrouter" in base:
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    try:
        r = requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return text, json.dumps(data)[:4000]
    except Exception as e:
        return "", json.dumps({"error": type(e).__name__, "detail": str(e)})[:4000]


# --------------------------------------------------------------------
# Prompt & deterministic rendering helpers
# --------------------------------------------------------------------
def build_prompt(context: str, question: str) -> str:
    return (
        "You must answer using ONLY the context below.\n"
        "Be concise but complete; avoid filler.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _apply_tone(text: str, tone: str) -> str:
    """Light-touch tone adjustments; Chatty adds a friendly closer for non-bullets."""
    if tone == "Chatty":
        if "\n" not in text.strip():
            return text.strip() + " Want a quick drill or a bit of history too?"
        return text
    return text


def _bullets_to_paragraph(text: str) -> str:
    """Convert our fielded bullet output into a compact paragraph."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return text
    head = lines[0].rstrip(":")
    fields = {}
    for ln in lines[1:]:
        if ln.startswith("- "):
            ln = ln[2:]
        if ":" in ln:
            k, v = ln.split(":", 1)
            fields[k.strip().lower()] = v.strip().rstrip(".")
    segs = [head + ":"]
    if "translation" in fields:
        segs.append(f'“{fields["translation"]}”.')
    if "type" in fields:
        segs.append(f'Type: {fields["type"]}.')
    if "focus" in fields:
        segs.append(f'Focus: {fields["focus"]}.')
    if "weapons" in fields:
        segs.append(f'Weapons: {fields["weapons"]}.')
    if "notes" in fields:
        segs.append(f'Notes: {fields["notes"]}.')
    return " ".join(segs).strip()


def _render_det(text: str, *, bullets: bool, tone: str) -> str:
    """
    Deterministic renderer:
    - Bullets + Chatty: prepend a synthesized 'Quick take' line from the fields.
    - Bullets + Crisp: return bullets as-is.
    - Paragraph modes: convert bullets to paragraph then tone-adjust.
    """
    if bullets:
        if tone == "Chatty":
            lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
            title = lines[0].rstrip(":") if lines else ""
            fields = {}
            for ln in lines[1:]:
                if ln.startswith("- "):
                    ln = ln[2:]
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    fields[k.strip().lower()] = v.strip()
            trans = fields.get("translation")
            typ = fields.get("type")
            focus = fields.get("focus")
            quick_bits = []
            if trans: quick_bits.append(trans)
            if typ: quick_bits.append(typ)
            quick = " — ".join(quick_bits) if quick_bits else None
            if quick and focus:
                summary = f"Quick take: {quick}; focus on {focus}."
            elif quick:
                summary = f"Quick take: {quick}."
            elif focus:
                summary = f"Quick take: focus on {focus}."
            else:
                summary = None

            out = []
            if summary:
                out.append(summary)
            out.append(text.strip())
            out = "\n".join(out)
            if not out.strip().endswith("?"):
                out += "\n\nWant examples, drills, or lineage notes next?"
            return out
        else:
            return text.strip()

    para = _bullets_to_paragraph(text)
    return _apply_tone(para, tone)


# --- School intent detection ---
def is_school_query(question: str) -> bool:
    ql = question.lower()
    for canon, aliases in SCHOOL_ALIASES.items():
        tokens = [canon.lower()] + [a.lower() for a in aliases]
        if any(tok in ql for tok in tokens):
            return True
    return (" ryu" in ql) or (" ryū" in ql)


def is_soke_query(q: str) -> bool:
    ql = q.lower()
    return any(token in ql for token in [
        "soke", "sōke", "current soke", "who is the soke", "who is the sōke",
        "grandmaster", "current grandmaster", "who is the grandmaster"
    ])


# --------------------------------------------------------------------
# Core RAG pipeline
# --------------------------------------------------------------------
def _prepare_deterministic_passages(question: str) -> List[Dict[str, Any]]:
    _, chunks = _load_index_and_meta()
    passages = list(chunks)
    passages = inject_rank_passage_if_needed(question, passages)
    passages = inject_leadership_passage_if_needed(question, passages)
    passages = inject_schools_passage_if_needed(question, passages)
    passages = inject_weapons_passage_if_needed(question, passages)
    passages = inject_kihon_passage_if_needed(question, passages)
    passages = inject_techniques_passage_if_needed(question, passages)
    passages = inject_specific_technique_line_if_needed(question, passages)
    return passages


def _answer_from_passages(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[Tuple[str, List[Dict[str, Any]], str]]:
    fast = answer_single_technique_if_synthetic(
        passages,
        bullets=(output_style == "Bullets"),
        tone=tone_style,
        detail_mode=TECH_DETAIL_MODE,
    )
    if fast:
        return f"🔒 Strict (context-only, explain)\n\n{fast}", passages, '{"det_path":"technique/single"}'

    if is_soke_query(question):
        ans = try_leadership(question, passages)
        if ans:
            return ans, passages, '{"det_path":"leadership/soke"}'

    if is_school_list_query(question):
        try:
            list_ans = try_answer_schools_list(
                question, passages, bullets=(output_style == "Bullets")
            )
        except Exception:
            list_ans = None
        if list_ans:
            rendered = _render_det(list_ans, bullets=(output_style == "Bullets"), tone=tone_style)
            return (
                f"🔒 Strict (context-only, explain)\n\n{rendered}",
                passages,
                '{"det_path":"schools/list"}',
            )

    if is_school_query(question):
        try:
            school_fact = try_answer_school_profile(
                question, passages, bullets=(output_style == "Bullets")
            )
        except Exception:
            school_fact = None
        if school_fact:
            rendered = _render_det(school_fact, bullets=(output_style == "Bullets"), tone=tone_style)
            return (
                f"🔒 Strict (context-only, explain)\n\n{rendered}",
                passages,
                '{"det_path":"schools/profile"}',
            )

    asking_soke = any(t in question.lower() for t in [
        "soke", "sÅke", "grandmaster", "headmaster", "current head", "current grandmaster"
    ])
    if asking_soke:
        try:
            fact = try_leadership(question, passages)
        except Exception:
            fact = None
        if fact:
            return f"🔒 Strict (context-only, explain)\n\n{fact}", passages, '{"det_path":"leadership/soke"}'

    try:
        wr = try_answer_weapon_rank(question, passages)
    except Exception:
        wr = None
    if wr:
        return f"🔒 Strict (context-only)\n\n{wr}", passages, '{"det_path":"weapons/rank"}'

    try:
        rr = try_answer_rank_requirements(question, passages)
    except Exception:
        rr = None
    if rr:
        rendered = _render_det(rr, bullets=(output_style == "Bullets"), tone=tone_style)
        return f"🔒 Strict (context-only, explain)\n\n{rendered}", passages, '{"det_path":"rank/requirements"}'

    q_low = (question or "").lower()
    if "kihon happo" in q_low or "kihon happÅ" in q_low:
        kihon_ans = try_answer_kihon_happo(question, passages)
        if kihon_ans:
            rendered = _render_det(kihon_ans, bullets=(output_style == "Bullets"), tone=tone_style)
            return f"🔒 Strict (context-only, explain)\n\n{rendered}", passages, '{"det_path":"deterministic/kihon"}'

    fact = try_extract_answer(question, passages)
    if fact:
        rendered = _render_det(fact, bullets=(output_style == "Bullets"), tone=tone_style)
        ql = question.lower()
        looks_like_kata = (" kata" in ql) or ("no kata" in ql) or re.search(r"\bexplain\s+.+\s+no\s+kata\b", ql)
        det_tag = '{"det_path":"technique/core"}' if looks_like_kata else '{"det_path":"deterministic/core"}'
        return f"🔒 Strict (context-only, explain)\n\n{rendered}", passages, det_tag

    return None


def _legacy_answer_with_rag(question: str, k: int | None = None) -> Tuple[str, List[Dict[str, Any]], str]:
    if k is None:
        k = TOP_K

    # 1) Retrieve
    hits = retrieve(question, k=k)

    # 2) Inject domain-critical sources
    hits = inject_rank_passage_if_needed(question, hits)
    hits = inject_leadership_passage_if_needed(question, hits)
    hits = inject_schools_passage_if_needed(question, hits)
    hits = inject_weapons_passage_if_needed(question, hits)
    hits = inject_kihon_passage_if_needed(question, hits)
    hits = inject_techniques_passage_if_needed(question, hits)
    hits = inject_specific_technique_line_if_needed(question, hits)

    # Fast-path: if we injected a single technique CSV line, answer immediately
    fast = answer_single_technique_if_synthetic(
        hits,
        bullets=(output_style == "Bullets"),
        tone=tone_style,
        detail_mode=TECH_DETAIL_MODE,
    )
    if fast:
        return f"🔒 Strict (context-only, explain)\n\n{fast}", hits, '{"det_path":"technique/single"}'

    # Leadership (Sōke) gets priority over school profile if asked directly
    if is_soke_query(question):
        ans = try_leadership(question, hits)
        if ans:
            return ans, hits, '{"det_path":"leadership/soke"}'

    # Schools LIST short-circuit
    if is_school_list_query(question):
        try:
            list_ans = try_answer_schools_list(
                question, hits, bullets=(output_style == "Bullets")
            )
        except Exception:
            list_ans = None
        if list_ans:
            rendered = _render_det(list_ans, bullets=(output_style == "Bullets"), tone=tone_style)
            return (
                f"🔒 Strict (context-only, explain)\n\n{rendered}",
                hits,
                '{"det_path":"schools/list"}'
            )

    # School PROFILE short-circuit
    if is_school_query(question):
        try:
            school_fact = try_answer_school_profile(
                question, hits, bullets=(output_style == "Bullets")
            )
        except Exception:
            school_fact = None
        if school_fact:
            rendered = _render_det(school_fact, bullets=(output_style == "Bullets"), tone=tone_style)
            return (
                f"🔒 Strict (context-only, explain)\n\n{rendered}",
                hits,
                '{"det_path":"schools/profile"}'
            )

        # fallback LLM for schools
        ctx = build_context(hits)
        prompt = build_prompt(ctx, question)
        text, raw = call_llm(prompt)
        if not text.strip():
            return "🔒 Strict (context-only)\n\n❌ Model returned no text.", hits, raw or "{}"
        return f"🔒 Strict (context-only, explain)\n\n{text.strip()}", hits, raw or "{}"

    # Leadership short-circuit (generic)
    asking_soke = any(t in question.lower() for t in [
        "soke","sōke","grandmaster","headmaster","current head","current grandmaster"
    ])
    if asking_soke:
        try:
            fact = try_leadership(question, hits)
        except Exception:
            fact = None
        if fact:
            return f"🔒 Strict (context-only, explain)\n\n{fact}", hits, '{"det_path":"leadership/soke"}'

    # Weapon rank short-circuit
    try:
        wr = try_answer_weapon_rank(question, hits)
    except Exception:
        wr = None
    if wr:
        return f"🔒 Strict (context-only)\n\n{wr}", hits, '{"det_path":"weapons/rank"}'

    # Rank requirements short-circuit
    try:
        rr = try_answer_rank_requirements(question, hits)
    except Exception:
        rr = None
    if rr:
        rendered = _render_det(rr, bullets=(output_style == "Bullets"), tone=tone_style)
        return f"🔒 Strict (context-only, explain)\n\n{rendered}", hits, '{"det_path":"rank/requirements"}'

    # Kihon Happo hard short-circuit
    q_low = (question or "").lower()
    if "kihon happo" in q_low or "kihon happō" in q_low:
        kihon_ans = try_answer_kihon_happo(question, hits)
        if kihon_ans:
            rendered = _render_det(kihon_ans, bullets=(output_style == "Bullets"), tone=tone_style)
            return f"🔒 Strict (context-only, explain)\n\n{rendered}", hits, '{"det_path":"deterministic/kihon"}'

    # Generic deterministic dispatcher
    fact = try_extract_answer(question, hits)
    if fact:
        rendered = _render_det(fact, bullets=(output_style == "Bullets"), tone=tone_style)
        ql = question.lower()
        looks_like_kata = (" kata" in ql) or ("no kata" in ql) or re.search(r"\bexplain\s+.+\s+no\s+kata\b", ql)
        det_tag = '{"det_path":"technique/core"}' if looks_like_kata else '{"det_path":"deterministic/core"}'
        return f"🔒 Strict (context-only, explain)\n\n{rendered}", hits, det_tag

    # LLM fallback with retrieved context
    ctx = build_context(hits)
    prompt = build_prompt(ctx, question)
    text, raw = call_llm(prompt)
    if not text.strip():
        return "🔒 Strict (context-only)\n\n❌ Model returned no text.", hits, raw or "{}"
    return f"🔒 Strict (context-only, explain)\n\n{text.strip()}", hits, raw or "{}"


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
def answer_with_rag(
    question: str,
    k: int | None = None,
) -> Tuple[str, List[Dict[str, Any]], str, Dict[str, Any]]:
    global _LAST_RETRIEVAL_RESULT

    if k is None:
        k = TOP_K

    prepass_passages = _prepare_deterministic_passages(question)
    prepass_answer = _answer_from_passages(question, prepass_passages)
    if prepass_answer:
        _LAST_RETRIEVAL_RESULT = None
        answer, hits, raw = prepass_answer
        debug_payload = _empty_retrieval_debug("Deterministic extractor answered before retrieval.")
        return answer, hits, raw, debug_payload

    hits = retrieve(question, k=k)
    retrieval_debug = get_last_retrieval_debug()

    hits = inject_rank_passage_if_needed(question, hits)
    hits = inject_leadership_passage_if_needed(question, hits)
    hits = inject_schools_passage_if_needed(question, hits)
    hits = inject_weapons_passage_if_needed(question, hits)
    hits = inject_kihon_passage_if_needed(question, hits)
    hits = inject_techniques_passage_if_needed(question, hits)
    hits = inject_specific_technique_line_if_needed(question, hits)

    deterministic_answer = _answer_from_passages(question, hits)
    if deterministic_answer:
        answer, det_hits, raw = deterministic_answer
        return answer, det_hits, raw, retrieval_debug

    ctx = build_context(hits)
    prompt = build_prompt(ctx, question)
    text, raw = call_llm(prompt)
    if not text.strip():
        return "🔒 Strict (context-only)\n\nModel returned no text.", hits, raw or "{}", retrieval_debug
    return f"🔒 Strict (context-only, explain)\n\n{text.strip()}", hits, raw or "{}", retrieval_debug


st.set_page_config(page_title="NTTV Chatbot (RAG)", page_icon="🥋", layout="wide")

st.title("🥋 NTTV Chatbot (RAG)")

with st.sidebar:
    st.markdown("### Options")
    show_debug = st.checkbox("Show debugging", value=True)

    st.markdown("### Output")
    output_style = st.radio("Format", ["Bullets", "Paragraph"], index=0, help="Affects deterministic answers only.")
    tone_style = st.radio("Tone", ["Crisp", "Chatty"], index=0, help="Affects deterministic answers only.")
    st.caption("Deterministic answers = school profiles, rank requirements, weapon-rank facts, technique definitions, etc.")

    TECH_DETAIL_MODE = st.selectbox(
        "Technique detail",
        options=["Brief", "Standard", "Full"],
        index=1,
        help="How much detail to show for single-technique answers."
    )

    st.markdown("---")
    st.markdown("**Backend**")
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_API_BASE")
        or os.environ.get("LM_STUDIO_BASE_URL")
        or "http://localhost:1234/v1"
    )
    model = os.environ.get("MODEL", "gpt-4o-mini")
    st.caption(f"LLM base: `{base}`")
    st.caption(f"Model: `{model}`")
    
    if show_debug:
        st.markdown("---")
        with st.expander("Index diagnostics", expanded=False):
            index_dir = os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)
            config_path = os.getenv("CONFIG_PATH", os.path.join(index_dir, "config.json"))
            meta_path = os.getenv("META_PATH", os.path.join(index_dir, "meta.pkl"))

            st.write("**Working directory:**", os.getcwd())
            st.write("**__file__ dir:**", os.path.dirname(__file__))
            st.write("**INDEX_DIR:**", index_dir)
            st.write("**CONFIG_PATH:**", config_path, "✅" if os.path.exists(config_path) else "❌")
            st.write("**META_PATH:**", meta_path, "✅" if os.path.exists(meta_path) else "❌")

            # Likely FAISS locations
            idx_env = os.getenv("INDEX_PATH")
            if idx_env:
                st.write("**INDEX_PATH (env):**", idx_env, "✅" if os.path.exists(idx_env) else "❌")

            faiss_guess_1 = os.path.join(index_dir, "index.faiss")
            faiss_guess_2 = os.path.join(index_dir, "faiss.index")
            st.write("**FAISS candidate 1:**", faiss_guess_1, "✅" if os.path.exists(faiss_guess_1) else "❌")
            st.write("**FAISS candidate 2:**", faiss_guess_2, "✅" if os.path.exists(faiss_guess_2) else "❌")
    

q = st.text_input("Ask a question:", value="", placeholder="e.g., what is omote gyaku")
go = st.button("Ask", type="primary")

if go and q.strip():
    try:
        with st.spinner("Thinking..."):
            ans, top_passages, raw_json, retrieval_debug = answer_with_rag(q.strip())
    except Exception as e:
        st.error(f"Backend error: {e}")
        if show_debug:
            st.exception(e)
        st.stop()

    st.markdown("### Answer")
    st.write(ans)

    if show_debug:
        st.markdown("### Retrieval Pipeline")
        if retrieval_debug.get("deterministic_short_circuit"):
            st.caption("Deterministic extractor answered before retrieval, so hybrid retrieval did not run.")
        else:
            st.caption(
                f"Reranker requested: {retrieval_debug.get('reranker_backend_requested')} | "
                f"used: {retrieval_debug.get('reranker_backend_used')}"
            )
            fallback_reason = retrieval_debug.get("reranker_fallback_reason")
            if fallback_reason:
                st.caption(f"Fallback: {fallback_reason}")

            _render_stage_candidates("Dense candidates", retrieval_debug.get("dense_candidates") or [])
            _render_stage_candidates("Lexical candidates", retrieval_debug.get("lexical_candidates") or [])
            _render_stage_candidates("Fused candidates", retrieval_debug.get("fused_candidates") or [])
            _render_stage_candidates("Reranked candidates", retrieval_debug.get("reranked_candidates") or [])

        st.markdown("### Final Answer Sources")
        for i, h in enumerate(top_passages, 1):
            meta = h.get("meta") or {}
            source_file = meta.get("source_file") or h.get("source") or ""
            name = os.path.basename(source_file)
            page_range = _format_page_range(meta)
            heading_path = " > ".join(meta.get("heading_path") or [])
            tags = _format_detected_tags(meta)
            priority_bucket = meta.get("priority_bucket") or f"p{max(1, 4 - int(meta.get('priority', 1) or 1))}"
            if page_range:
                st.caption(f"Pages: {page_range}")
            if heading_path:
                st.caption(f"Heading: {heading_path}")
            st.caption(f"Priority bucket: {priority_bucket}")
            if tags:
                st.caption(f"Tags: {tags}")
            st.write(
                f"[{i}] {name} — score {h.get('score', 0):.3f} — "
                f"priority {int(h.get('meta',{}).get('priority',0))}"
            )
        st.markdown("### Raw model response (JSON-ish)")
        st.code(raw_json, language="json")

else:
    st.info("Enter a question and click **Ask**.")
