# app.py (patched)
import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import unicodedata

import numpy as np
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from nttv_chatbot.composer import compose_deterministic_answer
from nttv_chatbot.deterministic import DeterministicResult, SourceRef, build_result
from nttv_chatbot.retrieval import (
    LexicalRetriever,
    RetrievalResult,
    RetrievalSettings,
    assemble_context as assemble_retrieval_context,
    search as search_retrieval_pipeline,
)
from nttv_chatbot.llm_routing import (
    LLMRoutingSettings,
    empty_route_debug,
    filter_supporting_chunks,
    generate_grounded_answer,
    select_generation_route,
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
from extractors import try_extract_answer
from extractors.rank import try_answer_rank_requirements
from extractors.kamae import try_answer_kamae
from extractors.lineage_people import try_answer_lineage_person
from extractors.weapons import try_answer_weapon_rank
from extractors.schools import (
    try_answer_school_catalog,
    try_answer_school_profile,
    try_answer_schools_list,   # list extractor
    SCHOOL_ALIASES,
    is_school_catalog_query,
    is_school_list_query,
)

from extractors.technique_match import (
    canonical_from_query as _canonical_technique_from_query,
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

    faiss_module = faiss
    if faiss_module is None:
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
        idx_local = faiss_module.read_index(fpath)
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
        "llm_routing": empty_route_debug(reason),
    }


def _with_llm_routing_debug(
    payload: Dict[str, Any],
    llm_debug: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = dict(payload)
    merged["llm_routing"] = llm_debug or empty_route_debug("No answer routing details were recorded.")
    return merged


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


_SESSION_FOLLOWUP_STATE_KEY = "last_answer_state"
_VAGUE_FOLLOWUP_PHRASES = {
    "unpack one part further",
    "go deeper",
    "say more",
    "expand on that",
    "tell me more",
    "can you explain that more",
    "break that down",
}
_FOLLOWUP_CLARIFICATION = "I can do that, but I need to know what you want me to unpack further."
_EMPTY_GROUNDED_ANSWER = (
    "I couldn't produce a grounded answer from the available material. "
    "Please name the topic more directly."
)


@dataclass(frozen=True)
class FollowupResolution:
    original_question: str
    effective_question: str
    is_vague_followup: bool = False
    used_prior_topic: bool = False
    resolved_topic: Optional[str] = None
    clarification_text: Optional[str] = None
    cached_result: Optional[DeterministicResult] = None

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification_text)


def _normalize_followup_prompt(question: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (question or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.startswith("please "):
        normalized = normalized[7:].strip()
    if normalized.endswith(" please"):
        normalized = normalized[:-7].strip()
    return normalized


def is_vague_followup_prompt(question: str) -> bool:
    return _normalize_followup_prompt(question) in _VAGUE_FOLLOWUP_PHRASES


def _get_followup_state(
    session_state: Any | None,
) -> Dict[str, Any]:
    if session_state is None:
        return {}
    raw = session_state.get(_SESSION_FOLLOWUP_STATE_KEY)
    return dict(raw) if isinstance(raw, dict) else {}


def _serialize_source_refs(source_refs: list[SourceRef]) -> list[dict[str, Any]]:
    return [ref.to_dict() for ref in source_refs]


def _restore_source_refs(raw_refs: Any) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for raw_ref in raw_refs or []:
        if isinstance(raw_ref, SourceRef):
            refs.append(raw_ref)
            continue
        if not isinstance(raw_ref, dict):
            continue
        refs.append(
            SourceRef(
                source=str(raw_ref.get("source") or ""),
                page_start=raw_ref.get("page_start"),
                page_end=raw_ref.get("page_end"),
                heading_path=list(raw_ref.get("heading_path") or []),
                chunk_id=raw_ref.get("chunk_id"),
            )
        )
    return refs


def _resolved_topic_from_result(result: DeterministicResult) -> Optional[str]:
    facts = result.facts
    if result.answer_type == "technique":
        return facts.get("technique_name")
    if result.answer_type == "technique_diff":
        left_name = ((facts.get("left") or {}).get("technique_name") or "").strip()
        right_name = ((facts.get("right") or {}).get("technique_name") or "").strip()
        if left_name and right_name:
            return f"{left_name} and {right_name}"
        return left_name or right_name or None
    if result.answer_type in {"school_profile", "school_list", "school_catalog"}:
        return facts.get("school_name") or facts.get("list_title")
    if result.answer_type in {"weapon_profile", "weapon_rank", "weapon_parts", "weapon_classification"}:
        return facts.get("weapon_name") or facts.get("title")
    if result.answer_type == "glossary_term":
        return facts.get("term")
    if result.answer_type == "kyusho_point":
        return facts.get("point_name")
    if result.answer_type == "leadership":
        return facts.get("soke_name") or facts.get("school_name")
    if result.answer_type == "lineage_person":
        return facts.get("person_name") or facts.get("related_person")
    if result.answer_type == "sanshin_element":
        return facts.get("element_name")
    if result.answer_type in {"sanshin_list", "sanshin_overview"}:
        return facts.get("title") or "Sanshin no Kata"
    if result.answer_type == "kihon_happo":
        return "Kihon Happo"
    if result.answer_type.startswith("rank_") or result.answer_type == "rank_requirements":
        return facts.get("rank")
    return None


def _resolved_topic_from_question(question: str) -> Optional[str]:
    return _canonical_technique_from_query(question)


def _restore_deterministic_result(state: Dict[str, Any]) -> Optional[DeterministicResult]:
    det_path = (state.get("last_det_path") or "").strip()
    answer_type = (state.get("last_answer_type") or "").strip()
    facts = state.get("last_facts")
    if not det_path or not answer_type or not isinstance(facts, dict) or not facts:
        return None
    return DeterministicResult(
        answered=True,
        det_path=det_path,
        answer_type=answer_type,
        facts=dict(facts),
        source_refs=_restore_source_refs(state.get("last_source_refs")),
        confidence=float(state.get("last_confidence") or 1.0),
        display_hints=dict(state.get("last_display_hints") or {}),
        followup_suggestions=list(state.get("last_followup_suggestions") or []),
    )


def _rewrite_vague_followup(topic: str) -> str:
    clean_topic = (topic or "").strip().rstrip(".!?")
    return f"Explain {clean_topic} in more detail."


def resolve_followup_question(
    question: str,
    session_state: Any | None = None,
) -> FollowupResolution:
    original_question = (question or "").strip()
    if not is_vague_followup_prompt(original_question):
        return FollowupResolution(
            original_question=original_question,
            effective_question=original_question,
        )

    state = _get_followup_state(session_state)
    topic = (state.get("last_resolved_topic") or "").strip()
    if not topic:
        return FollowupResolution(
            original_question=original_question,
            effective_question=original_question,
            is_vague_followup=True,
            clarification_text=_FOLLOWUP_CLARIFICATION,
        )

    return FollowupResolution(
        original_question=original_question,
        effective_question=_rewrite_vague_followup(topic),
        is_vague_followup=True,
        used_prior_topic=True,
        resolved_topic=topic,
        cached_result=_restore_deterministic_result(state),
    )


def _remember_last_answer(
    session_state: Any | None,
    *,
    original_question: str,
    effective_question: str,
    answer_text: str,
    deterministic_result: Optional[DeterministicResult] = None,
) -> None:
    if session_state is None:
        return

    resolved_topic = None
    if deterministic_result is not None:
        resolved_topic = _resolved_topic_from_result(deterministic_result)
    if not resolved_topic:
        resolved_topic = _resolved_topic_from_question(effective_question) or _resolved_topic_from_question(original_question)

    session_state[_SESSION_FOLLOWUP_STATE_KEY] = {
        "last_user_question": original_question,
        "last_effective_question": effective_question,
        "last_answer_text": answer_text.strip(),
        "last_det_path": deterministic_result.det_path if deterministic_result else None,
        "last_answer_type": deterministic_result.answer_type if deterministic_result else "grounded_generation",
        "last_facts": dict(deterministic_result.facts) if deterministic_result else {},
        "last_source_refs": _serialize_source_refs(deterministic_result.source_refs) if deterministic_result else [],
        "last_resolved_topic": resolved_topic,
        "last_confidence": deterministic_result.confidence if deterministic_result else None,
        "last_display_hints": dict(deterministic_result.display_hints) if deterministic_result else {},
        "last_followup_suggestions": list(deterministic_result.followup_suggestions) if deterministic_result else [],
    }


def _with_followup_debug(
    payload: Dict[str, Any],
    resolution: FollowupResolution,
) -> Dict[str, Any]:
    merged = dict(payload)
    if resolution.is_vague_followup:
        merged["followup"] = {
            "original_question": resolution.original_question,
            "effective_question": resolution.effective_question,
            "used_prior_topic": resolution.used_prior_topic,
            "resolved_topic": resolution.resolved_topic,
            "needs_clarification": resolution.needs_clarification,
        }
    return merged


def _empty_answer_text(resolution: FollowupResolution) -> str:
    if resolution.is_vague_followup and resolution.resolved_topic:
        return (
            f"I can unpack {resolution.resolved_topic} further, "
            "but I need a more specific angle to stay grounded."
        )
    if resolution.is_vague_followup:
        return _FOLLOWUP_CLARIFICATION
    return _EMPTY_GROUNDED_ANSWER


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


def _detail_style() -> str:
    value = (TECH_DETAIL_MODE or "Standard").strip().lower()
    if value not in {"brief", "standard", "full"}:
        return "standard"
    return value


def _deterministic_output_format() -> str:
    return "bullets" if output_style == "Bullets" else "paragraph"


def _compose_deterministic_result(
    question: str,
    result: DeterministicResult,
    passages: List[Dict[str, Any]],
    *,
    style_override: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    supporting_passages = filter_supporting_chunks(passages, result.source_refs, limit=6)
    route_decision = select_generation_route(
        question,
        supporting_passages,
        fact_count=len(result.facts),
        deterministic_mode=True,
    )
    strict_label = "🔒 Strict (context-only, explain)" if result.display_hints.get("explain", True) else "🔒 Strict (context-only)"

    if route_decision.use_model:
        generated = generate_grounded_answer(
            question,
            supporting_passages,
            facts=result.facts,
            source_refs=result.source_refs,
            deterministic_mode=True,
        )
        if generated.text.strip():
            return f"{strict_label}\n\n{generated.text.strip()}", generated.raw_json, generated.debug

        route_debug = dict(generated.debug)
        route_debug["model_used"] = "deterministic_composer"
        route_debug["local_composer_fallback"] = True
        route_debug["fallback_used"] = True
        route_debug["fallback_reason"] = (
            route_debug.get("fallback_reason")
            or "Synthesis composition returned no text; used local deterministic composer."
        )
    else:
        route_debug = route_decision.to_debug_payload(
            model_used="deterministic_composer",
            selected_chunk_count=len(supporting_passages),
            selected_fact_count=len(result.facts),
            context_char_count=0,
        )

    body = compose_deterministic_answer(
        result,
        style=style_override or _detail_style(),
        output_format=_deterministic_output_format(),
        explanation_mode=True,
        tone=tone_style,
    )
    return f"{strict_label}\n\n{body}", json.dumps(result.to_dict(), ensure_ascii=False), route_debug


def answer_single_technique_if_synthetic(
    passages: List[Dict[str, Any]],
    *,
    bullets: bool,
    tone: str,
    detail_mode: str
) -> Optional[DeterministicResult]:
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
    tags_raw = row.get("tags") or ""
    tags = [item.strip() for item in tags_raw.split(",") if item.strip()]
    return build_result(
        det_path="technique/single",
        answer_type="technique",
        facts={
            "technique_name": row.get("name"),
            "japanese": row.get("japanese"),
            "translation": row.get("english"),
            "type": row.get("family"),
            "rank_context": row.get("rank_intro"),
            "primary_focus": row.get("focus"),
            "safety": row.get("safety"),
            "partner_required": row.get("partner_required"),
            "solo": row.get("solo"),
            "tags": tags,
            "definition": row.get("definition"),
        },
        preferred_sources=["Technique Descriptions.md"],
        confidence=0.99,
        display_hints={"explain": True},
    )


# --- Technique CSV line injector (for single-technique queries) ----------------
import re as _re2  # avoid shadowing

def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


def _extract_single_technique_candidate(q: str) -> Optional[str]:
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


def _candidate_technique_name_variants(name: str) -> list[str]:
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
    cand = _extract_single_technique_candidate(question)
    if not cand:
        return passages

    variants = _candidate_technique_name_variants(cand)
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


# --- School intent detection ---
def is_school_query(question: str) -> bool:
    ql = question.lower()
    for canon, aliases in SCHOOL_ALIASES.items():
        tokens = [canon.lower()] + [a.lower() for a in aliases]
        if any(tok in ql for tok in tokens):
            return True
    return (" ryu" in ql) or (" ryū" in ql)




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


# --------------------------------------------------------------------
# Deterministic bridge
# --------------------------------------------------------------------
def _answer_from_passages(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[Tuple[str, List[Dict[str, Any]], str, Dict[str, Any], DeterministicResult]]:
    fast = answer_single_technique_if_synthetic(
        passages,
        bullets=(output_style == "Bullets"),
        tone=tone_style,
        detail_mode=TECH_DETAIL_MODE,
    )
    if fast:
        answer_text, raw, route_debug = _compose_deterministic_result(question, fast, passages)
        return answer_text, passages, raw, route_debug, fast

    if is_school_catalog_query(question):
        try:
            catalog_ans = try_answer_school_catalog(
                question, passages, bullets=(output_style == "Bullets")
            )
        except Exception:
            catalog_ans = None
        if catalog_ans:
            answer_text, raw, route_debug = _compose_deterministic_result(question, catalog_ans, passages)
            return answer_text, passages, raw, route_debug, catalog_ans

    if is_school_list_query(question):
        try:
            list_ans = try_answer_schools_list(
                question, passages, bullets=(output_style == "Bullets")
            )
        except Exception:
            list_ans = None
        if list_ans:
            answer_text, raw, route_debug = _compose_deterministic_result(question, list_ans, passages)
            return answer_text, passages, raw, route_debug, list_ans

    if is_school_query(question):
        try:
            school_fact = try_answer_school_profile(
                question, passages, bullets=(output_style == "Bullets")
            )
        except Exception:
            school_fact = None
        if school_fact:
            answer_text, raw, route_debug = _compose_deterministic_result(question, school_fact, passages)
            return answer_text, passages, raw, route_debug, school_fact

    try:
        wr = try_answer_weapon_rank(question, passages)
    except Exception:
        wr = None
    if wr:
        answer_text, raw, route_debug = _compose_deterministic_result(question, wr, passages)
        return answer_text, passages, raw, route_debug, wr

    try:
        rank_requirements = try_answer_rank_requirements(question, passages)
    except Exception:
        rank_requirements = None
    if rank_requirements:
        answer_text, raw, route_debug = _compose_deterministic_result(
            question,
            rank_requirements,
            passages,
        )
        return answer_text, passages, raw, route_debug, rank_requirements

    try:
        kamae_result = try_answer_kamae(question, passages)
    except Exception:
        kamae_result = None
    if kamae_result:
        answer_text, raw, route_debug = _compose_deterministic_result(question, kamae_result, passages)
        return answer_text, passages, raw, route_debug, kamae_result

    try:
        lineage_person = try_answer_lineage_person(question, passages)
    except Exception:
        lineage_person = None
    if lineage_person:
        answer_text, raw, route_debug = _compose_deterministic_result(question, lineage_person, passages)
        return answer_text, passages, raw, route_debug, lineage_person

    fact = try_extract_answer(question, passages)
    if fact:
        answer_text, raw, route_debug = _compose_deterministic_result(question, fact, passages)
        return answer_text, passages, raw, route_debug, fact

    return None


def _answer_from_cached_followup_result(
    question: str,
    result: DeterministicResult,
) -> Tuple[str, List[Dict[str, Any]], str, Dict[str, Any]]:
    answer_text, raw, route_debug = _compose_deterministic_result(
        question,
        result,
        [],
        style_override="full",
    )
    debug_payload = _with_llm_routing_debug(
        _empty_retrieval_debug("Reused prior deterministic answer for a vague follow-up."),
        route_debug,
    )
    return answer_text, [], raw, debug_payload


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
def answer_with_rag(
    question: str,
    k: int | None = None,
    *,
    session_state: Any | None = None,
) -> Tuple[str, List[Dict[str, Any]], str, Dict[str, Any]]:
    global _LAST_RETRIEVAL_RESULT

    if k is None:
        k = TOP_K

    resolution = resolve_followup_question(question, session_state)
    effective_question = resolution.effective_question

    if resolution.needs_clarification:
        _LAST_RETRIEVAL_RESULT = None
        debug_payload = _with_followup_debug(
            _empty_retrieval_debug("Vague follow-up requested without a prior resolved topic."),
            resolution,
        )
        return resolution.clarification_text or _FOLLOWUP_CLARIFICATION, [], "{}", debug_payload

    if resolution.cached_result is not None:
        _LAST_RETRIEVAL_RESULT = None
        answer, hits, raw, retrieval_debug = _answer_from_cached_followup_result(
            effective_question,
            resolution.cached_result,
        )
        retrieval_debug = _with_followup_debug(retrieval_debug, resolution)
        _remember_last_answer(
            session_state,
            original_question=resolution.original_question,
            effective_question=effective_question,
            answer_text=answer,
            deterministic_result=resolution.cached_result,
        )
        return answer, hits, raw, retrieval_debug

    prepass_passages = _prepare_deterministic_passages(effective_question)
    prepass_answer = _answer_from_passages(effective_question, prepass_passages)
    if prepass_answer:
        _LAST_RETRIEVAL_RESULT = None
        answer, hits, raw, route_debug, det_result = prepass_answer
        retrieval_debug = _with_llm_routing_debug(
            _empty_retrieval_debug("Deterministic extractor answered before retrieval."),
            route_debug,
        )
        retrieval_debug = _with_followup_debug(retrieval_debug, resolution)
        _remember_last_answer(
            session_state,
            original_question=resolution.original_question,
            effective_question=effective_question,
            answer_text=answer,
            deterministic_result=det_result,
        )
        return answer, hits, raw, retrieval_debug

    hits = retrieve(effective_question, k=k)
    retrieval_debug = dict(get_last_retrieval_debug())

    hits = inject_rank_passage_if_needed(effective_question, hits)
    hits = inject_leadership_passage_if_needed(effective_question, hits)
    hits = inject_schools_passage_if_needed(effective_question, hits)
    hits = inject_weapons_passage_if_needed(effective_question, hits)
    hits = inject_kihon_passage_if_needed(effective_question, hits)
    hits = inject_techniques_passage_if_needed(effective_question, hits)
    hits = inject_specific_technique_line_if_needed(effective_question, hits)

    deterministic_answer = _answer_from_passages(effective_question, hits)
    if deterministic_answer:
        answer, det_hits, raw, route_debug, det_result = deterministic_answer
        retrieval_debug = _with_llm_routing_debug(retrieval_debug, route_debug)
        retrieval_debug = _with_followup_debug(retrieval_debug, resolution)
        _remember_last_answer(
            session_state,
            original_question=resolution.original_question,
            effective_question=effective_question,
            answer_text=answer,
            deterministic_result=det_result,
        )
        return answer, det_hits, raw, retrieval_debug

    generated = generate_grounded_answer(effective_question, hits)
    retrieval_debug = _with_llm_routing_debug(retrieval_debug, generated.debug)
    retrieval_debug = _with_followup_debug(retrieval_debug, resolution)
    if not generated.text.strip():
        return _empty_answer_text(resolution), hits, generated.raw_json or "{}", retrieval_debug

    answer = f"\U0001F512 Strict (context-only, explain)\n\n{generated.text.strip()}"
    _remember_last_answer(
        session_state,
        original_question=resolution.original_question,
        effective_question=effective_question,
        answer_text=answer,
    )
    return answer, hits, generated.raw_json or "{}", retrieval_debug


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
    routing_settings = LLMRoutingSettings.from_env()
    st.caption(f"LLM base: `{routing_settings.api_base}`")
    st.caption(f"Primary model: `{routing_settings.primary_model}`")
    if routing_settings.use_synthesis_model:
        synthesis_label = routing_settings.synthesis_model or "(not set)"
        st.caption(f"Synthesis model: `{synthesis_label}`")
        st.caption(
            "Synthesis routing: "
            f"min_chunks={routing_settings.synthesis_min_context_chunks}, "
            f"explanation={routing_settings.synthesis_for_explanation_mode}, "
            f"deterministic={routing_settings.synthesis_for_deterministic_composer}"
        )
    else:
        st.caption("Synthesis model routing: disabled")
    
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
            ans, top_passages, raw_json, retrieval_debug = answer_with_rag(
                q.strip(),
                session_state=st.session_state,
            )
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

        llm_debug = retrieval_debug.get("llm_routing") or {}
        if llm_debug:
            st.markdown("### Answer Routing")
            st.caption(
                f"Route: {llm_debug.get('route')} | "
                f"model used: {llm_debug.get('model_used') or 'none'} | "
                f"requested: {llm_debug.get('model_requested') or 'none'}"
            )
            st.caption(f"Reason: {llm_debug.get('reason') or 'n/a'}")
            st.caption(
                f"Chunks supplied: {llm_debug.get('selected_chunk_count', 0)} / {llm_debug.get('input_chunk_count', 0)} | "
                f"Facts supplied: {llm_debug.get('selected_fact_count', 0)} / {llm_debug.get('input_fact_count', 0)}"
            )
            if llm_debug.get("fallback_used"):
                st.caption(f"Model fallback: {llm_debug.get('fallback_reason')}")

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
