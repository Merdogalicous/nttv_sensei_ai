from __future__ import annotations

from typing import Any

import app
from nttv_chatbot import retrieval


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "data/notes.txt",
    priority: int = 1,
    heading_path: list[str] | None = None,
) -> dict[str, Any]:
    meta = {
        "chunk_id": chunk_id,
        "source": source,
        "source_file": source,
        "priority": priority,
        "priority_bucket": "p3",
        "heading_path": heading_path or [],
        "page": None,
        "page_start": None,
        "page_end": None,
        "rank_tag": None,
        "school_tag": None,
        "weapon_tag": None,
        "technique_tag": None,
    }
    return {
        "text": text,
        "source": source,
        "chunk_id": chunk_id,
        "meta": meta,
    }


def _candidate(
    chunk_id: str,
    *,
    dense_rank: int | None = None,
    dense_score: float | None = None,
    lexical_rank: int | None = None,
    lexical_score: float | None = None,
    heuristic_score: float | None = None,
) -> dict[str, Any]:
    chunk = _chunk(chunk_id, f"content for {chunk_id}", source=f"data/{chunk_id}.txt")
    candidate = retrieval._candidate_from_chunk(chunk, 0)
    candidate["matched_stages"] = []
    if dense_rank is not None:
        candidate["dense_rank"] = dense_rank
        candidate["dense_score"] = dense_score
        candidate["matched_stages"].append("dense")
    if lexical_rank is not None:
        candidate["lexical_rank"] = lexical_rank
        candidate["lexical_score"] = lexical_score
        candidate["matched_stages"].append("lexical")
    if heuristic_score is not None:
        candidate["heuristic_score"] = heuristic_score
        candidate["rrf_score"] = 0.25
    return candidate


def test_lexical_retrieval_returns_candidates():
    chunks = [
        _chunk("c1", "Hanbo weapon profile with kamae and striking notes.", source="data/NTTV Weapons Reference.txt"),
        _chunk("c2", "Sanshin no Kata overview and training notes."),
        _chunk("c3", "School summaries for the Bujinkan lineages."),
    ]

    lexical = retrieval.LexicalRetriever(chunks)
    results = lexical.search("hanbo weapon", top_k=3)

    assert results
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["lexical_score"] > 0


def test_rrf_fusion_merges_dense_and_lexical_candidates():
    dense = [
        _candidate("a", dense_rank=1, dense_score=0.90),
        _candidate("b", dense_rank=2, dense_score=0.80),
    ]
    lexical = [
        _candidate("b", lexical_rank=1, lexical_score=6.0),
        _candidate("c", lexical_rank=2, lexical_score=5.0),
    ]

    fused = retrieval.fuse_candidate_rankings(dense, lexical, top_k=10)
    fused_ids = [candidate["chunk_id"] for candidate in fused]

    assert fused_ids[:3] == ["b", "a", "c"]
    assert set(fused_ids) == {"a", "b", "c"}
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_reranker_interface_falls_back_safely_without_jina_key():
    candidates = [
        _candidate("a", dense_rank=1, dense_score=0.9, heuristic_score=0.8),
        _candidate("b", dense_rank=2, dense_score=0.8, heuristic_score=0.7),
    ]

    settings = retrieval.RetrievalSettings(
        use_hybrid_retrieval=True,
        dense_top_k=12,
        lexical_top_k=12,
        fused_top_k=10,
        reranker_backend="jina_api",
        jina_api_key="",
        jina_api_url="https://api.jina.ai/v1/rerank",
        jina_model="jina-reranker-v2-base-multilingual",
    )

    reranked, backend_used, fallback_reason = retrieval.apply_optional_reranker(
        "hanbo",
        candidates,
        settings=settings,
    )

    assert reranked == candidates
    assert backend_used == "heuristic_only"
    assert fallback_reason is not None
    assert "JINA_API_KEY" in fallback_reason


def test_deterministic_extractors_short_circuit_before_retrieval(monkeypatch):
    monkeypatch.setattr(app, "_load_index_and_meta", lambda: (None, []))
    monkeypatch.setattr(
        app,
        "retrieve",
        lambda question, k=None: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )
    monkeypatch.setattr(app, "output_style", "Bullets", raising=False)
    monkeypatch.setattr(app, "tone_style", "Crisp", raising=False)
    monkeypatch.setattr(app, "TECH_DETAIL_MODE", "Standard", raising=False)

    answer, passages, raw_json, retrieval_debug = app.answer_with_rag("What is Sanshin no Kata?")

    assert "Sanshin no Kata" in answer
    assert passages == []
    assert "det_path" in raw_json
    assert retrieval_debug["deterministic_short_circuit"] is True
