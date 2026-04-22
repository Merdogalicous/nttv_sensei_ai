from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import app
from nttv_chatbot.deterministic import DeterministicResult, SourceRef


TECH = Path("data") / "Technique Descriptions.md"


def _oni_kudaki_row() -> str:
    prefix = "Oni Kudaki,"
    for line in TECH.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError("Missing Oni Kudaki row in technique data")


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str,
    priority: int = 1,
) -> dict[str, Any]:
    meta = {
        "chunk_id": chunk_id,
        "source": source,
        "source_file": source,
        "priority": priority,
        "priority_bucket": "p3",
        "heading_path": [],
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


def _routing_debug(hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dense_candidates": [],
        "lexical_candidates": [],
        "fused_candidates": [],
        "reranked_candidates": [],
        "final_candidates": hits,
        "reranker_backend_requested": "none",
        "reranker_backend_used": "none",
        "reranker_fallback_reason": None,
    }


def _set_render_defaults(monkeypatch) -> None:
    monkeypatch.setattr(app, "output_style", "Bullets", raising=False)
    monkeypatch.setattr(app, "tone_style", "Crisp", raising=False)
    monkeypatch.setattr(app, "TECH_DETAIL_MODE", "Standard", raising=False)


def _oni_kudaki_result() -> DeterministicResult:
    return DeterministicResult(
        answered=True,
        det_path="technique/core",
        answer_type="technique",
        facts={
            "technique_name": "Oni Kudaki",
            "japanese": "Oni Kudaki",
            "translation": "Demon Crusher",
            "type": "Arm Control",
            "rank_context": "8th Kyu",
            "primary_focus": "arm control and structure breaking",
            "safety": "Apply with control and protect the elbow and shoulder.",
            "partner_required": True,
            "solo": False,
            "tags": ["lock", "kihon"],
            "definition": "A control that folds the arm structure to break posture and take balance.",
        },
        source_refs=[SourceRef(source="Technique Descriptions.md")],
        confidence=0.98,
        display_hints={"explain": True},
        followup_suggestions=[],
    )


def test_followup_stays_anchored_to_previous_oni_kudaki_answer(monkeypatch):
    chunks = [
        _chunk("oni", _oni_kudaki_row(), source="data/Technique Descriptions.md"),
        _chunk("schema", "Technique schema fields: name, translation, type, rank.", source="data/schema.txt"),
    ]

    monkeypatch.setattr(app, "_load_index_and_meta", lambda: (None, chunks))
    monkeypatch.setattr(
        app,
        "retrieve",
        lambda question, k=None: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )
    _set_render_defaults(monkeypatch)

    session: dict[str, Any] = {}

    first_answer, _, _, _ = app.answer_with_rag("describe oni kudaki", session_state=session)
    second_answer, hits, _, debug = app.answer_with_rag("unpack one part further", session_state=session)

    assert "oni kudaki" in first_answer.lower()
    assert "oni kudaki" in second_answer.lower()
    assert "schema" not in second_answer.lower()
    assert hits == []
    assert debug["followup"]["used_prior_topic"] is True
    assert debug["followup"]["resolved_topic"] == "Oni Kudaki"


def test_followup_resolution_uses_prior_session_topic():
    session = {
        app._SESSION_FOLLOWUP_STATE_KEY: {
            "last_user_question": "describe oni kudaki",
            "last_answer_text": "anchored answer",
            "last_det_path": None,
            "last_answer_type": "grounded_generation",
            "last_facts": {},
            "last_source_refs": [],
            "last_resolved_topic": "Oni Kudaki",
        }
    }

    resolution = app.resolve_followup_question("say more", session)

    assert resolution.is_vague_followup is True
    assert resolution.used_prior_topic is True
    assert resolution.effective_question == "Explain Oni Kudaki in more detail."


def test_vague_followup_without_prior_topic_asks_for_clarification(monkeypatch):
    monkeypatch.setattr(
        app,
        "retrieve",
        lambda question, k=None: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )

    answer, hits, raw_json, debug = app.answer_with_rag("unpack one part further", session_state={})

    assert answer == app._FOLLOWUP_CLARIFICATION
    assert hits == []
    assert raw_json == "{}"
    assert debug["followup"]["needs_clarification"] is True


def test_deterministic_followup_reuses_prior_facts_safely(monkeypatch):
    _set_render_defaults(monkeypatch)
    session: dict[str, Any] = {}
    app._remember_last_answer(
        session,
        original_question="describe oni kudaki",
        effective_question="describe oni kudaki",
        answer_text="previous answer",
        deterministic_result=_oni_kudaki_result(),
    )

    monkeypatch.setattr(
        app,
        "_prepare_deterministic_passages",
        lambda question: (_ for _ in ()).throw(AssertionError("cached deterministic result should be reused")),
    )
    monkeypatch.setattr(
        app,
        "retrieve",
        lambda question, k=None: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )

    answer, hits, _, debug = app.answer_with_rag("tell me more", session_state=session)

    assert "oni kudaki" in answer.lower()
    assert hits == []
    assert debug["followup"]["used_prior_topic"] is True


def test_followup_empty_model_output_returns_safe_message(monkeypatch):
    hits = [_chunk("schema", "Technique schema fields only.", source="data/schema.txt")]

    monkeypatch.setattr(app, "_prepare_deterministic_passages", lambda question: [])
    monkeypatch.setattr(app, "_answer_from_passages", lambda question, passages: None)
    monkeypatch.setattr(app, "retrieve", lambda question, k=None: hits)
    monkeypatch.setattr(app, "get_last_retrieval_debug", lambda: _routing_debug(hits))
    monkeypatch.setattr(
        app,
        "generate_grounded_answer",
        lambda question, context_chunks: SimpleNamespace(
            text="",
            raw_json="{}",
            debug={
                "route": "primary",
                "model_requested": "openai/gpt-4o-mini",
                "model_used": "openai/gpt-4o-mini",
                "reason": "Primary model used because the question is direct and the retrieved context is small.",
                "reason_codes": ["direct_question"],
                "deterministic_mode": False,
                "explanation_mode": True,
                "interpretive_question": False,
                "input_chunk_count": len(context_chunks),
                "selected_chunk_count": len(context_chunks),
                "input_fact_count": 0,
                "selected_fact_count": 0,
                "context_char_count": 128,
                "fallback_used": False,
                "fallback_reason": "openai/gpt-4o-mini returned no text.",
                "attempted_models": ["openai/gpt-4o-mini"],
            },
        ),
    )

    session = {
        app._SESSION_FOLLOWUP_STATE_KEY: {
            "last_user_question": "tell me about oni kudaki",
            "last_answer_text": "anchored answer",
            "last_det_path": None,
            "last_answer_type": "grounded_generation",
            "last_facts": {},
            "last_source_refs": [],
            "last_resolved_topic": "Oni Kudaki",
        }
    }

    answer, returned_hits, raw_json, debug = app.answer_with_rag(
        "unpack one part further",
        session_state=session,
    )

    assert returned_hits
    assert raw_json == "{}"
    assert "oni kudaki" in answer.lower()
    assert "model returned no text" not in answer.lower()
    assert debug["followup"]["used_prior_topic"] is True
    assert debug["followup"]["effective_question"] == "Explain Oni Kudaki in more detail."
