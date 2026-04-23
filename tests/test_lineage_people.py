from __future__ import annotations

import pathlib
from typing import Any

import app
from extractors.lineage_people import (
    BUYU_SOURCE,
    LEADERSHIP_SOURCE,
    TRAINING_SOURCE,
    try_answer_lineage_person,
)
from tests.helpers import render_result


DATA = pathlib.Path("data")
TRAINING = DATA / TRAINING_SOURCE
BUYU = DATA / BUYU_SOURCE
LEADERSHIP = DATA / LEADERSHIP_SOURCE


def _passages() -> list[dict[str, Any]]:
    return [
        {
            "text": TRAINING.read_text(encoding="utf-8"),
            "source": TRAINING_SOURCE,
            "meta": {"priority": 1, "source_file": TRAINING_SOURCE},
        },
        {
            "text": BUYU.read_text(encoding="utf-8"),
            "source": BUYU_SOURCE,
            "meta": {"priority": 1, "source_file": BUYU_SOURCE},
        },
        {
            "text": LEADERSHIP.read_text(encoding="utf-8"),
            "source": LEADERSHIP_SOURCE,
            "meta": {"priority": 1, "source_file": LEADERSHIP_SOURCE},
        },
    ]


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str,
    priority: int = 1,
) -> dict[str, Any]:
    return {
        "text": text,
        "source": source,
        "chunk_id": chunk_id,
        "meta": {
            "chunk_id": chunk_id,
            "source": source,
            "source_file": source,
            "priority": priority,
            "priority_bucket": "p1",
            "heading_path": [],
            "page": None,
            "page_start": None,
            "page_end": None,
            "rank_tag": None,
            "school_tag": None,
            "weapon_tag": None,
            "technique_tag": None,
        },
    }


def test_takamatsu_profile_is_grounded_and_deterministic():
    ans = try_answer_lineage_person("Who was Toshitsugu Takamatsu?", _passages())

    assert ans and ans.answer_type == "lineage_person"
    assert ans.det_path == "lineage/person"
    assert ans.facts["person_name"] == "Toshitsugu Takamatsu"
    assert "teacher" in ans.facts["role_or_relationship"].lower()
    assert "previous grandmaster" in ans.facts["summary"].lower()
    assert ans.facts["related_person"] == "Masaaki Hatsumi"
    assert {ref.source for ref in ans.source_refs} >= {TRAINING_SOURCE, BUYU_SOURCE}


def test_hatsumi_profile_is_grounded_and_renderable():
    ans = try_answer_lineage_person("Who is Masaaki Hatsumi?", _passages())

    assert ans and ans.answer_type == "lineage_person"
    assert ans.facts["person_name"] == "Masaaki Hatsumi"
    assert "leader of the bujinkan" in ans.facts["role_or_relationship"].lower()
    assert "34th togakure ryu soke" in ans.facts["summary"].lower()
    rendered = render_result(ans, style="standard", output_format="paragraph")
    assert "Masaaki Hatsumi" in rendered
    assert "Bujinkan" in rendered
    assert {ref.source for ref in ans.source_refs} >= {LEADERSHIP_SOURCE, TRAINING_SOURCE}


def test_hatsumi_teacher_query_returns_takamatsu():
    ans = try_answer_lineage_person("Who taught Hatsumi?", _passages())

    assert ans and ans.answer_type == "lineage_person"
    assert ans.facts["person_name"] == "Toshitsugu Takamatsu"
    assert ans.facts["related_person"] == "Masaaki Hatsumi"
    assert "teacher of masaaki hatsumi" in ans.facts["role_or_relationship"].lower()
    assert "inherited the nine schools" in ans.facts["summary"].lower()


def test_answer_with_rag_lineage_short_circuits_before_llm(monkeypatch):
    chunks = [
        _chunk("training", TRAINING.read_text(encoding="utf-8"), source=TRAINING_SOURCE),
        _chunk("buyu", BUYU.read_text(encoding="utf-8"), source=BUYU_SOURCE),
        _chunk("leadership", LEADERSHIP.read_text(encoding="utf-8"), source=LEADERSHIP_SOURCE),
    ]

    monkeypatch.setattr(app, "_load_index_and_meta", lambda: (None, chunks))
    monkeypatch.setattr(
        app,
        "retrieve",
        lambda question, k=None: (_ for _ in ()).throw(AssertionError("retrieve should not run")),
    )
    monkeypatch.setattr(
        app,
        "generate_grounded_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generate_grounded_answer should not run for deterministic lineage answers")
        ),
    )
    monkeypatch.setattr(app, "output_style", "Bullets", raising=False)
    monkeypatch.setattr(app, "tone_style", "Crisp", raising=False)
    monkeypatch.setattr(app, "TECH_DETAIL_MODE", "Standard", raising=False)

    answer, passages, raw_json, retrieval_debug = app.answer_with_rag("Who was Toshitsugu Takamatsu?")

    assert "Toshitsugu Takamatsu" in answer
    assert "teacher" in answer.lower()
    assert passages
    assert '"det_path": "lineage/person"' in raw_json
    assert TRAINING_SOURCE in raw_json
    assert BUYU_SOURCE in raw_json
    assert retrieval_debug["deterministic_short_circuit"] is True
    assert retrieval_debug["llm_routing"]["model_used"] == "deterministic_composer"
    assert retrieval_debug["llm_routing"]["route"] == "deterministic_local"
