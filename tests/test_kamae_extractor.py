from __future__ import annotations

import pathlib
from typing import Any

import app
from extractors.kamae import try_answer_kamae


DATA = pathlib.Path("data")
RANK = DATA / "nttv rank requirements.txt"
TECH = DATA / "Technique Descriptions.md"
WEAP = DATA / "NTTV Weapons Reference.txt"


def _passages() -> list[dict[str, Any]]:
    return [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 1, "source_file": "nttv rank requirements.txt"},
        },
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1, "source_file": "Technique Descriptions.md"},
        },
        {
            "text": WEAP.read_text(encoding="utf-8"),
            "source": "NTTV Weapons Reference.txt",
            "meta": {"priority": 1, "source_file": "NTTV Weapons Reference.txt"},
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


def test_rank_kamae_question_returns_deterministic_result():
    ans = try_answer_kamae("what are the basic kamae in 9th kyu?", _passages())

    assert ans and ans.answer_type == "rank_kamae"
    assert ans.det_path == "rank/kamae"
    assert ans.facts["rank"] == "9th Kyu"
    assert "Shizen no Kamae" in ans.facts["items"]
    assert "Hicho no Kamae" in ans.facts["items"]
    assert any("nttv rank requirements.txt" in ref.source.lower() for ref in ans.source_refs)


def test_rank_kamae_alt_phrase_uses_same_rank_block():
    ans = try_answer_kamae("what are the kamae for 9th kyu?", _passages())

    assert ans and ans.answer_type == "rank_kamae"
    assert ans.facts["rank"] == "9th Kyu"
    assert "Ichimonji no Kamae" in ans.facts["items"]
    assert "Hanza no Kamae" in ans.facts["items"]


def test_specific_kamae_definition_uses_technique_shape():
    ans = try_answer_kamae("what is Hicho no Kamae?", _passages())

    assert ans and ans.answer_type == "technique"
    assert ans.det_path == "kamae/specific"
    assert ans.facts["technique_name"] == "Hicho no Kamae"
    assert ans.facts["translation"] == "Flying Bird Posture"
    assert ans.facts["type"] == "Kamae"
    assert ans.facts["rank_context"] == "9th Kyu"
    assert "balance" in (ans.facts["definition"] or "").lower()
    assert "hoko no kamae" not in (ans.facts["definition"] or "").lower()
    assert "kosei no kamae" not in (ans.facts["definition"] or "").lower()
    assert any("technique descriptions.md" in ref.source.lower() for ref in ans.source_refs)


def test_jumonji_description_stays_bounded_to_its_own_record():
    ans = try_answer_kamae("describe Jumonji no Kamae", _passages())

    assert ans and ans.answer_type == "technique"
    assert ans.det_path == "kamae/specific"
    assert ans.facts["technique_name"] == "Jumonji no Kamae"
    assert ans.facts["translation"] == "Cross Posture"
    definition = (ans.facts["definition"] or "").lower()
    assert "crossed in front" in definition
    assert "hicho no kamae" not in definition
    assert "hoko no kamae" not in definition
    assert "kosei no kamae" not in definition
    assert any("technique descriptions.md" in ref.source.lower() for ref in ans.source_refs)


def test_weapon_kamae_question_returns_weapon_profile_shape():
    ans = try_answer_kamae("what kamae do we use with the hanbo?", _passages())

    assert ans and ans.answer_type == "weapon_profile"
    assert ans.det_path == "weapons/kamae"
    assert ans.facts["weapon_name"] == "Hanbo"
    assert "Munen Muso" in ans.facts["kamae"]
    assert "Kata Yaburi" in ans.facts["kamae"]
    assert any("nttv weapons reference.txt" in ref.source.lower() for ref in ans.source_refs)


def test_answer_with_rag_rank_kamae_short_circuits_before_llm(monkeypatch):
    chunks = [
        _chunk(
            "rank",
            RANK.read_text(encoding="utf-8"),
            source="nttv rank requirements.txt",
        )
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
            AssertionError("generate_grounded_answer should not run for deterministic kamae answers")
        ),
    )
    monkeypatch.setattr(app, "output_style", "Bullets", raising=False)
    monkeypatch.setattr(app, "tone_style", "Crisp", raising=False)
    monkeypatch.setattr(app, "TECH_DETAIL_MODE", "Standard", raising=False)

    answer, passages, raw_json, retrieval_debug = app.answer_with_rag("what are the basic kamae in 9th kyu?")

    assert "9th Kyu kamae" in answer
    assert "Hicho no Kamae" in answer
    assert passages
    assert '"det_path": "rank/kamae"' in raw_json
    assert retrieval_debug["deterministic_short_circuit"] is True
    assert retrieval_debug["llm_routing"]["model_used"] == "deterministic_composer"
    assert retrieval_debug["llm_routing"]["route"] == "deterministic_local"


def test_answer_with_rag_specific_kamae_short_circuits_before_llm(monkeypatch):
    chunks = [
        _chunk(
            "technique",
            TECH.read_text(encoding="utf-8"),
            source="Technique Descriptions.md",
        )
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
            AssertionError("generate_grounded_answer should not run for deterministic kamae answers")
        ),
    )
    monkeypatch.setattr(app, "output_style", "Bullets", raising=False)
    monkeypatch.setattr(app, "tone_style", "Crisp", raising=False)
    monkeypatch.setattr(app, "TECH_DETAIL_MODE", "Standard", raising=False)

    answer, passages, raw_json, retrieval_debug = app.answer_with_rag("describe Jumonji no Kamae")

    assert "Jumonji no Kamae" in answer
    assert "Hicho no Kamae" not in answer
    assert passages
    assert '"det_path": "kamae/specific"' in raw_json
    assert retrieval_debug["deterministic_short_circuit"] is True
    assert retrieval_debug["llm_routing"]["model_used"] == "deterministic_composer"
    assert retrieval_debug["llm_routing"]["route"] == "deterministic_local"
