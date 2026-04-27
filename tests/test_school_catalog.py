from __future__ import annotations

import pathlib

import app
from extractors.leadership import LEADERSHIP_SOURCE
from extractors.schools import (
    AUTHORITATIVE_SCHOOL_PROFILE_SOURCE,
    CANONICAL_SCHOOL_ORDER,
    try_answer_school_catalog,
    try_answer_schools_list,
)
from tests.helpers import render_result


SCHOOLS = pathlib.Path("data") / "Schools of the Bujinkan Summaries.txt"
LEADERSHIP = pathlib.Path("data") / "Bujinkan Leadership and Wisdom.txt"

CATALOG_PROMPT = (
    "please describe the schools of the Bujinkan, including the English translation "
    "of the name, the style of the school, the current Soke, and a brief description"
)

EXPECTED_SOKES = {
    "Togakure Ryu": "Tsutsui Takumi",
    "Gyokushin Ryu": "Kan Jun'ichi",
    "Kumogakure Ryu": "Furuta Koji",
    "Gikan Ryu": "Sakasai Norio",
    "Gyokko Ryu": "Ishizuka Tetsuji",
    "Koto Ryu": "Noguchi Yukio",
    "Shinden Fudo Ryu": "Nagato Toshiro",
    "Kukishinden Ryu": "Yoshio Iwata",
    "Takagi Yoshin Ryu": "Sakasai Norio",
}


def _passages():
    return [
        {
            "text": SCHOOLS.read_text(encoding="utf-8"),
            "source": AUTHORITATIVE_SCHOOL_PROFILE_SOURCE,
            "meta": {"priority": 1, "source_file": AUTHORITATIVE_SCHOOL_PROFILE_SOURCE},
        },
        {
            "text": LEADERSHIP.read_text(encoding="utf-8"),
            "source": LEADERSHIP_SOURCE,
            "meta": {"priority": 1, "source_file": LEADERSHIP_SOURCE},
        },
    ]


def _chunk(text: str, source: str, *, priority: int = 1) -> dict:
    return {
        "text": text,
        "source": source,
        "meta": {
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


def test_school_catalog_full_prompt_returns_all_nine_grounded_entries():
    ans = try_answer_school_catalog(CATALOG_PROMPT, _passages())

    assert ans and ans.answer_type == "school_catalog"
    assert ans.det_path == "schools/catalog"
    assert [item["school_name"] for item in ans.facts["schools"]] == CANONICAL_SCHOOL_ORDER
    assert len(ans.facts["schools"]) == 9

    for item in ans.facts["schools"]:
        school_name = item["school_name"]
        assert item["translation"]
        assert item["type"]
        assert item["current_soke"] == EXPECTED_SOKES[school_name]
        assert item["brief_description"]

    source_names = {ref.source for ref in ans.source_refs}
    assert source_names == {AUTHORITATIVE_SCHOOL_PROFILE_SOURCE, LEADERSHIP_SOURCE}

    rendered = render_result(ans, style="standard", output_format="bullets")
    assert "Current soke" in rendered
    assert "Togakure Ryu" in rendered
    assert "Takagi Yoshin Ryu" in rendered


def test_school_catalog_overview_prompt_uses_catalog_path():
    ans = try_answer_school_catalog("give me an overview of the nine schools of the Bujinkan", _passages())

    assert ans and ans.answer_type == "school_catalog"
    assert [item["school_name"] for item in ans.facts["schools"]] == CANONICAL_SCHOOL_ORDER
    assert ans.facts["schools"][0]["translation"] == "Hidden Door School"
    assert ans.facts["schools"][0]["current_soke"] == "Tsutsui Takumi"
    assert ans.facts["schools"][-1]["translation"] == "High Tree, Raised Heart School"
    assert ans.facts["schools"][-1]["current_soke"] == "Sakasai Norio"


def test_simple_list_query_stays_on_simple_school_list_path():
    passages = _passages()

    assert try_answer_school_catalog("list the Bujinkan schools", passages) is None

    ans = try_answer_schools_list("list the Bujinkan schools", passages)

    assert ans and ans.answer_type == "school_list"
    assert ans.det_path == "schools/list"
    assert ans.facts["school_names"] == CANONICAL_SCHOOL_ORDER


def test_answer_with_rag_catalog_short_circuits_before_retrieval(monkeypatch):
    chunks = [
        _chunk(SCHOOLS.read_text(encoding="utf-8"), AUTHORITATIVE_SCHOOL_PROFILE_SOURCE),
        _chunk(LEADERSHIP.read_text(encoding="utf-8"), LEADERSHIP_SOURCE),
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
            AssertionError("generate_grounded_answer should not run for the school catalog path")
        ),
    )
    monkeypatch.setattr(app, "output_style", "Bullets", raising=False)
    monkeypatch.setattr(app, "tone_style", "Crisp", raising=False)
    monkeypatch.setattr(app, "TECH_DETAIL_MODE", "Standard", raising=False)

    answer, passages, raw_json, retrieval_debug = app.answer_with_rag(CATALOG_PROMPT)

    assert answer.strip()
    for school_name in CANONICAL_SCHOOL_ORDER:
        assert school_name in answer
    assert "Current soke" in answer
    assert AUTHORITATIVE_SCHOOL_PROFILE_SOURCE in raw_json
    assert LEADERSHIP_SOURCE in raw_json
    assert passages
    assert retrieval_debug["deterministic_short_circuit"] is True
    assert retrieval_debug["llm_routing"]["model_used"] == "deterministic_composer"
    assert retrieval_debug["llm_routing"]["route"] == "deterministic_local"
