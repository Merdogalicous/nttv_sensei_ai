import pathlib

from extractors.techniques import try_answer_technique

TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _passages():
    return [
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        }
    ]


def test_omote_gyaku_fields_present():
    ans = try_answer_technique("what is Omote Gyaku", _passages())
    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "omote gyaku"
    assert ans.facts["translation"]
    assert ans.facts["type"]
    assert ans.facts["rank_context"]
    assert "wrist" in ans.facts["definition"].lower() or "joint" in ans.facts["definition"].lower()
