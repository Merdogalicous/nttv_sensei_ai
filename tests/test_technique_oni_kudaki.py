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


def test_oni_kudaki_definition_not_truncated_by_commas():
    ans = try_answer_technique("Explain Oni Kudaki", _passages())
    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"
    assert ans.facts["translation"]
    assert ans.facts["type"]
    assert ans.facts["rank_context"]
    assert len(ans.facts["definition"]) > 60
