import pathlib

from extractors.technique_diff import try_answer_technique_diff

TECH_FILE = pathlib.Path("data") / "Technique Descriptions.md"


def _passages_tech_only():
    return [
        {
            "text": TECH_FILE.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        }
    ]


def test_diff_omote_vs_ura_gyaku_difference_between():
    ans = try_answer_technique_diff("What is the difference between Omote Gyaku and Ura Gyaku?", _passages_tech_only())
    assert ans and ans.answer_type == "technique_diff"
    assert ans.facts["left"]["technique_name"].lower() == "omote gyaku"
    assert ans.facts["right"]["technique_name"].lower() == "ura gyaku"
    assert ans.facts["left"]["translation"]
    assert ans.facts["right"]["definition"]


def test_diff_omote_vs_ura_gyaku_vs_syntax():
    ans = try_answer_technique_diff("Omote Gyaku vs Ura Gyaku", _passages_tech_only())
    assert ans and ans.answer_type == "technique_diff"
    assert ans.facts["left"]["technique_name"].lower() == "omote gyaku"
    assert ans.facts["right"]["technique_name"].lower() == "ura gyaku"


def test_non_diff_question_returns_none():
    assert not try_answer_technique_diff("Describe Omote Gyaku", _passages_tech_only())
