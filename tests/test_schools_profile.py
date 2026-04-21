import pathlib

from extractors.schools import try_answer_school_profile

SCHOOLS = pathlib.Path("data") / "Schools of the Bujinkan Summaries.txt"


def _passages():
    return [
        {
            "text": SCHOOLS.read_text(encoding="utf-8"),
            "source": "Schools of the Bujinkan Summaries.txt",
            "meta": {"priority": 1},
        }
    ]


def test_togakure_profile_has_translation_type_focus():
    ans = try_answer_school_profile("tell me about togakure ryu", _passages())
    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Togakure Ryu"
    assert ans.facts["translation"]
    assert ans.facts["type"]
    assert ans.facts["focus"]


def test_gyokko_profile_mentions_kosshijutsu():
    ans = try_answer_school_profile("tell me about gyokko ryu", _passages())
    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Gyokko Ryu"
    assert "kosshi" in (ans.facts.get("focus") or "").lower() or "kosshi" in (ans.facts.get("notes") or "").lower()
