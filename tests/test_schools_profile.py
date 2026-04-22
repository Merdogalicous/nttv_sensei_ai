import pathlib

from extractors.schools import try_answer_school_profile

SCHOOLS = pathlib.Path("data") / "Schools of the Bujinkan Summaries.txt"


def _passages(text: str | None = None):
    return [
        {
            "text": text if text is not None else SCHOOLS.read_text(encoding="utf-8"),
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
    assert (ans.facts.get("type") or "").lower() == "ninjutsu"
    assert "immovable heart school" not in (ans.facts.get("translation") or "").lower()
    assert "natural body mechanics" not in (ans.facts.get("focus") or "").lower()


def test_gyokko_profile_mentions_kosshijutsu():
    ans = try_answer_school_profile("tell me about gyokko ryu", _passages())
    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Gyokko Ryu"
    assert "kosshi" in (ans.facts.get("focus") or "").lower() or "kosshi" in (ans.facts.get("notes") or "").lower()


def test_togakure_plain_header_block_stays_anchored():
    text = """
Togakure Ryu

Togakure Ryu (The hidden entrance school), is a Ninjutsu Techniques school.
The school contained Yon-po Hiden (Four Secrets): Senban Shuriken, Shuko, Ashiko, Shinodake, and Kyoketsu Shoge.
""".strip()

    ans = try_answer_school_profile("tell me about togakure ryu", _passages(text))

    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Togakure Ryu"
    assert (ans.facts.get("type") or "").lower() == "ninjutsu"
    assert "immovable heart school" not in (ans.facts.get("translation") or "").lower()
    assert "samurai" not in (ans.facts.get("type") or "").lower()


def test_school_profile_fallback_rejects_cross_school_freeblock():
    text = """
Related schools: Togakure Ryu and Shinden Fudo Ryu
TRANSLATION: Immovable Heart School
TYPE: Samurai
FOCUS: natural body mechanics, dakentaijutsu (striking) and jutaijutsu (grappling), kamae from nature
""".strip()

    ans = try_answer_school_profile("tell me about togakure ryu", _passages(text))

    assert ans is None
