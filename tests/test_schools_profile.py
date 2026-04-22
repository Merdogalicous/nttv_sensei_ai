import pathlib

import pytest

from extractors.schools import AUTHORITATIVE_SCHOOL_PROFILE_SOURCE, try_answer_school_profile

SCHOOLS = pathlib.Path("data") / "Schools of the Bujinkan Summaries.txt"

SCHOOL_EXPECTATIONS = {
    "Togakure Ryu": {
        "translation": "Hidden Door School",
        "type": "Ninjutsu",
        "focus_term": "stealth",
    },
    "Gyokko Ryu": {
        "translation": "Jewel Tiger School",
        "type": "Samurai",
        "focus_term": "kosshijutsu",
    },
    "Koto Ryu": {
        "translation": "Tiger Knocking Down School",
        "type": "Samurai",
        "focus_term": "koppojutsu",
    },
    "Shinden Fudo Ryu": {
        "translation": "Immovable Heart School",
        "type": "Samurai",
        "focus_term": "natural body mechanics",
    },
    "Kukishinden Ryu": {
        "translation": "Nine Demon Gods School",
        "type": "Samurai",
        "focus_term": "battlefield",
    },
    "Takagi Yoshin Ryu": {
        "translation": "High Tree, Raised Heart School",
        "type": "Samurai",
        "focus_term": "close-quarters",
    },
    "Gikan Ryu": {
        "translation": "Truth, Loyalty, & Justice School",
        "type": "Samurai",
        "focus_term": "unusual angling",
    },
    "Gyokushin Ryu": {
        "translation": "Jeweled Heart School",
        "type": "Ninjutsu",
        "focus_term": "deception",
    },
    "Kumogakure Ryu": {
        "translation": "Hiding in the Clouds School",
        "type": "Ninjutsu",
        "focus_term": "concealment",
    },
}


def _passages(text: str | None = None, source: str = AUTHORITATIVE_SCHOOL_PROFILE_SOURCE):
    return [
        {
            "text": text if text is not None else SCHOOLS.read_text(encoding="utf-8"),
            "source": source,
            "meta": {"priority": 1},
        }
    ]


def _assert_authoritative_source(ans) -> None:
    assert ans.source_refs
    assert all(AUTHORITATIVE_SCHOOL_PROFILE_SOURCE in ref.source for ref in ans.source_refs)


@pytest.mark.parametrize("school_name,expected", SCHOOL_EXPECTATIONS.items())
def test_all_nine_school_profiles_are_anchored_and_authoritative(school_name, expected):
    ans = try_answer_school_profile(f"tell me about {school_name}", _passages())

    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == school_name
    assert ans.facts["translation"] == expected["translation"]
    assert ans.facts["type"] == expected["type"]
    assert expected["focus_term"] in (ans.facts.get("focus") or "").lower()
    assert ans.det_path == "schools/profile"
    _assert_authoritative_source(ans)

    other_translations = {
        data["translation"].lower()
        for other_school, data in SCHOOL_EXPECTATIONS.items()
        if other_school != school_name
    }
    assert (ans.facts.get("translation") or "").lower() not in other_translations


def test_togakure_profile_rejects_shinden_fudo_contamination():
    ans = try_answer_school_profile("tell me about togakure ryu", _passages())

    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Togakure Ryu"
    assert (ans.facts.get("type") or "").lower() == "ninjutsu"
    assert "immovable heart school" not in (ans.facts.get("translation") or "").lower()
    assert "natural body mechanics" not in (ans.facts.get("focus") or "").lower()


def test_non_authoritative_school_like_source_is_ignored():
    passages = [
        {
            "text": "\n".join(
                [
                    "SCHOOL: Togakure-ryu",
                    "TRANSLATION: Immovable Heart School",
                    "TYPE: Samurai",
                    "FOCUS: natural body mechanics, dakentaijutsu and jutaijutsu",
                ]
            ),
            "source": "Contaminated School Notes.txt",
            "meta": {"priority": 1},
        },
        {
            "text": SCHOOLS.read_text(encoding="utf-8"),
            "source": AUTHORITATIVE_SCHOOL_PROFILE_SOURCE,
            "meta": {"priority": 1},
        },
    ]

    ans = try_answer_school_profile("tell me about togakure ryu", passages)

    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Togakure Ryu"
    assert ans.facts["translation"] == "Hidden Door School"
    assert ans.facts["type"] == "Ninjutsu"
    _assert_authoritative_source(ans)


def test_mixed_profile_block_is_rejected():
    text = "\n".join(
        [
            "SCHOOL: Togakure-ryu",
            "ALIASES: Togakure-ryu, Togakure",
            "TRANSLATION: Hidden Door School",
            "TRANSLATION: Immovable Heart School",
            "TYPE: Ninjutsu",
            "FOCUS: stealth, infiltration",
        ]
    )

    ans = try_answer_school_profile("tell me about togakure ryu", _passages(text))

    assert ans is None


def test_plain_header_fallback_can_return_sparse_but_correct_profile():
    text = """
Togakure Ryu

Togakure Ryu was founded about nine hundred years ago.
""".strip()

    ans = try_answer_school_profile("tell me about togakure ryu", _passages(text))

    assert ans and ans.answer_type == "school_profile"
    assert ans.facts["school_name"] == "Togakure Ryu"
    assert ans.facts["translation"] == "Hidden Door School"
    assert ans.facts["type"] == "Ninjutsu"
    assert ans.facts.get("focus") is None
    assert ans.facts.get("weapons") is None
    _assert_authoritative_source(ans)
