import pathlib

from extractors import try_extract_answer

RANK = pathlib.Path("data") / "nttv rank requirements.txt"
TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _passages():
    return [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        },
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        },
    ]


def test_kihon_happo_answer_is_present_and_mentions_kata():
    ans = try_extract_answer("what is the kihon happo?", _passages())
    assert ans and ans.answer_type == "kihon_happo"
    assert "kihon happo" in ans.facts["topic"].lower()
    assert "ichimonji no kata" in [item.lower() for item in ans.facts["kosshi_items"]]
