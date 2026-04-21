import pathlib

from extractors import try_extract_answer

RANK = pathlib.Path("data") / "nttv rank requirements.txt"


def _passages():
    return [
        {
            "text": RANK.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        }
    ]


def test_requirements_scoped_single_rank():
    ans = try_extract_answer("What are the rank requirements for 3rd kyu?", _passages())
    assert ans and ans.answer_type == "rank_requirements"
    assert ans.facts["rank"] == "3rd Kyu"
    section_blob = " ".join(section["content"] for section in ans.facts["sections"]).lower()
    for other_rank in ["9th kyu", "8th kyu", "7th kyu", "6th kyu", "5th kyu", "4th kyu", "2nd kyu", "1st kyu", "shodan"]:
        assert other_rank not in section_blob
