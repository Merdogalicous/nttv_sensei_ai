import pathlib

from extractors import try_extract_answer

DATA = pathlib.Path("data") / "nttv rank requirements.txt"


def _passages_rank():
    return [
        {
            "text": DATA.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        }
    ]


def test_8th_kyu_kicks_rank_only():
    ans = try_extract_answer("What are the kicks for 8th kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_striking"
    kicks = [item.lower() for item in ans.facts["kicks"]]
    assert all("front kick" not in item and "zenpo geri" not in item and "mae geri" not in item for item in kicks)
    assert any(token in " ".join(kicks) for token in ["sokuho geri", "koho geri", "sakui geri", "happo geri"])


def test_8th_kyu_kicks_cumulative_from_need_to_know():
    ans = try_extract_answer("What are the kicks I need to know for 8th kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_striking"
    carry = " ".join(ans.facts["carryover_kicks"]).lower()
    assert "zenpo geri" in carry or "mae geri" in carry or "front kick" in carry


def test_4th_kyu_throws():
    ans = try_extract_answer("What are the throws for 4th kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_nage"
    low = " ".join(ans.facts["items"]).lower()
    assert any(token in low for token in ["osoto", "oosoto", "seoi", "nage"])


def test_3rd_kyu_chokes():
    ans = try_extract_answer("What are the chokes for 3rd kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_jime"
    assert "jime" in " ".join(ans.facts["items"]).lower()
