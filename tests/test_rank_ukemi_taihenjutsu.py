import pathlib

from extractors import try_extract_answer

RANK_FILE = pathlib.Path("data") / "nttv rank requirements.txt"


def _passages_rank_only():
    return [
        {
            "text": RANK_FILE.read_text(encoding="utf-8"),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        }
    ]


def test_9th_kyu_ukemi_list():
    ans = try_extract_answer("What ukemi do I need to know for 9th kyu?", _passages_rank_only())
    assert ans and ans.answer_type == "rank_ukemi"
    low = " ".join(ans.facts["items"]).lower()
    assert ans.facts["rank"] == "9th Kyu"
    assert "zenpo ukemi" in low
    assert "koho ukemi" in low
    assert "yoko ukemi" in low


def test_9th_kyu_ukemi_rolls_wording():
    ans = try_extract_answer("What rolls and breakfalls are required for 9th kyu?", _passages_rank_only())
    assert ans and ans.answer_type == "rank_ukemi"
    low = " ".join(ans.facts["items"]).lower()
    assert "zenpo ukemi" in low
    assert "koho ukemi" in low


def test_9th_kyu_taihenjutsu():
    ans = try_extract_answer("What taihenjutsu do I need to know for 9th kyu?", _passages_rank_only())
    assert ans and ans.answer_type == "rank_taihenjutsu"
    assert "tai sabaki" in " ".join(ans.facts["items"]).lower()


def test_ukemi_without_rank_does_not_use_rank_specific():
    assert not try_extract_answer("What ukemi do I need to know?", _passages_rank_only())
