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


def test_8th_kyu_kihon_happo_kata():
    ans = try_extract_answer("Which Kihon Happo kata are required for 8th kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_kihon_kata"
    assert ans.facts["rank"] == "8th Kyu"
    assert "ichimonji no kata" in " ".join(ans.facts["items"]).lower()


def test_8th_kyu_sanshin_kata():
    ans = try_extract_answer("What Sanshin no Kata do I need for 8th kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_sanshin_kata"
    low = " ".join(ans.facts["items"]).lower()
    for token in ["chi no kata", "sui no kata", "ka no kata", "fu no kata", "ku no kata"]:
        assert token in low


def test_8th_kyu_kihon_happo_without_word_kata():
    ans = try_extract_answer("What Kihon Happo do I need to know for 8th kyu?", _passages_rank())
    assert ans and ans.answer_type == "rank_kihon_kata"
    assert "ichimonji no kata" in " ".join(ans.facts["items"]).lower()


def test_kata_queries_without_rank_use_generic_kihon_description():
    ans = try_extract_answer("Which Kihon Happo kata are there?", _passages_rank())
    assert ans and ans.answer_type == "kihon_happo"
    assert "kosshi kihon sanpo" in ans.facts["definition"].lower()
    assert "torite" in ans.facts["definition"].lower()
