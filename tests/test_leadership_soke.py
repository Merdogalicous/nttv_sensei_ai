import os

from extractors import try_extract_answer


def _passages():
    with open(os.path.join("data", "Bujinkan Leadership and Wisdom.txt"), "r", encoding="utf-8") as handle:
        txt = handle.read()
    return [{"text": txt, "source": "Bujinkan Leadership and Wisdom.txt", "meta": {"priority": 1}}]


def test_gyokko_ryu_soke():
    ans = try_extract_answer("who is the soke of gyokko ryu?", _passages())
    assert ans and ans.det_path == "leadership/soke"
    assert "gyokko" in ans.facts["school_name"].lower()
    assert ("ishizuka" in ans.facts["soke_name"].lower()) or ("nagato" in ans.facts["soke_name"].lower())
