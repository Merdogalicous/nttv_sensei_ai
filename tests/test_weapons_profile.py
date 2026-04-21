import pathlib

from extractors.weapons import try_answer_katana_parts, try_answer_weapon_profile

WEAP = pathlib.Path("data") / "NTTV Weapons Reference.txt"


def _passages():
    return [
        {
            "text": WEAP.read_text(encoding="utf-8"),
            "source": "NTTV Weapons Reference.txt",
            "meta": {"priority": 1},
        }
    ]


def test_hanbo_profile_uses_structured_fields():
    ans = try_answer_weapon_profile("What is the hanbo weapon?", _passages())
    assert ans and ans.answer_type == "weapon_profile"
    assert ans.facts["weapon_name"].lower() == "hanbo"
    assert "short staff" in ans.facts["weapon_type"].lower()
    assert any("strike" in item.lower() or "thrust" in item.lower() for item in ans.facts["core_actions"])


def test_kusari_fundo_profile_mentions_chain_and_core_actions():
    ans = try_answer_weapon_profile("Explain the kusari fundo weapon.", _passages())
    assert ans and ans.answer_type == "weapon_profile"
    assert "kusari" in ans.facts["weapon_name"].lower()
    assert "chain" in ans.facts["weapon_type"].lower()
    assert ans.facts["core_actions"]


def test_hanbo_weapon_profile_basic_fields():
    ans = try_answer_weapon_profile("What is a hanbo weapon?", _passages())
    assert ans and ans.answer_type == "weapon_profile"
    assert ans.facts["weapon_type"]
    assert ans.facts["core_actions"]
    assert ans.facts["rank_context"]


def test_katana_parts_return_structured_part_list():
    ans = try_answer_katana_parts("What are the parts of the katana?", _passages())
    assert ans and ans.answer_type == "weapon_parts"
    parts = ans.facts["parts"]
    assert any(part["term"] == "Tsuka" for part in parts)
    assert any(part["term"] == "Kissaki" for part in parts)
