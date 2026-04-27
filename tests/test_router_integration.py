from pathlib import Path

from extractors import try_extract_answer
from tests.helpers import render_result

DATA = Path("data")

RANK = DATA / "nttv rank requirements.txt"
WEAPONS = DATA / "NTTV Weapons Reference.txt"
GLOSS = DATA / "Glossary - edit.txt"
TECH = DATA / "Technique Descriptions.md"
SANSHIN = DATA / "nttv training reference.txt"


def _passages_rank_and_gloss():
    return [
        {"text": RANK.read_text(encoding="utf-8"), "source": "nttv rank requirements.txt", "meta": {"priority": 3}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]


def _passages_weapons_and_gloss():
    return [
        {"text": WEAPONS.read_text(encoding="utf-8"), "source": "NTTV Weapons Reference.txt", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]


def _passages_tech_and_gloss():
    return [
        {"text": TECH.read_text(encoding="utf-8"), "source": "Technique Descriptions.md", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]


def _passages_sanshin_and_gloss():
    return [
        {"text": SANSHIN.read_text(encoding="utf-8"), "source": "nttv training reference.txt", "meta": {"priority": 2}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]


def test_router_prefers_rank_over_glossary_for_kicks():
    ans = try_extract_answer("What kicks do I need to know for 8th kyu?", _passages_rank_and_gloss())
    assert ans and ans.det_path == "rank/striking"
    rendered = render_result(ans, style="full").lower()
    assert "8th kyu" in rendered
    assert "happo geri" in rendered or "sokuho geri" in rendered


def test_router_prefers_weapon_profile_over_glossary():
    ans = try_extract_answer("What is a hanbo weapon?", _passages_weapons_and_gloss())
    assert ans and ans.det_path == "weapons/profile"
    assert ans.facts["weapon_name"].lower() == "hanbo"
    assert "short staff" in ans.facts["weapon_type"].lower()


def test_router_prefers_technique_over_glossary():
    ans = try_extract_answer("Describe Oni Kudaki", _passages_tech_and_gloss())
    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"
    assert ans.facts["translation"]
    assert ans.facts["definition"]


def test_router_technique_fallback_uses_on_disk_file_for_partial_passages():
    koshi_row = next(
        line for line in TECH.read_text(encoding="utf-8").splitlines() if line.startswith("Koshi Kudaki,")
    )
    passages = [
        {"text": koshi_row, "source": "Technique Descriptions.md", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]

    ans = try_extract_answer("Describe Oni Kudaki", passages)

    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"


def test_router_glossary_fallback_when_no_specific_extractor():
    passages = [
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}}
    ]
    ans = try_extract_answer("What is Happo Geri?", passages)
    assert ans and ans.answer_type == "glossary_term"
    assert ans.facts["term"].lower() == "happo geri"
    assert "eight" in ans.facts["definition"].lower()


def test_router_handles_many_noisy_passages_quickly():
    noisy = [
        {"text": "lorem ipsum dolor sit amet " * 5, "source": f"noise_{i}.txt", "meta": {"priority": 5}}
        for i in range(200)
    ]
    passages = noisy + [
        {"text": RANK.read_text(encoding="utf-8"), "source": "nttv rank requirements.txt", "meta": {"priority": 1}}
    ]
    ans = try_extract_answer("What kicks do I need to know for 8th kyu?", passages)
    assert ans and ans.det_path == "rank/striking"


def test_router_handles_many_noisy_passages():
    noisy = [
        {"text": "lorem ipsum dolor sit amet " * 5, "source": f"noise_{i}.txt", "meta": {"priority": 5}}
        for i in range(600)
    ]
    passages = noisy + [
        {"text": RANK.read_text(encoding="utf-8"), "source": "nttv rank requirements.txt", "meta": {"priority": 3}}
    ]
    ans = try_extract_answer("What kicks do I need to know for 8th kyu?", passages)
    assert ans and ans.det_path == "rank/striking"
    rendered = render_result(ans, style="full").lower()
    assert "sokuho geri" in rendered
    assert "koho geri" in rendered


def test_router_musha_dori_meaning_phrase():
    passages = [
        {"text": TECH.read_text(encoding="utf-8"), "source": "Technique Descriptions.md", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]
    assert not try_extract_answer("What does Musha Dori mean?", passages)


def test_router_explain_musha_dori_phrase():
    passages = [
        {"text": TECH.read_text(encoding="utf-8"), "source": "Technique Descriptions.md", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]
    ans = try_extract_answer("Explain Musha Dori", passages)
    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "musha dori"
    assert ans.facts["translation"]
    assert ans.facts["definition"]


def test_router_tell_me_about_oni_kudaki():
    passages = [
        {"text": TECH.read_text(encoding="utf-8"), "source": "Technique Descriptions.md", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]
    assert not try_extract_answer("Tell me about Oni Kudaki", passages)


def test_router_list_sanshin_uses_sanshin_extractor():
    ans = try_extract_answer("List Sanshin", _passages_sanshin_and_gloss())
    assert ans and ans.answer_type == "sanshin_list"
    low = " ".join(ans.facts["items"]).lower()
    for token in ["chi no kata", "sui no kata", "ka no kata", "fu no kata", "ku no kata"]:
        assert token in low


def test_router_diff_omote_vs_ura_gyaku():
    passages = [
        {"text": TECH.read_text(encoding="utf-8"), "source": "Technique Descriptions.md", "meta": {"priority": 1}},
        {"text": GLOSS.read_text(encoding="utf-8"), "source": "Glossary - edit.txt", "meta": {"priority": 1}},
    ]
    ans = try_extract_answer("What's the difference between Omote Gyaku and Ura Gyaku?", passages)
    assert ans and ans.answer_type == "technique_diff"
    rendered = render_result(ans, style="full").lower()
    assert "omote gyaku" in rendered
    assert "ura gyaku" in rendered
    assert "translation" in rendered
