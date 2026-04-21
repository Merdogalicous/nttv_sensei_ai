import pathlib

from extractors.glossary import try_answer_glossary

GLOSS = pathlib.Path("data") / "Glossary - edit.txt"
TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _gloss_passages():
    return [
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        }
    ]


def _gloss_and_tech_passages():
    return [
        {
            "text": TECH.read_text(encoding="utf-8"),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        },
        {
            "text": GLOSS.read_text(encoding="utf-8"),
            "source": "Glossary - edit.txt",
            "meta": {"priority": 1},
        },
    ]


def test_glossary_happo_geri_definition():
    ans = try_answer_glossary("What is Happo Geri?", _gloss_passages())
    assert ans and ans.answer_type == "glossary_term"
    assert ans.facts["term"].lower() == "happo geri"
    assert "eight" in ans.facts["definition"].lower()


def test_glossary_short_term_query():
    ans = try_answer_glossary("Happo Geri", _gloss_passages())
    assert ans and ans.answer_type == "glossary_term"
    assert ans.facts["term"].lower() == "happo geri"
    assert "eight" in ans.facts["definition"].lower()


def test_glossary_ignores_who_questions():
    assert not try_answer_glossary("Who is Hatsumi?", _gloss_passages())


def test_glossary_backs_off_for_technique_like_query():
    assert not try_answer_glossary("Describe Oni Kudaki", _gloss_and_tech_passages())


def test_glossary_backs_off_for_short_technique_name():
    assert not try_answer_glossary("Oni Kudaki", _gloss_and_tech_passages())
