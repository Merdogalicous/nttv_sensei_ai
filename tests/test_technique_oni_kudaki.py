import pathlib

import extractors.techniques as techniques

TECH = pathlib.Path("data") / "Technique Descriptions.md"


def _full_text() -> str:
    return TECH.read_text(encoding="utf-8")


def _passages(text: str | None = None):
    return [
        {
            "text": text if text is not None else _full_text(),
            "source": "Technique Descriptions.md",
            "meta": {"priority": 1},
        }
    ]


def _named_row(name: str) -> str:
    prefix = f"{name},"
    for line in _full_text().splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"Missing technique row for {name}")


def test_oni_kudaki_definition_not_truncated_by_commas():
    ans = techniques.try_answer_technique("Explain Oni Kudaki", _passages())
    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"
    assert ans.facts["translation"]
    assert ans.facts["type"]
    assert ans.facts["rank_context"]
    assert len(ans.facts["definition"]) > 60


def test_describe_oni_kudaki_returns_oni_not_koshi():
    ans = techniques.try_answer_technique("Describe Oni Kudaki", _passages())

    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"
    assert "koshi kudaki" not in ans.facts["definition"].lower()


def test_exact_alias_match_wins_over_close_name_similarity():
    partial_text = _named_row("Koshi Kudaki")

    ans = techniques.try_answer_technique("Describe Demon Crusher", _passages(partial_text))

    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"
    assert "demon crusher" in ans.facts["translation"].lower()


def test_extractor_loads_on_disk_technique_file_when_passages_are_partial():
    partial_text = _named_row("Koshi Kudaki")

    ans = techniques.try_answer_technique("Describe Oni Kudaki", _passages(partial_text))

    assert ans and ans.answer_type == "technique"
    assert ans.facts["technique_name"].lower() == "oni kudaki"


def test_direct_named_query_returns_none_instead_of_wrong_close_match(monkeypatch):
    partial_text = _named_row("Koshi Kudaki")
    monkeypatch.setattr(techniques, "_load_full_technique_text", lambda: "")

    ans = techniques.try_answer_technique("Describe Oni Kudaki", _passages(partial_text))

    assert ans is None
