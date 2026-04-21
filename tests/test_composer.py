from __future__ import annotations

from nttv_chatbot.composer import compose_deterministic_answer
from nttv_chatbot.deterministic import DeterministicResult, SourceRef


def _technique_result() -> DeterministicResult:
    return DeterministicResult(
        answered=True,
        det_path="technique/core",
        answer_type="technique",
        facts={
            "technique_name": "Omote Gyaku",
            "japanese": "Omote Gyaku",
            "translation": "Outside Reverse",
            "type": "Wrist Lock",
            "rank_context": "8th Kyu",
            "primary_focus": "wrist control",
            "safety": "Apply with control.",
            "partner_required": True,
            "solo": False,
            "tags": ["lock", "kihon"],
            "definition": "A reversal that turns the wrist outward to break balance.",
        },
        source_refs=[SourceRef(source="Technique Descriptions.md")],
        confidence=0.98,
        display_hints={"explain": True},
        followup_suggestions=[],
    )


def test_composer_preserves_facts_faithfully():
    rendered = compose_deterministic_answer(_technique_result(), style="standard", output_format="paragraph")
    low = rendered.lower()
    assert "omote gyaku" in low
    assert "outside reverse" in low
    assert "wrist lock" in low
    assert "8th kyu" in low
    assert "wrist control" in low


def test_composer_does_not_introduce_missing_fields():
    glossary_result = DeterministicResult(
        answered=True,
        det_path="glossary/term",
        answer_type="glossary_term",
        facts={"term": "Happo Geri", "definition": "Kicking in eight directions."},
        source_refs=[SourceRef(source="Glossary - edit.txt")],
        confidence=0.9,
        display_hints={"explain": False},
        followup_suggestions=[],
    )
    rendered = compose_deterministic_answer(glossary_result, style="full", output_format="paragraph")
    assert "happo geri" in rendered.lower()
    assert "eight directions" in rendered.lower()
    assert "rank intro" not in rendered.lower()
    assert "partner required" not in rendered.lower()


def test_composer_brief_standard_full_styles_differ():
    result = _technique_result()
    brief = compose_deterministic_answer(result, style="brief", output_format="bullets")
    standard = compose_deterministic_answer(result, style="standard", output_format="bullets")
    full = compose_deterministic_answer(result, style="full", output_format="bullets")

    assert brief != standard != full
    assert "Safety" not in brief
    assert "Rank intro" in standard
    assert "Safety" in full
    assert "Partner required" in full
