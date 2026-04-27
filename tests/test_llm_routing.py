from __future__ import annotations

from dataclasses import replace

from nttv_chatbot.deterministic import SourceRef
from nttv_chatbot.llm_routing import (
    LLMRoutingSettings,
    build_grounded_prompt,
    filter_supporting_chunks,
    generate_grounded_answer,
    select_generation_route,
)


def _settings(**overrides) -> LLMRoutingSettings:
    base = LLMRoutingSettings(
        primary_model="openai/gpt-4o-mini",
        synthesis_model="google/gemma-3-27b-it",
        use_synthesis_model=True,
        synthesis_min_context_chunks=3,
        synthesis_for_explanation_mode=True,
        synthesis_for_deterministic_composer=False,
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        temperature=0.2,
        max_tokens=600,
    )
    return replace(base, **overrides)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "data/Technique Descriptions.md",
    page: int | None = None,
    heading_path: list[str] | None = None,
) -> dict:
    meta = {
        "chunk_id": chunk_id,
        "source": source,
        "source_file": source,
        "page": page,
        "page_start": page,
        "page_end": page,
        "heading_path": heading_path or [],
        "priority_bucket": "p2",
    }
    return {
        "text": text,
        "source": source,
        "chunk_id": chunk_id,
        "meta": meta,
    }


def test_route_uses_primary_for_direct_single_chunk_question():
    decision = select_generation_route(
        "What is Omote Gyaku?",
        [_chunk("c1", "Omote Gyaku is an outward wrist lock.")],
        settings=_settings(),
    )

    assert decision.route == "primary"
    assert decision.model_requested == "openai/gpt-4o-mini"
    assert decision.reason_codes == ["direct_question"]


def test_route_uses_synthesis_for_explanation_mode():
    decision = select_generation_route(
        "Explain the difference between Omote Gyaku and Ura Gyaku.",
        [_chunk("c1", "Omote Gyaku details."), _chunk("c2", "Ura Gyaku details.")],
        settings=_settings(),
    )

    assert decision.route == "synthesis"
    assert decision.model_requested == "google/gemma-3-27b-it"
    assert "explanation_mode" in decision.reason_codes


def test_route_uses_synthesis_for_multi_chunk_context():
    chunks = [_chunk(f"c{i}", f"Chunk {i}") for i in range(4)]
    decision = select_generation_route(
        "Summarize the training progression.",
        chunks,
        settings=_settings(),
    )

    assert decision.route == "synthesis"
    assert "multi_chunk_context" in decision.reason_codes


def test_deterministic_route_stays_local_by_default():
    decision = select_generation_route(
        "Explain Omote Gyaku.",
        [_chunk("c1", "Omote Gyaku details.")],
        fact_count=4,
        deterministic_mode=True,
        settings=_settings(synthesis_for_deterministic_composer=False),
    )

    assert decision.route == "deterministic_local"
    assert decision.use_model is False
    assert "deterministic_local_default" in decision.reason_codes


def test_filter_supporting_chunks_prefers_matching_source_refs():
    chunks = [
        _chunk("rank-1", "Rank requirements", source="data/nttv rank requirements.txt"),
        _chunk("tech-1", "Technique details", source="data/Technique Descriptions.md", page=4),
        _chunk("tech-2", "More technique details", source="data/Technique Descriptions.md", page=5),
    ]
    refs = [SourceRef(source="data/Technique Descriptions.md", page_start=4, page_end=4)]

    selected = filter_supporting_chunks(chunks, refs, limit=4)

    assert [chunk["chunk_id"] for chunk in selected] == ["tech-1"]


def test_build_grounded_prompt_includes_grounding_rules_sources_and_budget():
    bundle = build_grounded_prompt(
        "Explain Omote Gyaku.",
        [
            _chunk(
                "tech-1",
                "Omote Gyaku applies outward wrist rotation and balance disruption.",
                page=4,
                heading_path=["Locks", "Omote Gyaku"],
            )
        ],
        facts={
            "technique_name": "Omote Gyaku",
            "definition": "An outward wrist manipulation.",
        },
        source_refs=[SourceRef(source="data/Technique Descriptions.md", page_start=4, heading_path=["Locks", "Omote Gyaku"])],
        max_context_chars=450,
    )

    assert "Answer only from the deterministic facts and retrieved context" in bundle.system_prompt
    assert "Do not add unsupported martial-arts lore" in bundle.system_prompt
    assert "cite it inline with [1], [2]" in bundle.user_prompt
    assert "explicitly say the available material is incomplete" in bundle.user_prompt
    assert '"technique_name": "Omote Gyaku"' in bundle.user_prompt
    assert "[F1] Technique Descriptions.md (p. 4) | Locks > Omote Gyaku" in bundle.user_prompt
    assert "[1] Source: Technique Descriptions.md | Page: 4 | Heading: Locks > Omote Gyaku | Priority: p2" in bundle.user_prompt
    assert bundle.context_char_count <= 450


def test_generate_grounded_answer_falls_back_from_synthesis_to_primary():
    calls: list[str] = []

    def fake_completion(model_name: str, messages: list[dict[str, str]], settings: LLMRoutingSettings) -> tuple[str, str]:
        calls.append(model_name)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        if model_name == settings.synthesis_model:
            raise RuntimeError("synthesis timeout")
        return "Grounded answer with citations [1].", '{"id":"primary-success"}'

    result = generate_grounded_answer(
        "Explain the difference between Omote Gyaku and Ura Gyaku.",
        [_chunk("c1", "Omote Gyaku details."), _chunk("c2", "Ura Gyaku details."), _chunk("c3", "Comparison notes.")],
        settings=_settings(),
        completion_callable=fake_completion,
    )

    assert result.text == "Grounded answer with citations [1]."
    assert calls == ["google/gemma-3-27b-it", "openai/gpt-4o-mini"]
    assert result.debug["fallback_used"] is True
    assert result.debug["model_used"] == "openai/gpt-4o-mini"
    assert result.debug["attempted_models"] == ["google/gemma-3-27b-it", "openai/gpt-4o-mini"]
    assert "synthesis timeout" in result.debug["fallback_reason"]
