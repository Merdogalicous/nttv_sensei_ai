from __future__ import annotations

from nttv_chatbot.composer import compose_deterministic_answer
from nttv_chatbot.deterministic import DeterministicResult


def render_result(
    result: DeterministicResult,
    *,
    style: str = "full",
    output_format: str = "paragraph",
) -> str:
    assert result and result.answered
    return compose_deterministic_answer(
        result,
        style=style,
        output_format=output_format,
        explanation_mode=True,
        tone="crisp",
    )
