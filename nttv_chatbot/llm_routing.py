from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

from nttv_chatbot.deterministic import SourceRef


PRIMARY_CONTEXT_CHAR_BUDGET = 6000
SYNTHESIS_CONTEXT_CHAR_BUDGET = 9000
MAX_CONTEXT_CHUNK_TEXT_CHARS = 1600
MIN_CONTEXT_CHUNK_TEXT_CHARS = 180
DEFAULT_TIMEOUT_SECONDS = 30

_EXPLANATION_PATTERNS = (
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bdescribe\b", re.IGNORECASE),
    re.compile(r"\btell me about\b", re.IGNORECASE),
    re.compile(r"\bwalk me through\b", re.IGNORECASE),
    re.compile(r"\bhelp me understand\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) the difference\b", re.IGNORECASE),
    re.compile(r"\bdifference between\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bhow\b", re.IGNORECASE),
    re.compile(r"\bwhy\b", re.IGNORECASE),
)

_INTERPRETIVE_PATTERNS = (
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bcontext\b", re.IGNORECASE),
    re.compile(r"\bhistory\b", re.IGNORECASE),
    re.compile(r"\bimportance\b", re.IGNORECASE),
    re.compile(r"\bwhat should i know\b", re.IGNORECASE),
    re.compile(r"\bwhat should i understand\b", re.IGNORECASE),
    re.compile(r"\bhow does .* work\b", re.IGNORECASE),
)


CompletionCallable = Callable[[str, list[dict[str, str]], "LLMRoutingSettings"], tuple[str, str]]


@dataclass(frozen=True)
class LLMRoutingSettings:
    primary_model: str
    synthesis_model: str
    use_synthesis_model: bool
    synthesis_min_context_chunks: int
    synthesis_for_explanation_mode: bool
    synthesis_for_deterministic_composer: bool
    api_base: str
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    primary_context_char_budget: int = PRIMARY_CONTEXT_CHAR_BUDGET
    synthesis_context_char_budget: int = SYNTHESIS_CONTEXT_CHAR_BUDGET
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "LLMRoutingSettings":
        return cls(
            primary_model=(os.getenv("MODEL") or "gpt-4o-mini").strip(),
            synthesis_model=(os.getenv("SYNTHESIS_MODEL") or "").strip(),
            use_synthesis_model=_parse_bool_env("USE_SYNTHESIS_MODEL", False),
            synthesis_min_context_chunks=max(1, _int_env("SYNTHESIS_MIN_CONTEXT_CHUNKS", 3)),
            synthesis_for_explanation_mode=_parse_bool_env("SYNTHESIS_FOR_EXPLANATION_MODE", True),
            synthesis_for_deterministic_composer=_parse_bool_env(
                "SYNTHESIS_FOR_DETERMINISTIC_COMPOSER",
                False,
            ),
            api_base=(
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENROUTER_API_BASE")
                or os.getenv("LM_STUDIO_BASE_URL")
                or "http://localhost:1234/v1"
            ).strip(),
            api_key=(
                (os.getenv("OPENAI_API_KEY") or "").strip()
                or (os.getenv("OPENROUTER_API_KEY") or "").strip()
                or None
            ),
            temperature=_float_env("TEMPERATURE", 0.2),
            max_tokens=max(1, _int_env("MAX_TOKENS", 600)),
        )

    @property
    def synthesis_available(self) -> bool:
        return self.use_synthesis_model and bool(self.synthesis_model)

    @property
    def chat_completions_url(self) -> str:
        base = (self.api_base or "http://localhost:1234/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    use_model: bool
    model_requested: Optional[str]
    reason: str
    reason_codes: list[str]
    deterministic_mode: bool
    explanation_mode: bool
    interpretive_question: bool
    input_chunk_count: int
    input_fact_count: int

    def to_debug_payload(
        self,
        *,
        model_used: Optional[str] = None,
        selected_chunk_count: int = 0,
        selected_fact_count: Optional[int] = None,
        context_char_count: int = 0,
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
        attempted_models: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        selected_fact_total = self.input_fact_count if selected_fact_count is None else selected_fact_count
        return {
            "route": self.route,
            "model_requested": self.model_requested,
            "model_used": model_used,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "deterministic_mode": self.deterministic_mode,
            "explanation_mode": self.explanation_mode,
            "interpretive_question": self.interpretive_question,
            "input_chunk_count": self.input_chunk_count,
            "selected_chunk_count": selected_chunk_count,
            "input_fact_count": self.input_fact_count,
            "selected_fact_count": selected_fact_total,
            "context_char_count": context_char_count,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "attempted_models": list(attempted_models or []),
        }


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    selected_chunks: list[dict[str, Any]]
    selected_fact_count: int
    context_char_count: int

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    raw_json: str
    debug: dict[str, Any]
    prompt_bundle: PromptBundle


def empty_route_debug(reason: str) -> dict[str, Any]:
    return {
        "route": "none",
        "model_requested": None,
        "model_used": None,
        "reason": reason,
        "reason_codes": [],
        "deterministic_mode": False,
        "explanation_mode": False,
        "interpretive_question": False,
        "input_chunk_count": 0,
        "selected_chunk_count": 0,
        "input_fact_count": 0,
        "selected_fact_count": 0,
        "context_char_count": 0,
        "fallback_used": False,
        "fallback_reason": None,
        "attempted_models": [],
    }


def select_generation_route(
    question: str,
    context_chunks: list[dict[str, Any]],
    *,
    fact_count: int = 0,
    deterministic_mode: bool = False,
    settings: Optional[LLMRoutingSettings] = None,
) -> RouteDecision:
    settings = settings or LLMRoutingSettings.from_env()
    chunk_count = len(context_chunks)
    explanation_mode = is_explanation_question(question)
    interpretive_question = is_interpretive_question(question)

    if deterministic_mode:
        if not settings.synthesis_for_deterministic_composer:
            return RouteDecision(
                route="deterministic_local",
                use_model=False,
                model_requested=None,
                reason="Deterministic composer stayed local because synthesis for deterministic answers is disabled.",
                reason_codes=["deterministic_local_default"],
                deterministic_mode=True,
                explanation_mode=explanation_mode,
                interpretive_question=interpretive_question,
                input_chunk_count=chunk_count,
                input_fact_count=fact_count,
            )
        if not settings.synthesis_available:
            return RouteDecision(
                route="deterministic_local",
                use_model=False,
                model_requested=None,
                reason="Deterministic composer stayed local because no synthesis model is configured.",
                reason_codes=["deterministic_local_default", "synthesis_unavailable"],
                deterministic_mode=True,
                explanation_mode=explanation_mode,
                interpretive_question=interpretive_question,
                input_chunk_count=chunk_count,
                input_fact_count=fact_count,
            )
        if explanation_mode or interpretive_question:
            reason_codes = ["deterministic_composer_enabled"]
            if explanation_mode:
                reason_codes.append("explanation_mode")
            if interpretive_question:
                reason_codes.append("interpretive_question")
            return RouteDecision(
                route="synthesis",
                use_model=True,
                model_requested=settings.synthesis_model,
                reason=_describe_route_reason(reason_codes, chunk_count),
                reason_codes=reason_codes,
                deterministic_mode=True,
                explanation_mode=explanation_mode,
                interpretive_question=interpretive_question,
                input_chunk_count=chunk_count,
                input_fact_count=fact_count,
            )
        return RouteDecision(
            route="deterministic_local",
            use_model=False,
            model_requested=None,
            reason="Deterministic composer stayed local because the question is direct and already covered by extracted facts.",
            reason_codes=["deterministic_local_default", "direct_fact_question"],
            deterministic_mode=True,
            explanation_mode=explanation_mode,
            interpretive_question=interpretive_question,
            input_chunk_count=chunk_count,
            input_fact_count=fact_count,
        )

    if not settings.synthesis_available:
        reason_code = "synthesis_disabled" if not settings.use_synthesis_model else "synthesis_model_missing"
        return RouteDecision(
            route="primary",
            use_model=True,
            model_requested=settings.primary_model,
            reason="Primary model used because synthesis routing is unavailable.",
            reason_codes=[reason_code],
            deterministic_mode=False,
            explanation_mode=explanation_mode,
            interpretive_question=interpretive_question,
            input_chunk_count=chunk_count,
            input_fact_count=fact_count,
        )

    reason_codes: list[str] = []
    if explanation_mode and settings.synthesis_for_explanation_mode:
        reason_codes.append("explanation_mode")
    if chunk_count >= settings.synthesis_min_context_chunks:
        reason_codes.append("multi_chunk_context")
    if interpretive_question:
        reason_codes.append("interpretive_question")

    if reason_codes:
        return RouteDecision(
            route="synthesis",
            use_model=True,
            model_requested=settings.synthesis_model,
            reason=_describe_route_reason(reason_codes, chunk_count),
            reason_codes=reason_codes,
            deterministic_mode=False,
            explanation_mode=explanation_mode,
            interpretive_question=interpretive_question,
            input_chunk_count=chunk_count,
            input_fact_count=fact_count,
        )

    return RouteDecision(
        route="primary",
        use_model=True,
        model_requested=settings.primary_model,
        reason="Primary model used because the question is direct and the retrieved context is small.",
        reason_codes=["direct_question"],
        deterministic_mode=False,
        explanation_mode=explanation_mode,
        interpretive_question=interpretive_question,
        input_chunk_count=chunk_count,
        input_fact_count=fact_count,
    )


def filter_supporting_chunks(
    chunks: list[dict[str, Any]],
    source_refs: list[SourceRef],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    if not source_refs:
        return list(chunks[:limit])

    matched: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        if not _chunk_matches_source_refs(chunk, source_refs):
            continue
        chunk_id = _chunk_identity(chunk)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        matched.append(chunk)
        if len(matched) >= limit:
            return matched

    if matched:
        return matched[:limit]
    return list(chunks[:limit])


def build_grounded_prompt(
    question: str,
    context_chunks: list[dict[str, Any]],
    *,
    facts: Optional[dict[str, Any]] = None,
    source_refs: Optional[list[SourceRef]] = None,
    max_context_chars: int = PRIMARY_CONTEXT_CHAR_BUDGET,
) -> PromptBundle:
    selected_chunks, context_text, context_char_count = _budget_context_chunks(
        context_chunks,
        max_context_chars=max_context_chars,
    )
    normalized_facts = dict(facts or {})
    fact_lines = _format_fact_section(normalized_facts)
    fact_source_lines = _format_source_ref_section(source_refs or [])

    system_prompt = (
        "You are the NTTV curriculum assistant. "
        "Answer only from the deterministic facts and retrieved context provided in this prompt. "
        "Do not add unsupported martial-arts lore, folklore, lineage claims, or training advice that is not explicitly stated. "
        "If the provided material is incomplete or ambiguous, say so clearly."
    )

    user_prompt = (
        "Use only the material below.\n"
        "When you rely on retrieved context, cite it inline with [1], [2], and so on.\n"
        "When you rely on deterministic facts, cite the matching fact source like [F1] when one is available.\n"
        "If the material does not fully answer the question, explicitly say the available material is incomplete.\n"
        "Do not fill gaps with outside martial-arts lore.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Deterministic facts:\n{fact_lines}\n\n"
        f"Fact sources:\n{fact_source_lines}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Grounded answer:"
    )

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        selected_chunks=selected_chunks,
        selected_fact_count=len(normalized_facts),
        context_char_count=context_char_count,
    )


def generate_grounded_answer(
    question: str,
    context_chunks: list[dict[str, Any]],
    *,
    facts: Optional[dict[str, Any]] = None,
    source_refs: Optional[list[SourceRef]] = None,
    deterministic_mode: bool = False,
    settings: Optional[LLMRoutingSettings] = None,
    completion_callable: Optional[CompletionCallable] = None,
) -> GeneratedAnswer:
    settings = settings or LLMRoutingSettings.from_env()
    completion = completion_callable or _chat_completion
    fact_count = len(facts or {})
    decision = select_generation_route(
        question,
        context_chunks,
        fact_count=fact_count,
        deterministic_mode=deterministic_mode,
        settings=settings,
    )
    max_context_chars = (
        settings.synthesis_context_char_budget
        if decision.route == "synthesis"
        else settings.primary_context_char_budget
    )
    prompt_bundle = build_grounded_prompt(
        question,
        context_chunks,
        facts=facts,
        source_refs=source_refs,
        max_context_chars=max_context_chars,
    )

    attempted_models: list[str] = []
    fallback_used = False
    fallback_reason: Optional[str] = None
    last_raw = "{}"

    requested_model = decision.model_requested
    if requested_model is None:
        debug = decision.to_debug_payload(
            model_used="deterministic_composer",
            selected_chunk_count=len(prompt_bundle.selected_chunks),
            selected_fact_count=prompt_bundle.selected_fact_count,
            context_char_count=prompt_bundle.context_char_count,
        )
        return GeneratedAnswer("", "{}", debug, prompt_bundle)

    models_to_try = [requested_model]
    if requested_model != settings.primary_model:
        models_to_try.append(settings.primary_model)

    final_model_used: Optional[str] = None
    final_text = ""
    for index, model_name in enumerate(models_to_try):
        attempted_models.append(model_name)
        try:
            text, raw_json = completion(model_name, prompt_bundle.to_messages(), settings)
            last_raw = raw_json or "{}"
            if text.strip():
                final_model_used = model_name
                final_text = text.strip()
                if index > 0:
                    fallback_used = True
                break
            fallback_reason = f"{model_name} returned no text."
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if index > 0:
                fallback_used = True
        if model_name != settings.primary_model and settings.primary_model in models_to_try:
            fallback_used = True

    debug = decision.to_debug_payload(
        model_used=final_model_used,
        selected_chunk_count=len(prompt_bundle.selected_chunks),
        selected_fact_count=prompt_bundle.selected_fact_count,
        context_char_count=prompt_bundle.context_char_count,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        attempted_models=attempted_models,
    )

    if not final_text:
        if not final_model_used and attempted_models:
            debug["model_used"] = attempted_models[-1]
        return GeneratedAnswer("", last_raw, debug, prompt_bundle)

    return GeneratedAnswer(final_text, last_raw, debug, prompt_bundle)


def is_explanation_question(question: str) -> bool:
    text = question or ""
    return any(pattern.search(text) for pattern in _EXPLANATION_PATTERNS)


def is_interpretive_question(question: str) -> bool:
    text = question or ""
    return any(pattern.search(text) for pattern in _INTERPRETIVE_PATTERNS)


def _chat_completion(
    model_name: str,
    messages: list[dict[str, str]],
    settings: LLMRoutingSettings,
) -> tuple[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    if "openrouter" in settings.api_base.lower():
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER", "https://ninjatrainingtv.com")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "NTTV Chatbot")

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }

    response = requests.post(
        settings.chat_completions_url,
        headers=headers,
        json=payload,
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return text, json.dumps(data, ensure_ascii=False)[:4000]


def _budget_context_chunks(
    context_chunks: list[dict[str, Any]],
    *,
    max_context_chars: int,
) -> tuple[list[dict[str, Any]], str, int]:
    selected_chunks: list[dict[str, Any]] = []
    blocks: list[str] = []
    total_chars = 0

    for index, chunk in enumerate(context_chunks, start=1):
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break

        block = _render_context_block(chunk, index=index, max_block_chars=remaining)
        if not block:
            continue

        selected_chunks.append(chunk)
        blocks.append(block)
        total_chars += len(block)

    return selected_chunks, "".join(blocks).strip() or "(none provided)", total_chars


def _render_context_block(
    chunk: dict[str, Any],
    *,
    index: int,
    max_block_chars: int,
) -> str:
    meta = chunk.get("meta") or {}
    source = (
        meta.get("source_file")
        or meta.get("source")
        or chunk.get("source_file")
        or chunk.get("source")
        or "unknown"
    )
    page_start = meta.get("page_start") or meta.get("page") or chunk.get("page_start") or chunk.get("page")
    page_end = meta.get("page_end") or chunk.get("page_end") or page_start
    heading_path = " > ".join(meta.get("heading_path") or chunk.get("heading_path") or [])
    priority_bucket = meta.get("priority_bucket") or ""

    header_lines = [f"[{index}] Source: {os.path.basename(source)}"]
    if page_start and page_end and page_start != page_end:
        header_lines.append(f"Pages: {page_start}-{page_end}")
    elif page_start:
        header_lines.append(f"Page: {page_start}")
    if heading_path:
        header_lines.append(f"Heading: {heading_path}")
    if priority_bucket:
        header_lines.append(f"Priority: {priority_bucket}")
    header = " | ".join(header_lines)

    text = (chunk.get("text") or "").strip()
    if not text:
        return ""

    room_for_text = min(
        MAX_CONTEXT_CHUNK_TEXT_CHARS,
        max_block_chars - len(header) - len("\nText:\n\n"),
    )
    if room_for_text < MIN_CONTEXT_CHUNK_TEXT_CHARS and max_block_chars < (len(header) + MIN_CONTEXT_CHUNK_TEXT_CHARS):
        return ""

    trimmed = text[: max(0, room_for_text)].rstrip()
    if len(trimmed) < len(text):
        trimmed = trimmed[: max(0, room_for_text - len("\n[truncated]"))].rstrip() + "\n[truncated]"

    return f"{header}\nText:\n{trimmed}\n\n"


def _format_fact_section(facts: dict[str, Any]) -> str:
    if not facts:
        return "(none provided)"
    return json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True)


def _format_source_ref_section(source_refs: list[SourceRef]) -> str:
    if not source_refs:
        return "(none provided)"

    lines: list[str] = []
    for index, source_ref in enumerate(source_refs, start=1):
        label = f"[F{index}] {os.path.basename(source_ref.source)}"
        if source_ref.page_start and source_ref.page_end and source_ref.page_start != source_ref.page_end:
            label += f" (pp. {source_ref.page_start}-{source_ref.page_end})"
        elif source_ref.page_start:
            label += f" (p. {source_ref.page_start})"
        if source_ref.heading_path:
            label += f" | {' > '.join(source_ref.heading_path)}"
        lines.append(label)
    return "\n".join(lines)


def _chunk_matches_source_refs(chunk: dict[str, Any], source_refs: list[SourceRef]) -> bool:
    meta = chunk.get("meta") or {}
    chunk_source = (
        meta.get("source_file")
        or meta.get("source")
        or chunk.get("source_file")
        or chunk.get("source")
        or ""
    )
    chunk_base = os.path.basename(chunk_source).lower()
    chunk_id = meta.get("chunk_id") or chunk.get("chunk_id")
    page_start = meta.get("page_start") or meta.get("page") or chunk.get("page_start") or chunk.get("page")
    page_end = meta.get("page_end") or chunk.get("page_end") or page_start
    heading_path = list(meta.get("heading_path") or chunk.get("heading_path") or [])

    for source_ref in source_refs:
        ref_base = os.path.basename(source_ref.source).lower()
        if ref_base != chunk_base:
            continue
        if source_ref.chunk_id and chunk_id and source_ref.chunk_id == chunk_id:
            return True
        if source_ref.heading_path and heading_path and source_ref.heading_path == heading_path:
            return True
        if source_ref.page_start and page_start:
            ref_end = source_ref.page_end or source_ref.page_start
            chunk_end = page_end or page_start
            if ref_end >= page_start and source_ref.page_start <= chunk_end:
                return True
        if not source_ref.chunk_id and not source_ref.heading_path and not source_ref.page_start:
            return True
    return False


def _chunk_identity(chunk: dict[str, Any]) -> str:
    meta = chunk.get("meta") or {}
    return (
        str(meta.get("chunk_id") or chunk.get("chunk_id") or "")
        or f"{meta.get('source_file') or meta.get('source') or chunk.get('source') or 'unknown'}:{id(chunk)}"
    )


def _describe_route_reason(reason_codes: list[str], chunk_count: int) -> str:
    fragments: list[str] = []
    if "explanation_mode" in reason_codes:
        fragments.append("explanation-style wording")
    if "multi_chunk_context" in reason_codes:
        fragments.append(f"{chunk_count} context chunks")
    if "interpretive_question" in reason_codes:
        fragments.append("interpretive wording")
    if "deterministic_composer_enabled" in reason_codes:
        fragments.append("deterministic synthesis is enabled")
    if not fragments:
        return "Selected by routing policy."
    return "Synthesis model selected for " + ", ".join(fragments) + "."


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return _parse_bool(value)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value.strip())


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value.strip())
