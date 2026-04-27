from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result

from .common import dedupe_preserve


_ELEMENT_DATA: Dict[str, Dict[str, Any]] = {
    "chi": {
        "name": "Chi no Kata",
        "english": "Earth Form",
        "aliases": ["chi no kata", "earth form"],
        "summary": (
            "Chi no Kata (Earth Form) emphasizes grounding, structure, and a strong, stable base. "
            "Movements tend to sink and rise, teaching you to connect to the ground and generate power from the legs and hips."
        ),
    },
    "sui": {
        "name": "Sui no Kata",
        "english": "Water Form",
        "aliases": ["sui no kata", "water form"],
        "summary": (
            "Sui no Kata (Water Form) focuses on flowing, outward-inward and circular movement. "
            "It trains adaptability, continuous motion, and the ability to redirect force rather than meeting it head-on."
        ),
    },
    "ka": {
        "name": "Ka no Kata",
        "english": "Fire Form",
        "aliases": ["ka no kata", "fire form"],
        "summary": (
            "Ka no Kata (Fire Form) develops sharp, accelerating strikes with a twisting quality. "
            "It represents expansion, intensity, and the ability to explode through an opponent's guard."
        ),
    },
    "fu": {
        "name": "Fu no Kata",
        "english": "Wind Form",
        "aliases": ["fu no kata", "wind form"],
        "summary": (
            "Fu no Kata (Wind Form) trains light, off-line movement and angled entries. "
            "It emphasizes evasion, changing position, and striking from unexpected angles like the movement of wind around obstacles."
        ),
    },
    "ku": {
        "name": "Ku no Kata",
        "english": "Void Form",
        "aliases": ["ku no kata", "void form"],
        "summary": (
            "Ku no Kata (Void Form) expresses timing, distance, and the use of space. "
            "It represents emptiness and potential, teaching you to move at the right moment and appear where the opponent is unprepared."
        ),
    },
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _looks_like_sanshin_question(question: str) -> bool:
    q = _norm(question)
    if "sanshin" in q or "san shin" in q:
        return True
    for meta in _ELEMENT_DATA.values():
        for alias in meta["aliases"]:
            if alias in q:
                return True
    return False


def _detect_element(question: str) -> Optional[Dict[str, Any]]:
    q = _norm(question)
    for meta in _ELEMENT_DATA.values():
        for alias in meta["aliases"]:
            if alias in q:
                return meta
    return None


def _wants_list(question: str) -> bool:
    q = _norm(question)
    return ("what are" in q or "list" in q or "which" in q or "name the" in q) and (
        "sanshin" in q or "san shin" in q or "five elements" in q or "5 elements" in q
    )


def _wants_overview(question: str) -> bool:
    q = _norm(question)
    if ("what is" in q or "explain" in q or "describe" in q) and ("sanshin" in q or "san shin" in q):
        return True
    return "sanshin no kata" in q or "san shin no kata" in q


def _collect_after_anchor(blob: str, anchor_regex: str, window: int = 3000) -> str:
    match = re.search(anchor_regex, blob, flags=re.I)
    if not match:
        return ""
    return blob[match.end() : match.end() + window]


def _parse_bullets_or_shortlines(seg: str) -> List[str]:
    lines: list[str] = []
    started = False
    for raw in seg.splitlines():
        stripped = raw.strip()
        if not stripped:
            if started:
                break
            continue
        if stripped.startswith(("·", "-", "*", "•")):
            started = True
            lines.append(stripped.lstrip("·-*• ").strip())
        elif started:
            break
    return lines


def try_answer_sanshin(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    if not _looks_like_sanshin_question(question):
        return None

    element = _detect_element(question)
    if element is not None:
        return build_result(
            det_path="sanshin/element",
            answer_type="sanshin_element",
            facts={
                "element_name": element["name"],
                "english_name": element["english"],
                "summary": element["summary"],
            },
            passages=passages,
            preferred_sources=["nttv training reference.txt"],
            confidence=0.98,
            display_hints={"explain": True},
        )

    ordered = dedupe_preserve([meta["name"] for meta in _ELEMENT_DATA.values()])

    if _wants_list(question):
        return build_result(
            det_path="sanshin/list",
            answer_type="sanshin_list",
            facts={
                "title": "Sanshin no Kata (Five Elements)",
                "items": ordered,
            },
            passages=passages,
            preferred_sources=["nttv training reference.txt"],
            confidence=0.97,
            display_hints={"explain": True},
        )

    if _wants_overview(question):
        return build_result(
            det_path="sanshin/overview",
            answer_type="sanshin_overview",
            facts={
                "summary": (
                    "Sanshin no Kata (Three Hearts / Five Elements) is a set of five fundamental solo forms "
                    "used in the Bujinkan to train body structure, timing, and feeling. Each form is associated "
                    "with an element and a characteristic way of moving."
                ),
                "items": ordered,
            },
            passages=passages,
            preferred_sources=["nttv training reference.txt"],
            confidence=0.97,
            display_hints={"explain": True},
        )

    return build_result(
        det_path="sanshin/list",
        answer_type="sanshin_list",
        facts={
            "title": "Sanshin no Kata",
            "items": ordered,
        },
        passages=passages,
        preferred_sources=["nttv training reference.txt"],
        confidence=0.93,
        display_hints={"explain": True},
    )
