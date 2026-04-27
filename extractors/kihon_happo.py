from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result


CANON_DEF = "Kihon Happo consists of Kosshi Kihon Sanpo and Torite Goho."
CANON_KOSSHI = ["Ichimonji no Kata", "Hicho no Kata", "Jumonji no Kata"]
CANON_TORITE = ["Omote Gyaku", "Omote Gyaku Ken Sabaki", "Ura Gyaku", "Musha Dori", "Ganseki Nage"]

UNWANTED_HINTS = (
    "drill the kihon happo",
    "practice the kihon happo",
    "use it against attackers",
    "from all kamae",
    "the five forms of grappling",
    "torite goho gata",
    "kihon happo.",
    "#",
)

TRIGGER_PHRASES = (
    "kihon happo",
    "kihon happo",
    "kihon-happo",
    "kihon-happo",
    "eight basics",
    "8 basics",
    "the eight basics",
)


def _is_junk_line(text: str) -> bool:
    lowered = text.lower().strip()
    return any(hint in lowered for hint in UNWANTED_HINTS)


def _clean_item(text: str) -> str:
    cleaned = text.strip(" -•\t.,;").replace("  ", " ")
    return cleaned.replace("no  kata", "no Kata")


def _split_items(tail: str) -> List[str]:
    items: list[str] = []
    for part in re.split(r"[;,]", tail):
        cleaned = _clean_item(part)
        if 2 <= len(cleaned) <= 60 and not _is_junk_line(cleaned):
            items.append(cleaned)
    return items


def _extract_lists_from_text(text: str) -> tuple[List[str], List[str]]:
    kosshi: list[str] = []
    torite: list[str] = []

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or _is_junk_line(line):
            continue
        lowered = line.lower()
        if "kosshi" in lowered and ("sanpo" in lowered or "sanpo" in lowered):
            tail = line.split(":", 1)[1].strip() if ":" in line else line
            kosshi.extend(_split_items(tail))
            continue
        if "torite" in lowered and ("goho" in lowered or "goho" in lowered):
            tail = line.split(":", 1)[1].strip() if ":" in line else line
            torite.extend(_split_items(tail))
            continue

    def dedupe(items: List[str]) -> List[str]:
        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            if item and item not in seen:
                output.append(item)
                seen.add(item)
        return output

    kosshi = dedupe(kosshi)
    torite = dedupe(torite)

    def looks_bad(items: List[str], expected: List[str]) -> bool:
        if not items:
            return True
        bad_hits = sum(1 for item in items if _is_junk_line(item.lower()))
        overlap = sum(1 for item in items if item in expected)
        return bad_hits > 0 or overlap < 1

    if looks_bad(kosshi, CANON_KOSSHI):
        kosshi = CANON_KOSSHI[:]
    else:
        ordered = [item for item in CANON_KOSSHI if item in kosshi]
        for item in kosshi:
            if item not in ordered:
                ordered.append(item)
        kosshi = ordered[:3]

    if looks_bad(torite, CANON_TORITE):
        torite = CANON_TORITE[:]
    else:
        ordered = [item for item in CANON_TORITE if item in torite]
        for item in torite:
            if item not in ordered:
                ordered.append(item)
        torite = ordered[:5]

    return kosshi, torite


def _question_triggers_kihon(question: str) -> bool:
    ql = (question or "").lower()
    return any(trigger in ql for trigger in TRIGGER_PHRASES)


def try_answer_kihon_happo(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    if not _question_triggers_kihon(question):
        return None

    kosshi: list[str] = []
    torite: list[str] = []
    for passage in passages[:12]:
        parsed_kosshi, parsed_torite = _extract_lists_from_text(passage.get("text", ""))
        if parsed_kosshi and not kosshi:
            kosshi = parsed_kosshi
        if parsed_torite and not torite:
            torite = parsed_torite
        if kosshi and torite:
            break

    if not kosshi:
        kosshi = CANON_KOSSHI[:]
    if not torite:
        torite = CANON_TORITE[:]

    return build_result(
        det_path="deterministic/kihon",
        answer_type="kihon_happo",
        facts={
            "topic": "Kihon Happo",
            "definition": CANON_DEF,
            "kosshi_items": kosshi,
            "torite_items": torite,
        },
        passages=passages,
        preferred_sources=["nttv training reference.txt", "nttv rank requirements.txt"],
        confidence=0.98,
        display_hints={"explain": True},
        followup_suggestions=["Ask about a specific Kihon Happo kata if you want a narrower answer."],
    )
