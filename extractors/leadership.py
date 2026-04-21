from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result


SCHOOL_ALIASES = {
    "gyokko-ryu": [
        "gyokko-ryu",
        "gyokko ryu",
        "gyokko-ryu",
        "gyokko ryu",
        "gyokku ryu",
        "gyokku-ryu",
        "gyokku ryu",
    ],
    "koto-ryu": [
        "koto-ryu",
        "koto ryu",
        "koto-ryu",
        "koto ryu",
        "koto ryu koppojutsu",
        "koto-ryu koppojutsu",
    ],
    "togakure-ryu": ["togakure-ryu", "togakure ryu", "togakure-ryu", "togakure ryu"],
    "shinden fudo-ryu": ["shinden fudo-ryu", "shinden fudo ryu", "shinden fudo-ryu", "shinden fudo ryu"],
    "kukishinden-ryu": ["kukishinden-ryu", "kukishinden ryu", "kukishinden-ryu", "kukishinden ryu"],
    "takagi yoshin-ryu": ["takagi yoshin-ryu", "takagi yoshin ryu", "takagi yoshin-ryu", "takagi yoshin ryu"],
    "gikan-ryu": ["gikan-ryu", "gikan ryu", "gikan-ryu", "gikan ryu"],
    "gyokushin-ryu": ["gyokushin-ryu", "gyokushin ryu", "gyokushin-ryu", "gyokushin ryu"],
    "kumogakure-ryu": ["kumogakure-ryu", "kumogakure ryu", "kumogakure-ryu", "kumogakure ryu"],
}

QUALIFIERS = [
    "koshijutsu",
    "kosshijutsu",
    "koppojutsu",
    "dakentaijutsu",
    "jutaijutsu",
    "happo bikenjutsu",
    "happo hikenjutsu",
    "happo bikenjutsu",
    "hikenjutsu",
    "ninpo taijutsu",
    "ninjutsu",
    "budo taijutsu",
    "budo taijutsu",
]

SOKESHIP_KV = re.compile(r"^\s*([A-Za-z0-9 .'`/\-]+?)\s*[:\-\u2013\u2014]\s*(.+?)\s*$")
NAT_FORMS = [
    re.compile(r"^\s*(.+?)\s+(?:has\s+been|was|became)\s+(?:named|appointed|designated\s+as\s+)?(?:the\s+)?s[o\u014d]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s+is\s+(?:the\s+)?s[o\u014d]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*s[o\u014d]ke\s+of\s+(.+?)\s+is\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s+s[o\u014d]ke\s*[:\-\u2013\u2014]\s*(.+?)\s*$", re.IGNORECASE),
]


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_macrons(text: str) -> str:
    return (
        text.replace("ō", "o")
        .replace("ū", "u")
        .replace("ā", "a")
        .replace("ī", "i")
        .replace("Ō", "O")
        .replace("Ū", "U")
        .replace("Ā", "A")
        .replace("Ī", "I")
    )


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _just_school_ryu(text: str) -> str:
    normalized = _strip_macrons(_norm_ws(text)).lower()
    for qualifier in QUALIFIERS:
        normalized = normalized.replace(qualifier, "")
    normalized = _norm_ws(normalized)
    match = re.search(r"\b([a-z' .]+?\sryu)\b", normalized)
    if match:
        return match.group(1)
    if "ryu" in normalized:
        return normalized.split("ryu", 1)[0].strip() + " ryu"
    return normalized


def _alias_to_key(name_like: str) -> Optional[str]:
    core = _just_school_ryu(name_like)
    for key, aliases in SCHOOL_ALIASES.items():
        for alias in aliases:
            if _strip_macrons(alias).lower() in core:
                return key
    match = re.search(r"\b([a-z]+)\s+ryu\b", core)
    if match:
        guess = match.group(0)
        for key, aliases in SCHOOL_ALIASES.items():
            if any(guess in _strip_macrons(alias).lower() for alias in aliases):
                return key
    return None


def _pretty_school(key: str) -> str:
    return key.replace("-", " ").title().replace("Ryu", "Ryu")


def _harvest_pairs_from_text(text: str) -> List[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines = (text or "").splitlines()

    for line in lines:
        if "|" in line:
            cols = [col.strip() for col in line.split("|")]
            if len(cols) >= 2 and len(cols[0]) >= 4 and len(cols[1]) >= 2:
                school_like = _norm_ws(cols[0])
                person = _norm_ws(cols[1])
                if re.search(r"[A-Za-z]", school_like) and re.search(r"[A-Za-z]", person):
                    pairs.append((school_like, person))

    for line in lines:
        match = SOKESHIP_KV.match(line)
        if match:
            school_like = _norm_ws(match.group(1))
            person = _norm_ws(match.group(2))
            if len(school_like) >= 4 and len(person) >= 2:
                pairs.append((school_like, person))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in NAT_FORMS:
            match = pattern.match(stripped)
            if not match:
                continue
            if pattern.pattern.startswith("^\\s*s"):
                school_like = _norm_ws(match.group(1))
                person = _norm_ws(match.group(2))
            elif "s[o\\u014d]ke\\s*" in pattern.pattern:
                school_like = _norm_ws(match.group(1))
                person = _norm_ws(match.group(2))
            else:
                person = _norm_ws(match.group(1))
                school_like = _norm_ws(match.group(2))
            if len(school_like) >= 4 and len(person) >= 2:
                pairs.append((school_like, person))
    return pairs


def _aggregate_leadership_text(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        src = src_raw.lower()
        if _same_source_name(src_raw, "Bujinkan Leadership and Wisdom.txt") or "leadership" in src:
            text = passage.get("text") or ""
            if text:
                parts.append(text)
    return "\n".join(parts)


def _build_leadership_result(
    *,
    school_key: str,
    person: str,
    passages: List[Dict[str, Any]],
    confidence: float,
) -> DeterministicResult:
    return build_result(
        det_path="leadership/soke",
        answer_type="leadership",
        facts={
            "school_name": _pretty_school(school_key),
            "soke_name": person,
            "role": "current soke",
        },
        passages=passages,
        preferred_sources=["Bujinkan Leadership and Wisdom.txt"],
        confidence=confidence,
        display_hints={"explain": False},
    )


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    ql = _strip_macrons(question.lower())
    if not any(token in ql for token in ["soke", "soke'", "sōke", "grandmaster", "headmaster", "current head", "current grandmaster"]):
        return None

    ql = ql.replace("gyokku ryu", "gyokko ryu").replace("gyokku-ryu", "gyokko-ryu")

    target: Optional[str] = None
    for key, aliases in SCHOOL_ALIASES.items():
        if any(_strip_macrons(alias).lower() in ql for alias in aliases):
            target = key
            break
    if not target:
        match = re.search(r"\b([a-z]+)\s+ryu\b", ql)
        if match:
            target = _alias_to_key(match.group(0))
    if not target:
        return None

    pairs = _harvest_pairs_from_text(_aggregate_leadership_text(passages))
    for school_like, person in pairs:
        if _alias_to_key(school_like) == target:
            return _build_leadership_result(school_key=target, person=person, passages=passages, confidence=0.98)

    for passage in passages:
        text = passage.get("text") or ""
        for school_like, person in _harvest_pairs_from_text(text):
            if _alias_to_key(school_like) == target:
                return _build_leadership_result(school_key=target, person=person, passages=passages, confidence=0.95)

    return None
