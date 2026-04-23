from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result


LEADERSHIP_SOURCE = "Bujinkan Leadership and Wisdom.txt"
SCHOOLS_SUMMARY_SOURCE = "Schools of the Bujinkan Summaries.txt"


SCHOOL_ALIASES = {
    "gyokko-ryu": [
        "gyokko-ryu",
        "gyokko ryu",
        "gyokku ryu",
        "gyokku-ryu",
    ],
    "koto-ryu": [
        "koto-ryu",
        "koto ryu",
        "koto ryu koppojutsu",
        "koto-ryu koppojutsu",
    ],
    "togakure-ryu": [
        "togakure-ryu",
        "togakure ryu",
    ],
    "shinden fudo-ryu": [
        "shinden fudo-ryu",
        "shinden fudo ryu",
    ],
    "kukishinden-ryu": [
        "kukishinden-ryu",
        "kukishinden ryu",
    ],
    "takagi yoshin-ryu": [
        "takagi yoshin-ryu",
        "takagi yoshin ryu",
    ],
    "gikan-ryu": [
        "gikan-ryu",
        "gikan ryu",
    ],
    "gyokushin-ryu": [
        "gyokushin-ryu",
        "gyokushin ryu",
    ],
    "kumogakure-ryu": [
        "kumogakure-ryu",
        "kumogakure ryu",
    ],
}


QUALIFIERS = [
    "koshijutsu",
    "kosshijutsu",
    "koppojutsu",
    "dakentaijutsu",
    "jutaijutsu",
    "happo bikenjutsu",
    "happo hikenjutsu",
    "hikenjutsu",
    "ninpo taijutsu",
    "ninjutsu",
    "budo taijutsu",
]


SOKESHIP_KV = re.compile(r"^\s*([A-Za-z0-9 .'`/\-]+?)\s*[:\-\u2013\u2014]\s*(.+?)\s*$")
NAT_FORMS = [
    re.compile(
        r"^\s*(.+?)\s+(?:has\s+been|was|became)\s+(?:named|appointed|designated\s+as\s+)?(?:the\s+)?s[o\u014d]ke\s+of\s+(.+?)\s*\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(.+?)\s+is\s+(?:the\s+)?s[o\u014d]ke\s+of\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*s[o\u014d]ke\s+of\s+(.+?)\s+is\s+(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(.+?)\s+s[o\u014d]ke\s*[:\-\u2013\u2014]\s*(.+?)\s*$", re.IGNORECASE),
]
SOKE_SENTENCE = re.compile(
    r"^\s*(?P<person>.+?)\s+(?:has\s+been|was|became)\s+(?:named|appointed|designated\s+as\s+|made\s+)?(?P<details>.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
SOKE_OF_FRAGMENT = re.compile(
    r"(?:\bthe\s+)?(?:\d{1,2}(?:st|nd|rd|th)\s+)?s[o\u014d]ke\s+of\s+(.+?)(?=(?:\s+and\s+(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)\s+)?s[o\u014d]ke\s+of\s+)|$)",
    re.IGNORECASE,
)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_macrons(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return (
        normalized.replace("Ã…Â", "o")
        .replace("Ã…Â«", "u")
        .replace("Ã„Â", "a")
        .replace("Ã„Â«", "i")
        .replace("Ã…Å’", "O")
        .replace("Ã…Âª", "U")
        .replace("Ã„â‚¬", "A")
        .replace("Ã„Âª", "I")
    )


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _just_school_ryu(text: str) -> str:
    normalized = _strip_macrons(_norm_ws(text)).lower().replace("-", " ")
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
        if any(_strip_macrons(alias).lower() in core for alias in aliases):
            return key
    match = re.search(r"\b([a-z]+)\s+ryu\b", core)
    if not match:
        return None
    guess = match.group(0)
    for key, aliases in SCHOOL_ALIASES.items():
        if any(guess in _strip_macrons(alias).lower() for alias in aliases):
            return key
    return None


def _pretty_school(key: str) -> str:
    return key.replace("-", " ").title()


def _extract_year(value: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", value or "")
    if not match:
        return None
    return int(match.group(0))


def _extract_school_mentions_from_soke_details(details: str) -> List[str]:
    schools: list[str] = []
    for match in SOKE_OF_FRAGMENT.finditer(details or ""):
        school_like = _norm_ws(match.group(1))
        school_like = re.sub(r"\s+in\s+.+$", "", school_like, flags=re.IGNORECASE)
        school_like = school_like.strip(" .,:;")
        school_like = re.sub(r"^both\s+", "", school_like, flags=re.IGNORECASE)
        if not school_like:
            continue
        parts = [school_like]
        if " and " in school_like:
            parts = [
                part.strip(" .,:;")
                for part in re.split(r"\s+and\s+", school_like)
                if part.strip(" .,:;")
            ]
        schools.extend(parts)
    return schools


def _harvest_soke_sentence_records(line: str, order: int) -> List[dict[str, Any]]:
    match = SOKE_SENTENCE.match(line.strip())
    if not match:
        return []

    details = _norm_ws(match.group("details"))
    if "soke of" not in _strip_macrons(details).lower():
        return []

    person = _norm_ws(match.group("person"))
    schools = _extract_school_mentions_from_soke_details(details)
    if not person or not schools:
        return []

    year = _extract_year(line)
    return [
        {
            "school_like": school_like,
            "person": person,
            "year": year,
            "order": order + idx,
        }
        for idx, school_like in enumerate(schools)
    ]


def _harvest_records_from_text(text: str) -> List[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    order = 0
    lines = (text or "").splitlines()

    for line in lines:
        if "|" not in line:
            continue
        cols = [col.strip() for col in line.split("|")]
        if len(cols) < 2:
            continue
        school_like = _norm_ws(cols[0])
        person = _norm_ws(cols[1])
        if not (re.search(r"[A-Za-z]", school_like) and re.search(r"[A-Za-z]", person)):
            continue
        records.append(
            {
                "school_like": school_like,
                "person": person,
                "year": _extract_year(cols[2]) if len(cols) >= 3 else None,
                "order": order,
            }
        )
        order += 1

    for line in lines:
        match = SOKESHIP_KV.match(line)
        if not match:
            continue
        school_like = _norm_ws(match.group(1))
        person = _norm_ws(match.group(2))
        if len(school_like) < 4 or len(person) < 2:
            continue
        records.append(
            {
                "school_like": school_like,
                "person": person,
                "year": _extract_year(line),
                "order": order,
            }
        )
        order += 1

    for line in lines:
        sentence_records = _harvest_soke_sentence_records(line, order)
        if not sentence_records:
            continue
        records.extend(sentence_records)
        order += len(sentence_records)

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
            if len(school_like) < 4 or len(person) < 2:
                continue
            records.append(
                {
                    "school_like": school_like,
                    "person": person,
                    "year": _extract_year(stripped),
                    "order": order,
                }
            )
            order += 1
            break
    return records


def _harvest_soke_updates_from_text(text: str) -> List[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    order = 0
    for line in (text or "").splitlines():
        sentence_records = _harvest_soke_sentence_records(line, order)
        if not sentence_records:
            continue
        records.extend(sentence_records)
        order += len(sentence_records)
    return records


def _aggregate_leadership_text(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        src = src_raw.lower()
        if _same_source_name(src_raw, LEADERSHIP_SOURCE) or "leadership" in src:
            text = passage.get("text") or ""
            if text:
                parts.append(text)
    return "\n".join(parts)


def _aggregate_supplemental_sokeship_text(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        if not _same_source_name(src_raw, SCHOOLS_SUMMARY_SOURCE):
            continue
        text = passage.get("text") or ""
        if text:
            parts.append(text)
    return "\n".join(parts)


def _best_soke_payloads(
    text: str,
    *,
    explicit_updates_only: bool = False,
) -> Dict[str, tuple[tuple[int, int], str]]:
    best: dict[str, tuple[tuple[int, int], str]] = {}
    harvest = _harvest_soke_updates_from_text if explicit_updates_only else _harvest_records_from_text
    for record in harvest(text or ""):
        key = _alias_to_key(record["school_like"])
        if not key:
            continue
        school_name = _pretty_school(key)
        year_score = record["year"] if record["year"] is not None else -1
        score = (year_score, record["order"])
        current = best.get(school_name)
        if current is None or score >= current[0]:
            best[school_name] = (score, record["person"])
    return best


def extract_current_soke_map(passages: List[Dict[str, Any]]) -> Dict[str, str]:
    leadership_text = _aggregate_leadership_text(passages)
    if leadership_text:
        best = _best_soke_payloads(leadership_text)
    else:
        parts = [passage.get("text") or "" for passage in passages if passage.get("text")]
        best = _best_soke_payloads("\n".join(parts))

    supplemental_text = _aggregate_supplemental_sokeship_text(passages)
    for school_name, payload in _best_soke_payloads(
        supplemental_text,
        explicit_updates_only=True,
    ).items():
        current = best.get(school_name)
        if current is None or payload[0] > current[0]:
            best[school_name] = payload

    return {school_name: payload[1] for school_name, payload in best.items()}


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
        preferred_sources=[LEADERSHIP_SOURCE],
        confidence=confidence,
        display_hints={"explain": False},
    )


def try_extract_answer(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    ql = _strip_macrons(question.lower())
    if not any(
        token in ql
        for token in ["soke", "soke'", "s\u014dke", "grandmaster", "headmaster", "current head", "current grandmaster"]
    ):
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

    school_name = _pretty_school(target)
    soke_map = extract_current_soke_map(passages)
    person = soke_map.get(school_name)
    if not person:
        return None

    confidence = 0.98 if _aggregate_leadership_text(passages) else 0.95
    return _build_leadership_result(
        school_key=target,
        person=person,
        passages=passages,
        confidence=confidence,
    )
