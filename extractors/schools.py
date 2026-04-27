from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import re
import unicodedata

from nttv_chatbot.deterministic import DeterministicResult, SourceRef, build_result

from .leadership import LEADERSHIP_SOURCE, extract_current_soke_map


AUTHORITATIVE_SCHOOL_PROFILE_SOURCE = "Schools of the Bujinkan Summaries.txt"

CANONICAL_SCHOOL_ORDER = [
    "Togakure Ryu",
    "Gyokushin Ryu",
    "Kumogakure Ryu",
    "Gikan Ryu",
    "Gyokko Ryu",
    "Koto Ryu",
    "Shinden Fudo Ryu",
    "Kukishinden Ryu",
    "Takagi Yoshin Ryu",
]


SCHOOL_ALIASES: Dict[str, List[str]] = {
    "Togakure Ryu": [
        "togakure ryu",
        "togakure-ryu",
        "togakure ryu ninpo",
        "togakure ryu ninpo taijutsu",
        "togakure",
    ],
    "Gyokko Ryu": [
        "gyokko ryu",
        "gyokko-ryu",
        "gyokko",
        "gyokko ryu kosshijutsu",
    ],
    "Koto Ryu": [
        "koto ryu",
        "koto-ryu",
        "koto",
        "koto ryu koppojutsu",
    ],
    "Shinden Fudo Ryu": [
        "shinden fudo ryu",
        "shinden fudo-ryu",
        "shinden fudo",
        "shinden fudo ryu dakentaijutsu",
        "shinden fudo ryu jutaijutsu",
    ],
    "Kukishinden Ryu": [
        "kukishinden ryu",
        "kukishinden-ryu",
        "kukishinden",
        "kukishinden ryu happo hikenjutsu",
    ],
    "Takagi Yoshin Ryu": [
        "takagi yoshin ryu",
        "takagi yoshin-ryu",
        "takagi yoshin",
        "hoko ryu takagi yoshin ryu",
        "takagi",
    ],
    "Gikan Ryu": [
        "gikan ryu",
        "gikan-ryu",
        "gikan",
    ],
    "Gyokushin Ryu": [
        "gyokushin ryu",
        "gyokushin-ryu",
        "gyokushin",
        "gyokshin",
    ],
    "Kumogakure Ryu": [
        "kumogakure ryu",
        "kumogakure-ryu",
        "kumogakure",
    ],
}


SCHOOL_CANONICAL_FACTS: Dict[str, Dict[str, str]] = {
    "Togakure Ryu": {
        "translation": "Hidden Door School",
        "type": "Ninjutsu",
    },
    "Gyokko Ryu": {
        "translation": "Jewel Tiger School",
        "type": "Samurai",
    },
    "Koto Ryu": {
        "translation": "Tiger Knocking Down School",
        "type": "Samurai",
    },
    "Shinden Fudo Ryu": {
        "translation": "Immovable Heart School",
        "type": "Samurai",
    },
    "Kukishinden Ryu": {
        "translation": "Nine Demon Gods School",
        "type": "Samurai",
    },
    "Takagi Yoshin Ryu": {
        "translation": "High Tree, Raised Heart School",
        "type": "Samurai",
    },
    "Gikan Ryu": {
        "translation": "Truth, Loyalty, & Justice School",
        "type": "Samurai",
    },
    "Gyokushin Ryu": {
        "translation": "Jeweled Heart School",
        "type": "Ninjutsu",
    },
    "Kumogakure Ryu": {
        "translation": "Hiding in the Clouds School",
        "type": "Ninjutsu",
    },
}


_PROFILE_FIELD_KEYS = ("translation", "type", "focus", "weapons", "notes")
_INTEGRITY_FIELD_KEYS = ("aliases", "key points") + _PROFILE_FIELD_KEYS
_CATALOG_DETAIL_TOKENS = (
    "overview",
    "describe",
    "tell me about",
    "including",
    "include",
    "translation",
    "english translation",
    "current soke",
    "soke",
    "style",
    "type",
    "brief description",
    "description",
)


def _norm(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2010", "-").replace("\u2011", "-")
    normalized = normalized.replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _is_authoritative_school_profile_source(source: str) -> bool:
    source_lower = (source or "").lower()
    return _same_source_name(source, AUTHORITATIVE_SCHOOL_PROFILE_SOURCE) or (
        "schools of the bujinkan summaries" in source_lower
    )


def _school_tokens(canon: str) -> List[str]:
    return [_norm(canon)] + [_norm(alias) for alias in SCHOOL_ALIASES.get(canon, [])]


def _canon_from_school_label(label: str) -> Optional[str]:
    candidate = _norm(label).strip(" :-")
    if not candidate:
        return None
    for canon in SCHOOL_ALIASES:
        if candidate in _school_tokens(canon):
            return canon
    return None


def _school_header_canon(line: str) -> Optional[str]:
    normalized = _norm(line)
    if not normalized or normalized.startswith(("-", "*")):
        return None

    labels = [normalized]
    match = re.match(r"^school\s*[:\-]\s*(.+?)\s*$", normalized)
    if match:
        labels.append(match.group(1))
    if normalized.endswith(":"):
        labels.append(normalized[:-1].strip())

    for label in labels:
        canon = _canon_from_school_label(label)
        if canon:
            return canon
    return None


def _looks_like_school_header(line: str) -> bool:
    return _school_header_canon(line) is not None


def _canon_for_query(question: str) -> Optional[str]:
    normalized = _norm(question)
    for canon in SCHOOL_ALIASES:
        if any(token in normalized for token in _school_tokens(canon)):
            return canon
    match = re.search(r"([a-z0-9\- ]+)\s+ryu\b", normalized)
    if not match:
        return None
    guess = match.group(1).strip().replace("-", " ")
    for canon in SCHOOL_ALIASES:
        if _norm(canon).startswith(guess):
            return canon
    return None


def _is_school_group_query(question: str) -> bool:
    normalized = _norm(question)
    if "schools of the bujinkan" in normalized:
        return True
    if "bujinkan" in normalized and ("school" in normalized or "schools" in normalized):
        return True
    if "nine schools of the bujinkan" in normalized:
        return True
    if "nine schools" in normalized and "bujinkan" in normalized:
        return True
    return False


def is_school_catalog_query(question: str) -> bool:
    normalized = _norm(question)
    if not _is_school_group_query(question):
        return False
    return any(token in normalized for token in _CATALOG_DETAIL_TOKENS)


def is_school_list_query(question: str) -> bool:
    normalized = _norm(question)
    if is_school_catalog_query(question):
        return False
    triggers = [
        "what are the schools of the bujinkan",
        "list the schools of the bujinkan",
        "nine schools of the bujinkan",
        "what are the nine schools",
        "list the nine schools",
        "what schools are in the bujinkan",
        "which schools are in the bujinkan",
    ]
    if any(trigger in normalized for trigger in triggers):
        return True
    return _is_school_group_query(question) and "list" in normalized


def _extract_school_block(lines: List[str], start_index: int) -> List[str]:
    end_index = len(lines)
    for idx in range(start_index + 1, len(lines)):
        if lines[idx].strip() == "---" or _looks_like_school_header(lines[idx]):
            end_index = idx
            break
    return lines[start_index:end_index]


def _slice_school_blocks(blob: str) -> List[Tuple[str, List[str]]]:
    lines = blob.splitlines()
    block_starts = [idx for idx, line in enumerate(lines) if _looks_like_school_header(line)]
    blocks: List[Tuple[str, List[str]]] = []
    for idx in block_starts:
        block = _extract_school_block(lines, idx)
        if block:
            blocks.append((block[0], block[1:]))
    return blocks


def _header_matches(header_line: str, canon: str) -> bool:
    return _school_header_canon(header_line) == canon


def _structured_field_key(line: str) -> Optional[str]:
    match = re.match(r"^\s*([A-Za-z][A-Za-z ]{1,20}):\s*", line)
    if not match:
        return None
    key = _norm(match.group(1))
    if key in _INTEGRITY_FIELD_KEYS:
        return key
    return None


def _parse_fields(block_lines: List[str]) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in block_lines:
        if not line.strip():
            continue
        match = re.match(r"^\s*([A-Za-z][A-Za-z ]{1,20}):\s*(.*)$", line)
        if match:
            key = _norm(match.group(1))
            value = match.group(2).strip()
            if key in _PROFILE_FIELD_KEYS:
                data[key] = value
        elif data:
            last_key = list(data.keys())[-1]
            data[last_key] = f"{data[last_key]} {line.strip()}".strip()
    return {key: value.strip() for key, value in data.items() if value.strip()}


def _parse_alias_values(line: str) -> List[str]:
    match = re.match(r"^\s*aliases:\s*(.*)$", _norm(line))
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def _block_integrity_ok(canon: str, header: str, body: List[str]) -> bool:
    if _school_header_canon(header) != canon:
        return False

    field_counts: Dict[str, int] = {}
    for line in body:
        nested_canon = _school_header_canon(line)
        if nested_canon is not None:
            return False

        field_key = _structured_field_key(line)
        if field_key is None:
            continue

        field_counts[field_key] = field_counts.get(field_key, 0) + 1
        if field_counts[field_key] > 1:
            return False

        if field_key == "aliases":
            for alias in _parse_alias_values(line):
                alias_canon = _canon_from_school_label(alias)
                if alias_canon is not None and alias_canon != canon:
                    return False

    return True


def _collect_schools_blob(passages: List[Dict[str, Any]]) -> str:
    candidates: List[Tuple[int, int, str]] = []
    for passage in passages:
        source = passage.get("source") or ""
        text = (passage.get("text") or "").strip()
        if not text or not _is_authoritative_school_profile_source(source):
            continue
        synthetic_rank = 0 if "(synthetic)" in source.lower() else 1
        candidates.append((synthetic_rank, -len(text), text))

    if not candidates:
        return ""

    candidates.sort()
    seen = set()
    ordered: List[str] = []
    for _, _, text in candidates:
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return "\n\n".join(ordered)


def _fallback_block_by_alias(blob: str, canon: str) -> Optional[List[str]]:
    if not blob.strip():
        return None

    allowed_tokens = set(_school_tokens(canon))
    lines = blob.splitlines()
    for idx, line in enumerate(lines):
        token = _norm(line).strip(" :")
        if token not in allowed_tokens:
            continue
        block = _extract_school_block(lines, idx)
        if block and _block_integrity_ok(canon, block[0], block[1:]):
            return block
    return None


def _infer_fields_from_freeblock(free_lines: List[str]) -> Dict[str, str]:
    parsed = _parse_fields(free_lines)
    if parsed:
        return parsed
    return {}


def _translation_is_contaminated(canon: str, translation: str) -> bool:
    candidate = _norm(translation)
    if not candidate:
        return False

    expected = _norm((SCHOOL_CANONICAL_FACTS.get(canon) or {}).get("translation", ""))
    if expected and candidate == expected:
        return False

    for other_canon, facts in SCHOOL_CANONICAL_FACTS.items():
        if other_canon == canon:
            continue
        other_translation = _norm(facts.get("translation", ""))
        if other_translation and other_translation in candidate:
            return True
    return False


def _type_is_contaminated(canon: str, school_type: str) -> bool:
    candidate = _norm(school_type)
    expected = _norm((SCHOOL_CANONICAL_FACTS.get(canon) or {}).get("type", ""))
    return bool(expected and candidate and candidate != expected)


def _finalize_school_fields(canon: str, fields: Dict[str, str]) -> Optional[Dict[str, str]]:
    cleaned = {key: value.strip() for key, value in fields.items() if isinstance(value, str) and value.strip()}

    if _translation_is_contaminated(canon, cleaned.get("translation", "")):
        cleaned.pop("translation", None)
    if _type_is_contaminated(canon, cleaned.get("type", "")):
        cleaned.pop("type", None)

    safety = SCHOOL_CANONICAL_FACTS.get(canon) or {}
    if not cleaned.get("translation") and safety.get("translation"):
        cleaned["translation"] = safety["translation"]
    if not cleaned.get("type") and safety.get("type"):
        cleaned["type"] = safety["type"]

    if not cleaned:
        return None
    return cleaned


def _extract_brief_description(body: List[str], fields: Dict[str, str]) -> Optional[str]:
    collecting_key_points = False
    for line in body:
        token = _norm(line)
        if token == "key points:":
            collecting_key_points = True
            continue
        if collecting_key_points:
            if _structured_field_key(line) is not None:
                break
            stripped = line.strip()
            if stripped.startswith(("-", "*")):
                return stripped.lstrip("-* ").strip()
            if stripped:
                return stripped

    return fields.get("focus") or fields.get("notes")


def _extract_fields_from_school_block(
    canon: str,
    header: str,
    body: List[str],
) -> Optional[Dict[str, str]]:
    if not _block_integrity_ok(canon, header, body):
        return None

    parsed = _parse_fields(body)
    if parsed:
        return _finalize_school_fields(canon, parsed)

    inferred = _infer_fields_from_freeblock([header] + body)
    return _finalize_school_fields(canon, inferred)


def _summary_fields_by_school(passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    blob = _collect_schools_blob(passages)
    if not blob.strip():
        return {}

    summaries: Dict[str, Dict[str, str]] = {}
    for header, body in _slice_school_blocks(blob):
        canon = _school_header_canon(header)
        if not canon or canon in summaries:
            continue
        fields = _extract_fields_from_school_block(canon, header, body)
        if not fields:
            continue
        summary = dict(fields)
        brief_description = _extract_brief_description(body, summary)
        if brief_description:
            summary["brief_description"] = brief_description
        summaries[canon] = summary
    return summaries


def _catalog_source_refs() -> List[SourceRef]:
    return [
        SourceRef(source=AUTHORITATIVE_SCHOOL_PROFILE_SOURCE),
        SourceRef(source=LEADERSHIP_SOURCE),
    ]


def try_answer_school_catalog(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[DeterministicResult]:
    if not is_school_catalog_query(question):
        return None

    summaries = _summary_fields_by_school(passages)
    if not summaries:
        return None

    soke_map = extract_current_soke_map(passages)
    schools: List[Dict[str, str]] = []
    for canon in CANONICAL_SCHOOL_ORDER:
        fields = dict(summaries.get(canon) or {})
        finalized = _finalize_school_fields(canon, fields) or {}

        school_entry: Dict[str, str] = {"school_name": canon}
        for key in ("translation", "type"):
            if finalized.get(key):
                school_entry[key] = finalized[key]
        current_soke = soke_map.get(canon)
        if current_soke:
            school_entry["current_soke"] = current_soke
        brief_description = fields.get("brief_description")
        if brief_description:
            school_entry["brief_description"] = brief_description

        schools.append(school_entry)

    return build_result(
        det_path="schools/catalog",
        answer_type="school_catalog",
        facts={
            "list_title": "The Nine Schools of the Bujinkan",
            "schools": schools,
        },
        source_refs=_catalog_source_refs(),
        confidence=0.97,
        display_hints={"explain": True},
        followup_suggestions=["Ask about one school if you want a more detailed profile."],
    )


def try_answer_schools_list(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[DeterministicResult]:
    if not is_school_list_query(question):
        return None

    summaries = _summary_fields_by_school(passages)
    if not summaries:
        return None

    names = [canon for canon in CANONICAL_SCHOOL_ORDER if canon in summaries]
    if not names:
        return None

    return build_result(
        det_path="schools/list",
        answer_type="school_list",
        facts={
            "list_title": "The Nine Schools of the Bujinkan",
            "school_names": names,
        },
        passages=passages,
        preferred_sources=[AUTHORITATIVE_SCHOOL_PROFILE_SOURCE],
        confidence=0.97,
        display_hints={"explain": True},
        followup_suggestions=["Ask about one school if you want a short profile."],
    )


def try_answer_school_profile(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[DeterministicResult]:
    """
    Return a compact profile for a single school (translation / type / focus / weapons / notes).

    If the query looks like a soke or grandmaster query, return None so the leadership
    extractor can handle lineage questions.
    """
    normalized_question = _norm(question)
    if any(term in normalized_question for term in ["soke", "grandmaster"]):
        return None

    canon = _canon_for_query(question)
    if not canon:
        return None

    summaries = _summary_fields_by_school(passages)
    fields = summaries.get(canon)

    if not fields:
        blob = _collect_schools_blob(passages)
        window = _fallback_block_by_alias(blob, canon)
        if window:
            fallback_fields = _extract_fields_from_school_block(canon, window[0], window[1:])
            if fallback_fields:
                fields = dict(fallback_fields)
                brief_description = _extract_brief_description(window[1:], fields)
                if brief_description:
                    fields["brief_description"] = brief_description

    if not fields:
        return None

    return build_result(
        det_path="schools/profile",
        answer_type="school_profile",
        facts={
            "school_name": canon,
            "translation": fields.get("translation"),
            "type": fields.get("type"),
            "focus": fields.get("focus"),
            "weapons": fields.get("weapons"),
            "notes": fields.get("notes"),
        },
        passages=passages,
        preferred_sources=[AUTHORITATIVE_SCHOOL_PROFILE_SOURCE],
        confidence=0.95,
        display_hints={"explain": True},
        followup_suggestions=["Ask for the other Bujinkan schools if you want the full list."],
    )
