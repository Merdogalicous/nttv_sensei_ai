from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import re
import os
import unicodedata

from nttv_chatbot.deterministic import DeterministicResult, build_result

# ----------------------------
# Canonical names + aliases
# ----------------------------
SCHOOL_ALIASES: Dict[str, List[str]] = {
    "Togakure Ryu": [
        "togakure ryu", "togakure-ryu",
        "togakure ryu ninpo", "togakure ryu ninpo taijutsu", "togakure"
    ],
    "Gyokko Ryu": [
        "gyokko ryu", "gyokko-ryu", "gyokko"
    ],
    "Koto Ryu": [
        "koto ryu", "koto-ryu", "koto"
    ],
    "Shinden Fudo Ryu": [
        "shinden fudo ryu", "shinden fudo-ryu",
        "shinden fudo", "shinden fudo ryu dakentaijutsu", "shinden fudo ryu jutaijutsu"
    ],
    "Kukishinden Ryu": [
        "kukishinden ryu", "kukishinden-ryu", "kukishinden"
    ],
    "Takagi Yoshin Ryu": [
        "takagi yoshin ryu", "takagi yoshin-ryu",
        "takagi yoshin", "hoko ryu takagi yoshin ryu", "takagi"
    ],
    "Gikan Ryu": [
        "gikan ryu", "gikan-ryu", "gikan"
    ],
    "Gyokushin Ryu": [
        "gyokushin ryu", "gyokushin-ryu", "gyokushin"
    ],
    "Kumogakure Ryu": [
        "kumogakure ryu", "kumogakure-ryu", "kumogakure"
    ],
}

SCHOOL_PROFILE_SAFETY: Dict[str, Dict[str, str]] = {
    "Togakure Ryu": {"translation": "Hidden Door School", "type": "Ninjutsu"},
    "Gyokko Ryu": {"translation": "Jewel Tiger School", "type": "Samurai"},
    "Koto Ryu": {"translation": "Tiger Knocking Down School", "type": "Samurai"},
    "Shinden Fudo Ryu": {"translation": "Immovable Heart School", "type": "Samurai"},
    "Kukishinden Ryu": {"translation": "Nine Demon Gods School", "type": "Samurai"},
    "Takagi Yoshin Ryu": {"translation": "High Tree, Raised Heart School", "type": "Samurai"},
    "Gikan Ryu": {"translation": "Truth, Loyalty, & Justice School", "type": "Samurai"},
    "Gyokushin Ryu": {"translation": "Jeweled Heart School", "type": "Ninjutsu"},
    "Kumogakure Ryu": {"translation": "Hidden Clouds School", "type": "Ninjutsu"},
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("â€“", "-").replace("â€”", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _same_source_name(p_source: str, target_name: str) -> bool:
    """
    Compare FAISS/meta 'source' values (which may include paths) with the
    logical filename used by the extractor. Basename + lowercase.
    """
    if not p_source:
        return False
    base_actual = os.path.basename(p_source).lower()
    base_target = os.path.basename(target_name).lower()
    return base_actual == base_target


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
    raw = (line or "").strip()
    if not raw or raw.startswith(("-", "*")):
        return None

    labels = [raw]
    header_match = re.match(r"^\s*school\s*[:\-–—]\s*(.+?)\s*$", raw, flags=re.IGNORECASE)
    if header_match:
        labels.append(header_match.group(1))
    if raw.endswith(":"):
        labels.append(raw[:-1])

    for label in labels:
        canon = _canon_from_school_label(label)
        if canon:
            return canon
    return None


def _looks_like_school_header(line: str) -> bool:
    return _school_header_canon(line) is not None


def _canon_for_query(question: str) -> Optional[str]:
    qn = _norm(question)
    for canon in SCHOOL_ALIASES:
        if any(tok in qn for tok in _school_tokens(canon)):
            return canon
    m = re.search(r"([a-z0-9\- ]+)\s+ryu\b", qn)
    if m:
        guess = m.group(1).strip().replace("-", " ")
        for canon in SCHOOL_ALIASES.keys():
            if _norm(canon).startswith(guess):
                return canon
    return None


# ----------------------------
# List-intent detection (EXPORTED)
# ----------------------------
def is_school_list_query(question: str) -> bool:
    q = _norm(question)
    triggers = [
        "what are the schools of the bujinkan",
        "list the schools of the bujinkan",
        "nine schools of the bujinkan",
        "what are the nine schools",
        "list the nine schools",
        "what schools are in the bujinkan",
        "which schools are in the bujinkan",
    ]
    return any(t in q for t in triggers)


# ----------------------------
# Slicing & field extraction
# ----------------------------
_FIELD_KEYS = ["translation", "type", "focus", "weapons", "notes"]


def _extract_school_block(lines: List[str], start: int) -> List[str]:
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].strip() == "---" or _looks_like_school_header(lines[idx]):
            end = idx
            break
    return lines[start:end]


def _slice_school_blocks(blob: str) -> List[Tuple[str, List[str]]]:
    lines = blob.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _looks_like_school_header(ln)]
    blocks: List[Tuple[str, List[str]]] = []
    for idx in idxs:
        block = _extract_school_block(lines, idx)
        if block:
            blocks.append((block[0], block[1:]))
    return blocks


def _header_matches(header_line: str, canon: str) -> bool:
    return _school_header_canon(header_line) == canon


def _parse_fields(block_lines: List[str]) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for ln in block_lines:
        if not ln.strip():
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z ]{1,20}):\s*(.*)$", ln)
        if m:
            key = _norm(m.group(1))
            val = m.group(2).strip()
            data[key] = (data.get(key, "") + (" " if key in data and data[key] else "") + val).strip()
        else:
            if data:
                last_key = list(data.keys())[-1]
                data[last_key] = (data[last_key] + " " + ln.strip()).strip()
    return {k: v.strip() for k, v in data.items() if k in _FIELD_KEYS and v.strip()}


def _format_profile(canon: str, fields: Dict[str, str], bullets: bool) -> str:
    title = canon
    if bullets:
        parts = [f"{title}:"]
        for key in ["translation", "type", "focus", "weapons", "notes"]:
            if key in fields:
                parts.append(f"- {key.capitalize()}: {fields[key]}")
        return "\n".join(parts)

    segs = []
    if "translation" in fields:
        segs.append(f'"{fields["translation"]}".')
    if "type" in fields:
        segs.append(f'Type: {fields["type"]}.')
    if "focus" in fields:
        segs.append(f'Focus: {fields["focus"]}.')
    if "weapons" in fields:
        segs.append(f'Weapons: {fields["weapons"]}.')
    if "notes" in fields:
        segs.append(f'Notes: {fields["notes"]}.')
    return f"{title}: " + (" ".join(segs) if segs else "")


def _collect_schools_blob(passages: List[Dict[str, Any]]) -> str:
    candidates: List[Tuple[int, int, str]] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        src = src_raw.lower()
        txt = (passage.get("text") or "").strip()
        if not txt:
            continue
        if _same_source_name(src_raw, "Schools of the Bujinkan Summaries.txt") or "schools of the bujinkan summaries" in src:
            syn = 0 if "(synthetic)" in src else 1
            candidates.append((syn, -len(txt), txt))
        elif "school:" in _norm(txt):
            candidates.append((1, -len(txt), txt))
    if not candidates:
        return ""
    candidates.sort()
    seen = set()
    out: List[str] = []
    for _, _, txt in candidates:
        if txt not in seen:
            seen.add(txt)
            out.append(txt)
    return "\n\n".join(out)


def _fallback_block_by_alias(blob: str, canon: str) -> Optional[List[str]]:
    if not blob.strip():
        return None
    lines = blob.splitlines()
    for idx, line in enumerate(lines):
        if _school_header_canon(line) == canon:
            return _extract_school_block(lines, idx)
    return None


def _infer_fields_from_freeblock(free_lines: List[str]) -> Dict[str, str]:
    txt = "\n".join(free_lines)
    data = _parse_fields(free_lines)
    if data:
        return data

    n = _norm(txt)
    inferred: Dict[str, str] = {}

    if any(term in n for term in ["ninpo", "ninjutsu"]):
        inferred["type"] = "Ninjutsu"
    elif "samurai school" in n or "samurai schools" in n:
        inferred["type"] = "Samurai"

    translation_match = re.search(r'translation[: ]+["“](.+?)["”]', txt, flags=re.IGNORECASE)
    if translation_match:
        inferred["translation"] = translation_match.group(1).strip()

    focus_terms = []
    for term in [
        "stealth", "infiltration", "surprise", "espionage", "distance", "timing", "kamae",
        "kosshijutsu", "koppojutsu", "striking", "bone", "joint", "throws", "grappling",
        "dakentaijutsu", "jutaijutsu",
    ]:
        if term in n:
            focus_terms.append(term)
    if focus_terms:
        inferred["focus"] = ", ".join(sorted(set(focus_terms)))

    weapon_terms = []
    for term in [
        "shuriken", "senban", "kunai", "kodachi", "katana", "yari", "naginata", "bo", "hanbo",
        "kusarifundo", "kusari fundo", "kyoketsu shoge", "kyoketsu-shoge", "tessen", "jutte", "jitte",
    ]:
        if term in n:
            weapon_terms.append(term)
    if weapon_terms:
        inferred["weapons"] = ", ".join(sorted(set(weapon_terms)))

    return inferred


def _translation_matches_other_school(canon: str, translation: str) -> bool:
    candidate = _norm(translation)
    if not candidate:
        return False
    own = _norm((SCHOOL_PROFILE_SAFETY.get(canon) or {}).get("translation", ""))
    if own and candidate == own:
        return False
    for other_canon, metadata in SCHOOL_PROFILE_SAFETY.items():
        if other_canon == canon:
            continue
        other_translation = _norm(metadata.get("translation", ""))
        if other_translation and candidate == other_translation:
            return True
    return False


def _type_conflicts_with_school(canon: str, school_type: str) -> bool:
    expected = _norm((SCHOOL_PROFILE_SAFETY.get(canon) or {}).get("type", ""))
    actual = _norm(school_type)
    return bool(expected and actual and actual != expected)


def _finalize_school_fields(
    canon: str,
    fields: Dict[str, str],
    *,
    strict: bool,
) -> Optional[Dict[str, str]]:
    cleaned = {key: value.strip() for key, value in fields.items() if isinstance(value, str) and value.strip()}
    if not cleaned:
        return None

    if _translation_matches_other_school(canon, cleaned.get("translation", "")):
        return None
    if strict and _type_conflicts_with_school(canon, cleaned.get("type", "")):
        return None

    safety = SCHOOL_PROFILE_SAFETY.get(canon) or {}
    if not cleaned.get("translation") and safety.get("translation"):
        cleaned["translation"] = safety["translation"]
    if not cleaned.get("type") and safety.get("type"):
        cleaned["type"] = safety["type"]
    return cleaned


def _extract_fields_from_school_block(
    canon: str,
    header: str,
    body: List[str],
) -> Optional[Dict[str, str]]:
    parsed = _parse_fields(body)
    if parsed:
        return _finalize_school_fields(canon, parsed, strict=False)
    inferred = _infer_fields_from_freeblock([header] + body)
    return _finalize_school_fields(canon, inferred, strict=True)


def _canon_from_header(header_line: str) -> Optional[str]:
    return _school_header_canon(header_line)


# ----------------------------
# Public API (EXPORTED)
# ----------------------------
def try_answer_schools_list(
    question: str,
    passages: List[Dict[str, Any]],
    *,
    bullets: bool = True,
) -> Optional[DeterministicResult]:
    """Return a list of the nine schools, if the question asks for the list."""
    if not is_school_list_query(question):
        return None

    blob = _collect_schools_blob(passages)
    if not blob.strip():
        return None

    blocks = _slice_school_blocks(blob)
    if not blocks:
        return None

    names: List[str] = []
    seen = set()
    for header, _ in blocks:
        canon = _canon_from_header(header)
        if canon and canon not in seen:
            seen.add(canon)
            names.append(canon)

    if not names:
        return None

    canonical_order = [
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
    if set(n.lower() for n in names) >= set(n.lower() for n in canonical_order):
        order_map = {name.lower(): idx for idx, name in enumerate(canonical_order)}
        names.sort(key=lambda name: order_map.get(name.lower(), 999))

    return build_result(
        det_path="schools/list",
        answer_type="school_list",
        facts={
            "list_title": "The Nine Schools of the Bujinkan",
            "school_names": names,
        },
        passages=passages,
        preferred_sources=["Schools of the Bujinkan Summaries.txt"],
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
    Return a compact profile for a single school (Translation / Type / Focus / Weapons / Notes).

    IMPORTANT: If the query looks like a soke/grandmaster query, return None so the leadership
    extractor can take over.
    """
    ql = _norm(question)
    if any(term in ql for term in ["soke", "grandmaster"]):
        return None

    canon = _canon_for_query(question)
    if not canon:
        return None

    blob = _collect_schools_blob(passages)
    if not blob.strip():
        return None

    blocks = _slice_school_blocks(blob)
    fields: Optional[Dict[str, str]] = None

    if blocks:
        for header, body in blocks:
            if _header_matches(header, canon):
                fields = _extract_fields_from_school_block(canon, header, body)
                if fields:
                    break

    if not fields:
        window = _fallback_block_by_alias(blob, canon)
        if window:
            fields = _finalize_school_fields(canon, _infer_fields_from_freeblock(window), strict=True)

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
        preferred_sources=["Schools of the Bujinkan Summaries.txt"],
        confidence=0.95,
        display_hints={"explain": True},
        followup_suggestions=["Ask for the other Bujinkan schools if you want the full list."],
    )
