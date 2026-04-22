from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result

from .technique_match import canonical_from_query

try:
    from .technique_loader import parse_technique_md
except Exception:  # pragma: no cover - optional compatibility path
    parse_technique_md = None


CONCEPT_BANS = ("kihon happo", "kihon happo", "sanshin", "school", "schools", "ryu", "ryu")
TRIGGERS = ("what is", "define", "explain", "describe")
NAME_HINTS = (
    "gyaku",
    "dori",
    "kudaki",
    "gatame",
    "otoshi",
    "nage",
    "seoi",
    "kote",
    "musha",
    "take ori",
    "juji",
    "omote",
    "ura",
    "ganseki",
    "hodoki",
    "kata",
    "no kata",
)
EXPECTED_COLS = 12


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _fold(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _lite(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(text))


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _looks_like_technique_q(question: str) -> bool:
    ql = _norm_space(question).lower()
    if any(banned in ql for banned in CONCEPT_BANS):
        return False
    if any(trigger in ql for trigger in TRIGGERS):
        return True
    return len(ql.split()) <= 7 and any(hint in ql for hint in NAME_HINTS)


def _has_direct_technique_trigger(question: str) -> bool:
    ql = _norm_space(question).lower()
    return any(trigger in ql for trigger in TRIGGERS)


def _load_full_technique_text() -> str:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "Technique Descriptions.md",
        here.parent / "Technique Descriptions.md",
    ]
    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""


def _gather_full_technique_text(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        src = src_raw.lower()
        if _same_source_name(src_raw, "Technique Descriptions.md") or "technique descriptions" in src:
            text = passage.get("text", "")
            if text:
                parts.append(text)

    passage_text = "\n".join(parts).strip()
    if not parts:
        return ""

    full_text = _load_full_technique_text().strip()
    if full_text and _looks_incomplete_technique_text(passage_text, full_text):
        return full_text
    return passage_text or full_text


def _looks_incomplete_technique_text(passage_text: str, full_text: str) -> bool:
    if not passage_text.strip():
        return True
    if not full_text.strip():
        return False
    if passage_text.strip() == full_text.strip():
        return False

    passage_rows = _count_technique_rows(passage_text)
    full_rows = _count_technique_rows(full_text)
    if full_rows <= 0:
        return False
    if passage_rows <= 0:
        return True
    return passage_rows < full_rows


def _extract_candidate(question_text: str) -> str:
    match = re.search(r"(?:what\s+is|define|explain|describe)\s+(.+)$", question_text, flags=re.I)
    candidate = (match.group(1) if match else question_text).strip().rstrip("?!.")
    candidate = re.sub(r"\b(technique|in ninjutsu|in bujinkan)\b", "", candidate, flags=re.I).strip()
    return candidate


def _candidate_variants(raw: str) -> List[str]:
    variants: list[str] = []
    raw = _norm_space(raw)
    variants.append(raw)
    if raw.lower().endswith(" no kata"):
        variants.append(raw[:-8].strip())
    else:
        variants.append(f"{raw} no kata")

    raw_no_hyphen = raw.replace("-", " ")
    if raw_no_hyphen != raw:
        variants.append(raw_no_hyphen)
        if raw_no_hyphen.lower().endswith(" no kata"):
            variants.append(raw_no_hyphen[:-8].strip())
        else:
            variants.append(f"{raw_no_hyphen} no kata")

    variants.append(_fold(raw))
    variants.append(_lite(raw))

    seen: set[str] = set()
    output: list[str] = []
    for item in variants:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _iter_csv_like_lines(md_text: str):
    for raw in (md_text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("```"):
            continue
        if "," in raw:
            yield raw


def _split_row_limited(raw: str) -> List[str]:
    parts = raw.split(",", EXPECTED_COLS - 1)
    parts = [part.strip() for part in parts]
    if len(parts) > EXPECTED_COLS:
        parts = parts[: EXPECTED_COLS - 1] + [",".join(parts[EXPECTED_COLS - 1 :])]
    if len(parts) < EXPECTED_COLS:
        parts += [""] * (EXPECTED_COLS - len(parts))
    return parts


def _scan_csv_rows_limited(md_text: str) -> List[List[str]]:
    return [_split_row_limited(raw) for raw in _iter_csv_like_lines(md_text)]


def _has_header(cells: List[str]) -> bool:
    lowered = [cell.strip().lower() for cell in cells]
    return any(item in {"name", "translation", "japanese", "description"} for item in lowered)


def _to_bool(value: str) -> Optional[bool]:
    lowered = (value or "").strip().lower()
    if lowered in {"1", "true", "yes", "y", "âœ…", "âœ“", "âœ”"}:
        return True
    if lowered in {"0", "false", "no", "n", "âŒ", "âœ—", "âœ•"}:
        return False
    return None


def _row_to_record_positional(row: List[str]) -> Dict[str, Any]:
    return {
        "name": row[0],
        "japanese": row[1],
        "translation": row[2],
        "type": row[3],
        "rank": row[4],
        "in_rank": row[5],
        "primary_focus": row[6],
        "safety": row[7],
        "partner_required": _to_bool(row[8]),
        "solo": _to_bool(row[9]),
        "tags": row[10],
        "description": row[11],
    }


def _count_technique_rows(md_text: str) -> int:
    if not md_text.strip():
        return 0

    records = _parse_records(md_text)
    if records:
        return len(records)

    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return 0
    return len(rows[1:] if _has_header(rows[0]) else rows)


def _parse_records(md_text: str) -> List[Dict[str, Any]]:
    if not md_text.strip():
        return []

    if parse_technique_md is not None:
        try:
            return parse_technique_md(md_text)
        except Exception:
            pass

    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return []
    data_rows = rows[1:] if _has_header(rows[0]) else rows
    return [_row_to_record_positional(row) for row in data_rows if row and row[0].strip()]


def _direct_line_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    anchors = {_fold(value) for value in cand_variants if value and not value.startswith("#")}
    if not anchors:
        return None

    for raw in (md_text or "").splitlines():
        line = raw.rstrip()
        if not line or "," not in line:
            continue
        first = line.split(",", 1)[0].strip()
        if _fold(first) in anchors:
            return _row_to_record_positional(_split_row_limited(line))
    return None


def _csv_exact_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return None

    data_rows = rows[1:] if _has_header(rows[0]) else rows
    cand_folded = {_fold(item) for item in cand_variants}
    cand_lite = {_lite(item) for item in cand_variants}

    for row in data_rows:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if _fold(name) in cand_folded or _lite(name) in cand_lite:
            return _row_to_record_positional(row)
    return None


def _record_field_matches(rec: Dict[str, Any], field_name: str, cand_variants: List[str]) -> bool:
    value = rec.get(field_name) or ""
    if not str(value).strip():
        return False

    candidate_folded = {_fold(item) for item in cand_variants if item}
    candidate_lite = {_lite(item) for item in cand_variants if item}
    field_variants = _candidate_variants(str(value))
    return any(_fold(item) in candidate_folded or _lite(item) in candidate_lite for item in field_variants)


def _lookup_record_from_records(
    records: List[Dict[str, Any]],
    cand_variants: List[str],
) -> Optional[Dict[str, Any]]:
    if not records:
        return None

    for rec in records:
        if _record_field_matches(rec, "name", cand_variants):
            return rec

    for field_name in ("translation", "japanese"):
        for rec in records:
            if _record_field_matches(rec, field_name, cand_variants):
                return rec

    return None


def _normalize_tags(tags_raw: Any) -> List[str]:
    if isinstance(tags_raw, list):
        return [str(item).strip() for item in tags_raw if str(item).strip()]
    if isinstance(tags_raw, str):
        return [item.strip() for item in re.split(r"[|,]", tags_raw) if item.strip()]
    return []


def _record_to_result(
    rec: Dict[str, Any],
    passages: List[Dict[str, Any]],
    *,
    det_path: str,
) -> DeterministicResult:
    return build_result(
        det_path=det_path,
        answer_type="technique",
        facts={
            "technique_name": rec.get("name"),
            "japanese": rec.get("japanese"),
            "translation": rec.get("translation"),
            "type": rec.get("type"),
            "rank_context": rec.get("rank"),
            "primary_focus": rec.get("primary_focus"),
            "safety": rec.get("safety"),
            "partner_required": rec.get("partner_required"),
            "solo": rec.get("solo"),
            "tags": _normalize_tags(rec.get("tags")),
            "definition": rec.get("description"),
        },
        passages=passages,
        preferred_sources=["Technique Descriptions.md"],
        confidence=0.95,
        display_hints={"explain": True},
        followup_suggestions=["Ask how it compares with another technique if you want a side-by-side answer."],
    )


def _exact_record_lookup(
    md_text: str,
    records: List[Dict[str, Any]],
    cand_variants: List[str],
) -> Optional[Dict[str, Any]]:
    rec = _direct_line_lookup(md_text, cand_variants)
    if rec:
        return rec

    rec = _csv_exact_lookup(md_text, cand_variants)
    if rec:
        return rec

    return _lookup_record_from_records(records, cand_variants)


def try_answer_technique(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    if not _looks_like_technique_q(question):
        return None

    md_text = _gather_full_technique_text(passages)
    if not md_text.strip():
        return None

    records = _parse_records(md_text)
    cand_raw = _extract_candidate(question)
    if not cand_raw:
        return None

    canonical_name = canonical_from_query(question) if _has_direct_technique_trigger(question) else None
    if canonical_name:
        rec = _exact_record_lookup(md_text, records, _candidate_variants(canonical_name))
        if rec:
            return _record_to_result(rec, passages, det_path="technique/single")
        return None

    rec = _exact_record_lookup(md_text, records, _candidate_variants(cand_raw))
    if rec:
        return _record_to_result(rec, passages, det_path="technique/core")

    # Direct single-technique questions should fail safely rather than guess a nearby row.
    return None
