from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result

try:
    from .technique_loader import build_indexes, parse_technique_md
except Exception:  # pragma: no cover - optional compatibility path
    parse_technique_md = None
    build_indexes = None


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


def _gather_full_technique_text(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        src = src_raw.lower()
        if _same_source_name(src_raw, "Technique Descriptions.md") or "technique descriptions" in src:
            parts.append(passage.get("text", ""))
    return "\n".join(parts)


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
    if lowered in {"1", "true", "yes", "y", "✅", "✓", "✔"}:
        return True
    if lowered in {"0", "false", "no", "n", "❌", "✗", "✕"}:
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


def _csv_fallback_lookup(md_text: str, cand_variants: List[str]) -> Optional[Dict[str, Any]]:
    rows = _scan_csv_rows_limited(md_text)
    if not rows:
        return None

    has_header = _has_header(rows[0])
    data_rows = rows[1:] if has_header else rows
    cand_folded = [_fold(item) for item in cand_variants]
    cand_lite = [_lite(item) for item in cand_variants]

    for row in data_rows:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if _fold(name) in cand_folded or _lite(name) in cand_lite:
            return _row_to_record_positional(row)

    best_row: Optional[List[str]] = None
    best_score = 0.0
    target = _fold(cand_variants[0]) if cand_variants else ""
    for row in data_rows:
        name = (row[0] or "").strip()
        if not name:
            continue
        score = SequenceMatcher(None, _fold(name), target).ratio()
        if score > best_score:
            best_row = row
            best_score = score
    if best_row is not None and best_score >= 0.85:
        return _row_to_record_positional(best_row)
    return None


def _record_to_result(
    rec: Dict[str, Any],
    passages: List[Dict[str, Any]],
    *,
    det_path: str,
) -> DeterministicResult:
    tags_raw = rec.get("tags") or ""
    tags = [item.strip() for item in str(tags_raw).split(",") if item.strip()]
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
            "tags": tags,
            "definition": rec.get("description"),
        },
        passages=passages,
        preferred_sources=["Technique Descriptions.md"],
        confidence=0.95,
        display_hints={"explain": True},
        followup_suggestions=["Ask how it compares with another technique if you want a side-by-side answer."],
    )


def try_answer_technique(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    if not _looks_like_technique_q(question):
        return None

    md_text = _gather_full_technique_text(passages)
    if not md_text.strip():
        return None

    ql = _norm_space(question).lower()
    cand_raw = _extract_candidate(ql)
    variants = _candidate_variants(cand_raw)

    rec = _direct_line_lookup(md_text, variants)
    if rec:
        return _record_to_result(rec, passages, det_path="technique/single")

    rec = _csv_fallback_lookup(md_text, variants)
    if rec:
        return _record_to_result(rec, passages, det_path="technique/core")

    if parse_technique_md and build_indexes:
        try:
            records = parse_technique_md(md_text)
            idx = build_indexes(records) if records else None
        except Exception:
            idx = None
        if idx:
            by_name = idx["by_name"]
            by_lower = idx["by_lower"]
            by_fold = idx["by_fold"]
            by_key = idx["by_keylite"]

            for variant in variants:
                key = variant.lower()
                if key in by_lower:
                    return _record_to_result(by_name[by_lower[key]], passages, det_path="technique/core")
                folded_key = _fold(variant)
                if folded_key in by_fold:
                    return _record_to_result(by_name[by_fold[folded_key]], passages, det_path="technique/core")
                lite_key = _lite(variant)
                if lite_key in by_key:
                    return _record_to_result(by_name[by_key[lite_key]], passages, det_path="technique/core")

            cq = _fold(cand_raw)
            best_name: Optional[str] = None
            best_score = 0.0
            for name in by_name.keys():
                score = SequenceMatcher(None, _fold(name), cq).ratio()
                if score > best_score:
                    best_name = name
                    best_score = score
            if best_name and best_score >= 0.80:
                return _record_to_result(by_name[best_name], passages, det_path="technique/core")

    return None
