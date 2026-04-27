from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result

from .technique_loader import build_indexes, parse_technique_md


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _gather_technique_md(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        if _same_source_name(src_raw, "Technique Descriptions.md") or "technique descriptions" in src_raw.lower():
            text = passage.get("text", "")
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts)

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


def _build_indexes_from_md(md_text: str) -> Optional[Dict[str, Any]]:
    if not md_text.strip():
        return None
    records = parse_technique_md(md_text)
    if not records:
        return None
    return build_indexes(records)


def _resolve_technique_name(cand_raw: str, indexes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not cand_raw:
        return None
    candidate = cand_raw.strip()
    if not candidate:
        return None

    by_name = indexes.get("by_name", {})
    by_lower = indexes.get("by_lower", {})
    lowered = candidate.lower()

    for name, record in by_name.items():
        if name.lower() == lowered:
            return record

    if lowered in by_lower:
        record = by_name.get(by_lower[lowered])
        if record:
            return record

    best_record = None
    best_score = 0.0
    for name, record in by_name.items():
        score = SequenceMatcher(None, lowered, name.lower()).ratio()
        if score > best_score:
            best_record = record
            best_score = score
    if best_record and best_score >= 0.75:
        return best_record
    return None


def _looks_like_diff_question(question: str) -> bool:
    q = question.lower()
    return any(token in q for token in ["difference between", "different from", "diff between", " vs ", "versus", "compare "])


def _extract_pair(question: str) -> Optional[tuple[str, str]]:
    q = question.strip().rstrip("?.! ")
    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+)",
        r"compare\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+versus\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None


def try_answer_technique_diff(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[DeterministicResult]:
    if not _looks_like_diff_question(question):
        return None

    pair = _extract_pair(question)
    if not pair:
        return None
    left_raw, right_raw = pair

    md_text = _gather_technique_md(passages)
    indexes = _build_indexes_from_md(md_text)
    if not indexes:
        return None

    rec1 = _resolve_technique_name(left_raw, indexes)
    rec2 = _resolve_technique_name(right_raw, indexes)
    if not rec1 or not rec2:
        return None

    return build_result(
        det_path="technique/diff",
        answer_type="technique_diff",
        facts={
            "left": {
                "technique_name": rec1.get("name"),
                "translation": rec1.get("translation"),
                "type": rec1.get("type"),
                "rank_context": rec1.get("rank"),
                "primary_focus": rec1.get("primary_focus"),
                "safety": rec1.get("safety"),
                "partner_required": rec1.get("partner_required"),
                "solo": rec1.get("solo"),
                "definition": rec1.get("description"),
            },
            "right": {
                "technique_name": rec2.get("name"),
                "translation": rec2.get("translation"),
                "type": rec2.get("type"),
                "rank_context": rec2.get("rank"),
                "primary_focus": rec2.get("primary_focus"),
                "safety": rec2.get("safety"),
                "partner_required": rec2.get("partner_required"),
                "solo": rec2.get("solo"),
                "definition": rec2.get("description"),
            },
        },
        passages=passages,
        preferred_sources=["Technique Descriptions.md"],
        confidence=0.96,
        display_hints={"explain": True},
        followup_suggestions=["Ask for either technique by itself if you want the standalone definition."],
    )
