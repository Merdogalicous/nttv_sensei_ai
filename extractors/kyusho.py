from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result

from .common import dedupe_preserve


def _fold(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _looks_like_kyusho_question(question: str) -> bool:
    q = _fold(question)
    return "kyusho" in q or "pressure point" in q or "pressure points" in q


def _load_full_kyusho_file() -> str:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "KYUSHO.txt",
        here.parent / "KYUSHO.txt",
    ]
    for path in candidates:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8")
        except Exception:
            continue
    return ""


def _gather_kyusho_text(passages: List[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for passage in passages:
        src_raw = passage.get("source") or ""
        src_fold = _fold(src_raw)
        if _same_source_name(src_raw, "KYUSHO.txt") or "kyusho" in src_fold:
            parts.append(passage.get("text", ""))
    full_file = _load_full_kyusho_file()
    if full_file.strip():
        parts.append(full_file)
    return "\n".join(parts)


def _parse_points(text: str) -> Dict[str, str]:
    points: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•")):
            line = line[1:].strip()
        if ":" not in line:
            continue
        name, desc = line.split(":", 1)
        name = name.strip()
        desc = desc.strip()
        if not name:
            continue
        key = _fold(name)
        if key not in points:
            points[key] = desc
    return points


def _match_point_name(question: str, points: Dict[str, str]) -> Optional[str]:
    q = _fold(question)
    for key in points.keys():
        if key and re.search(r"\b" + re.escape(key) + r"\b", q):
            return key
    return None


def try_answer_kyusho(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    if not _looks_like_kyusho_question(question):
        return None

    text = _gather_kyusho_text(passages)
    if not text.strip():
        return None

    points = _parse_points(text)
    if not points:
        return None

    q = _fold(question)
    is_list = "list" in q or ("what" in q and "points" in q)
    if is_list:
        names = dedupe_preserve(list(points.keys()))
        if not names:
            return None
        display_names = [" ".join(word.capitalize() for word in name.split()) for name in names[:20]]
        return build_result(
            det_path="kyusho/list",
            answer_type="kyusho_list",
            facts={
                "point_names": display_names,
            },
            passages=passages,
            preferred_sources=["KYUSHO.txt"],
            confidence=0.94,
            display_hints={"explain": False},
        )

    key = _match_point_name(question, points)
    if key:
        desc = points.get(key, "").strip()
        name_display = " ".join(word.capitalize() for word in key.split())
        return build_result(
            det_path="kyusho/point",
            answer_type="kyusho_point",
            facts={
                "point_name": name_display,
                "description": desc or "(location/description not listed in the provided context)",
            },
            passages=passages,
            preferred_sources=["KYUSHO.txt"],
            confidence=0.96,
            display_hints={"explain": False},
        )

    return None


def try_kyusho(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    return try_answer_kyusho(question, passages)
