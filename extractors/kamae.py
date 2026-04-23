from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import re
import unicodedata

from nttv_chatbot.deterministic import DeterministicResult, build_result


EXPECTED_COLS = 12
TECHNIQUE_SOURCE = "Technique Descriptions.md"
RANK_SOURCE = "nttv rank requirements.txt"
WEAPONS_SOURCE = "NTTV Weapons Reference.txt"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _fold(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _looks_like_kamae_question(question: str) -> bool:
    q = _fold(question)
    return "kamae" in q or "stance" in q or "stances" in q


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


def _data_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent / "data"


def _load_file(name: str) -> str:
    path = _data_dir() / name
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def _join_passages_text(passages: List[Dict[str, Any]], source_name: str) -> str:
    chunks: list[str] = []
    for passage in passages:
        src = passage.get("source") or passage.get("meta", {}).get("source") or ""
        text = passage.get("text") or ""
        if text and (_same_source_name(src, source_name) or source_name.lower() in src.lower()):
            chunks.append(text)
    return "\n".join(chunks).strip()


def _source_text(passages: List[Dict[str, Any]], source_name: str) -> str:
    from_passages = _join_passages_text(passages, source_name)
    if from_passages:
        return from_passages
    return _load_file(source_name)


def _split_row_limited(raw: str) -> List[str]:
    parts = raw.split(",", EXPECTED_COLS - 1)
    parts = [part.strip() for part in parts]
    if len(parts) > EXPECTED_COLS:
        head = parts[: EXPECTED_COLS - 1]
        tail = ",".join(parts[EXPECTED_COLS - 1 :])
        parts = head + [tail]
    if len(parts) < EXPECTED_COLS:
        parts += [""] * (EXPECTED_COLS - len(parts))
    return parts


def _iter_csv_lines(md_text: str):
    for raw in (md_text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("```"):
            continue
        if "," in raw:
            yield raw


def _load_kamae_records(passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    text = _source_text(passages, TECHNIQUE_SOURCE)
    out: Dict[str, Dict[str, Any]] = {}
    if not text:
        return out

    for raw in _iter_csv_lines(text):
        row = _split_row_limited(raw)
        rec = {
            "name": row[0],
            "japanese": row[1],
            "translation": row[2],
            "type": row[3],
            "rank": row[4],
            "in_rank": row[5],
            "primary_focus": row[6],
            "safety": row[7],
            "partner_required": row[8],
            "solo": row[9],
            "tags": row[10],
            "description": row[11],
        }
        if _fold(rec["type"]) != "kamae":
            continue

        name = rec["name"] or ""
        out[_fold(name)] = rec

        if "no Kamae" in name:
            short = name.replace("no Kamae", "").strip()
            if short:
                out[_fold(short)] = rec

    return out


def _normalize_tags(tags_raw: Any) -> List[str]:
    if isinstance(tags_raw, str):
        return [item.strip() for item in re.split(r"[|,]", tags_raw) if item.strip()]
    if isinstance(tags_raw, list):
        return [str(item).strip() for item in tags_raw if str(item).strip()]
    return []


def _build_specific_kamae_result(
    rec: Dict[str, Any],
    passages: List[Dict[str, Any]],
) -> DeterministicResult:
    return build_result(
        det_path="kamae/specific",
        answer_type="technique",
        facts={
            "technique_name": rec.get("name"),
            "japanese": rec.get("japanese"),
            "translation": rec.get("translation"),
            "type": rec.get("type"),
            "rank_context": rec.get("rank"),
            "primary_focus": rec.get("primary_focus"),
            "safety": rec.get("safety"),
            "partner_required": None,
            "solo": None,
            "tags": _normalize_tags(rec.get("tags")),
            "definition": rec.get("description"),
        },
        passages=passages,
        preferred_sources=[TECHNIQUE_SOURCE],
        confidence=0.95,
        display_hints={"explain": True},
    )


def _answer_specific_kamae(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[DeterministicResult]:
    records = _load_kamae_records(passages)
    if not records:
        return None

    q = _fold(question)
    for key, rec in records.items():
        if not key:
            continue
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, q):
            return _build_specific_kamae_result(rec, passages)
    return None


def _load_rank_text(passages: List[Dict[str, Any]]) -> str:
    return _source_text(passages, RANK_SOURCE)


def _extract_rank_kamae(rank_label: str, passages: List[Dict[str, Any]]) -> Optional[List[str]]:
    text = _load_rank_text(passages)
    if not text:
        return None

    lines = text.splitlines()
    start_idx = None
    target = _fold(rank_label)
    for idx, raw in enumerate(lines):
        if _fold(raw.strip()) == target:
            start_idx = idx
            break

    if start_idx is None:
        return None

    kamae_line: Optional[str] = None
    for raw in lines[start_idx + 1 :]:
        stripped = raw.strip()
        if not stripped:
            break
        if re.search(r"\b(\d+)(st|nd|rd|th)\s+kyu\b", _fold(stripped)):
            break
        if stripped.startswith("Kamae:"):
            kamae_line = stripped
            break

    if kamae_line is None:
        return None

    after = kamae_line.split(":", 1)[1].strip()
    if not after:
        return []
    return [part.strip() for part in after.split(";") if part.strip()]


def _answer_rank_kamae(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[DeterministicResult]:
    q = _fold(question)
    match = re.search(r"(\d+)(st|nd|rd|th)\s+kyu", q)
    if not match:
        return None

    label = f"{match.group(1)}{match.group(2)} Kyu"
    kamae_list = _extract_rank_kamae(label, passages)
    if kamae_list is None:
        return None

    return build_result(
        det_path="rank/kamae",
        answer_type="rank_kamae",
        facts={
            "rank": label,
            "items": kamae_list,
            "category_label": "kamae",
        },
        passages=passages,
        preferred_sources=[RANK_SOURCE],
        confidence=0.97,
        display_hints={"explain": True},
    )


def _build_weapon_kamae_index(passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    text = _source_text(passages, WEAPONS_SOURCE)
    if not text:
        return {}

    index: Dict[str, Dict[str, Any]] = {}
    current_weapon: Optional[str] = None
    current_aliases: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("[WEAPON]"):
            current_weapon = line[len("[WEAPON]") :].strip()
            current_aliases = [current_weapon]
        elif line.upper().startswith("ALIASES:"):
            alias_str = line.split(":", 1)[1]
            aliases = [alias.strip() for alias in alias_str.split(",") if alias.strip()]
            current_aliases.extend(aliases)
        elif line.upper().startswith("KAMAE:"):
            if not current_weapon:
                continue
            kamae_str = line.split(":", 1)[1]
            kamae = [item.strip() for item in kamae_str.split(",") if item.strip()]
            payload = {
                "weapon_name": current_weapon,
                "kamae": kamae,
            }
            for alias in current_aliases:
                index[_fold(alias)] = payload

    return index


def _answer_weapon_kamae(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[DeterministicResult]:
    idx = _build_weapon_kamae_index(passages)
    if not idx:
        return None

    q = _fold(question)
    best_alias = None
    for alias in idx.keys():
        if alias and alias in q:
            best_alias = alias
            break

    if not best_alias:
        return None

    payload = idx[best_alias]
    return build_result(
        det_path="weapons/kamae",
        answer_type="weapon_profile",
        facts={
            "weapon_name": payload["weapon_name"],
            "weapon_type": "",
            "kamae": list(payload["kamae"]),
            "core_actions": [],
            "rank_context": "",
            "notes": "",
        },
        passages=passages,
        preferred_sources=[WEAPONS_SOURCE],
        confidence=0.95,
        display_hints={"explain": True},
    )


def try_answer_kamae(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[DeterministicResult]:
    if not _looks_like_kamae_question(question):
        return None

    ans = _answer_rank_kamae(question, passages)
    if ans:
        return ans

    ans = _answer_weapon_kamae(question, passages)
    if ans:
        return ans

    ans = _answer_specific_kamae(question, passages)
    if ans:
        return ans

    return None
