from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import re
import unicodedata

from nttv_chatbot.deterministic import DeterministicResult, build_result


TRAINING_SOURCE = "nttv training reference.txt"
BUYU_SOURCE = "What is Buyu.txt"
LEADERSHIP_SOURCE = "Bujinkan Leadership and Wisdom.txt"

HATSUMI_NAME = "Masaaki Hatsumi"
TAKAMATSU_NAME = "Toshitsugu Takamatsu"

HATSUMI_ALIASES = (
    "masaaki hatsumi",
    "hatsumi sensei",
    "dr masaaki hatsumi",
    "hatsumi",
)
TAKAMATSU_ALIASES = (
    "toshitsugu takamatsu",
    "takamatsu sensei",
    "takamatsu soke",
    "takamatsu",
)

_PROFILE_PROMPTS = (
    "who is",
    "who was",
    "tell me about",
    "describe",
)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized).strip().lower()


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


def _source_text(passages: List[Dict[str, Any]], source_name: str) -> str:
    parts: list[str] = []
    source_name_lower = source_name.lower()
    for passage in passages:
        source = (
            passage.get("meta", {}).get("source_file")
            or passage.get("meta", {}).get("source")
            or passage.get("source")
            or ""
        )
        text = passage.get("text") or ""
        if text and (_same_source_name(source, source_name) or source_name_lower in source.lower()):
            parts.append(text)
    if parts:
        return "\n".join(parts)
    return _load_file(source_name)


def _supporting_passages(
    passages: List[Dict[str, Any]],
    source_names: List[str],
) -> List[Dict[str, Any]]:
    selected: list[Dict[str, Any]] = []
    for passage in passages:
        source = (
            passage.get("meta", {}).get("source_file")
            or passage.get("meta", {}).get("source")
            or passage.get("source")
            or ""
        )
        if any(_same_source_name(source, source_name) for source_name in source_names):
            selected.append(passage)
    return selected


def _mentions_alias(question: str, aliases: tuple[str, ...]) -> bool:
    q = _fold(question)
    return any(alias in q for alias in aliases)


def _looks_like_profile_prompt(question: str) -> bool:
    q = _fold(question)
    return any(token in q for token in _PROFILE_PROMPTS)


def _is_hatsumi_teacher_question(question: str) -> bool:
    q = _fold(question)
    if not any(alias in q for alias in HATSUMI_ALIASES):
        return False
    return (
        "who taught" in q
        or "teacher of" in q
        or "teacher for" in q
        or "hatsumi's teacher" in q
        or "hatsumis teacher" in q
        or "who was hatsumi's teacher" in q
        or "who was hatsumis teacher" in q
    )


def _takamatsu_support_is_available(passages: List[Dict[str, Any]]) -> bool:
    training = _fold(_source_text(passages, TRAINING_SOURCE))
    buyu = _fold(_source_text(passages, BUYU_SOURCE))
    has_name = TAKAMATSU_NAME.lower() in training or TAKAMATSU_NAME.lower() in buyu
    has_teacher_link = "met hatsumi yoshiaki" in training or "inherited by dr. hatsumi from toshitsugu takamatsu" in training
    has_previous_master_link = "previous grandmaster" in buyu or "late takamatsu soke" in buyu
    return has_name and (has_teacher_link or has_previous_master_link)


def _hatsumi_support_is_available(passages: List[Dict[str, Any]]) -> bool:
    training = _fold(_source_text(passages, TRAINING_SOURCE))
    leadership = _fold(_source_text(passages, LEADERSHIP_SOURCE))
    has_name = HATSUMI_NAME.lower() in training or HATSUMI_NAME.lower() in leadership
    has_role = (
        "leader of the bujinkan" in leadership
        or "headed by dr. masaaki hatsumi" in training
        or "34th togakure ryu soke" in training
    )
    return has_name and has_role


def _build_takamatsu_result(
    passages: List[Dict[str, Any]],
    *,
    teacher_focus: bool,
) -> DeterministicResult:
    support_sources = [TRAINING_SOURCE, BUYU_SOURCE]
    support_passages = _supporting_passages(passages, support_sources)
    role = "teacher of Masaaki Hatsumi" if teacher_focus else "Hatsumi's teacher and the previous grandmaster"
    summary = (
        "The material describes Toshitsugu Takamatsu as the late Takamatsu Soke, the previous grandmaster, "
        "and says Dr. Masaaki Hatsumi inherited the nine schools from him after his passing in 1972."
    )
    return build_result(
        det_path="lineage/person",
        answer_type="lineage_person",
        facts={
            "person_name": TAKAMATSU_NAME,
            "role_or_relationship": role,
            "summary": summary,
            "related_person": HATSUMI_NAME,
        },
        passages=support_passages,
        preferred_sources=support_sources,
        confidence=0.95,
        display_hints={"explain": True},
        followup_suggestions=["Ask who Masaaki Hatsumi is if you want the current Bujinkan lineage context."],
    )


def _build_hatsumi_result(passages: List[Dict[str, Any]]) -> DeterministicResult:
    support_sources = [LEADERSHIP_SOURCE, TRAINING_SOURCE]
    support_passages = _supporting_passages(passages, support_sources)
    summary = (
        "The material describes Masaaki Hatsumi as the leader of the Bujinkan overall and the 34th Togakure Ryu Soke, "
        "and credits him with bringing the art to students in the Bujinkan."
    )
    return build_result(
        det_path="lineage/person",
        answer_type="lineage_person",
        facts={
            "person_name": HATSUMI_NAME,
            "role_or_relationship": "leader of the Bujinkan overall and the 34th Togakure Ryu Soke",
            "summary": summary,
            "related_person": TAKAMATSU_NAME,
        },
        passages=support_passages,
        preferred_sources=support_sources,
        confidence=0.95,
        display_hints={"explain": True},
        followup_suggestions=["Ask who taught Hatsumi if you want the prior generation in the lineage."],
    )


def try_answer_lineage_person(
    question: str,
    passages: List[Dict[str, Any]],
) -> Optional[DeterministicResult]:
    if _is_hatsumi_teacher_question(question):
        if not _takamatsu_support_is_available(passages):
            return None
        return _build_takamatsu_result(passages, teacher_focus=True)

    if _mentions_alias(question, TAKAMATSU_ALIASES) and _looks_like_profile_prompt(question):
        if not _takamatsu_support_is_available(passages):
            return None
        return _build_takamatsu_result(passages, teacher_focus=False)

    if _mentions_alias(question, HATSUMI_ALIASES) and _looks_like_profile_prompt(question):
        if not _hatsumi_support_is_available(passages):
            return None
        return _build_hatsumi_result(passages)

    return None
