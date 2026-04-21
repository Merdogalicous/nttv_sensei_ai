from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from nttv_chatbot.deterministic import DeterministicResult, build_result


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _lc(text: str) -> str:
    return _norm(text).lower()


def _dedup(seq: List[str]) -> List[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in seq:
        normalized = _norm(item)
        key = normalized.lower()
        if normalized and key not in seen:
            output.append(normalized)
            seen.add(key)
    return output


def _same_source_name(p_source: str, target_name: str) -> bool:
    if not p_source:
        return False
    return os.path.basename(p_source).lower() == os.path.basename(target_name).lower()


_KICK_ALIASES: Dict[str, List[str]] = {
    "zenpo geri": ["Mae Geri", "Front Kick"],
    "mae geri": ["Zenpo Geri", "Front Kick"],
    "front kick": ["Zenpo Geri", "Mae Geri"],
    "sokuho geri": ["Side Kick"],
    "koho geri": ["Back Kick"],
}

_PUNCH_ALIASES: Dict[str, List[str]] = {
    "fudo ken": ["Immovable Fist"],
    "shuto": ["Knife-hand"],
    "shuto uchi": ["Knife-hand Strike"],
    "shikan ken": ["Foreknuckle Fist"],
    "shako ken": ["Claw Hand"],
    "boshi ken": ["Thumb Knuckle Strike"],
    "tsuki": ["Punch"],
    "jodan tsuki": ["High Punch"],
    "gedan tsuki": ["Low Punch"],
    "ken kudaki": ["Fist Crusher"],
    "happa ken": ["Double-Palm Strike"],
    "kikaku ken": ["Headbutt"],
}


def _with_kick_aliases(name: str) -> str:
    key = _lc(name)
    aliases = _KICK_ALIASES.get(key, [])
    if not aliases:
        return _norm(name)
    seen = {_norm(name)}
    alias_clean: list[str] = []
    for alias in aliases:
        cleaned = _norm(alias)
        if cleaned not in seen:
            alias_clean.append(cleaned)
            seen.add(cleaned)
    return f'{_norm(name)} ({"/".join(alias_clean)})' if alias_clean else _norm(name)


def _with_punch_aliases(name: str) -> str:
    key = _lc(name)
    aliases = _PUNCH_ALIASES.get(key, [])
    if not aliases:
        return _norm(name)
    seen = {_norm(name)}
    alias_clean: list[str] = []
    for alias in aliases:
        cleaned = _norm(alias)
        if cleaned not in seen:
            alias_clean.append(cleaned)
            seen.add(cleaned)
    return f'{_norm(name)} ({"/".join(alias_clean)})' if alias_clean else _norm(name)


_RANK_HEADER_RE = re.compile(r"^(?P<hdr>(?:\d+(?:st|nd|rd|th)\s+kyu|shodan))\b", re.IGNORECASE | re.MULTILINE)


def _rank_key_from_question(question: str) -> Optional[str]:
    ql = _lc(question)
    match = re.search(r"\b(\d+)\s*(?:st|nd|rd|th)?\s*kyu\b", ql)
    if match:
        number = match.group(1)
        if number == "1":
            return "1st kyu"
        if number == "2":
            return "2nd kyu"
        if number == "3":
            return "3rd kyu"
        return f"{number}th kyu"
    if "shodan" in ql:
        return "shodan"
    return None


def _display_rank(rank_key: str) -> str:
    match = re.match(r"(\d+)(st|nd|rd|th)\s+kyu", rank_key, flags=re.I)
    if match:
        return f'{match.group(1)}{match.group(2).lower()} Kyu'
    return "Shodan" if rank_key.lower() == "shodan" else rank_key.title()


def _find_rank_text_from_passages(passages: List[Dict[str, Any]]) -> Optional[str]:
    for passage in passages:
        src = passage.get("source") or passage.get("meta", {}).get("source") or ""
        text = passage.get("text", "")
        if text and (_same_source_name(src, "nttv rank requirements.txt") or "nttv rank requirements" in src.lower()):
            return text
    for passage in passages:
        text = passage.get("text") or ""
        if text and ("kyu" in text.lower() and "kamae" in text.lower()):
            return text
    return None


def _extract_rank_block(full_text: str, rank_key: str) -> Optional[str]:
    if not full_text or not rank_key:
        return None
    pattern = re.compile(rf"^(?P<hdr>{re.escape(rank_key)})\b.*$", re.IGNORECASE | re.MULTILINE)
    start_match = pattern.search(full_text)
    if not start_match:
        return None
    start = start_match.start()
    next_match = _RANK_HEADER_RE.search(full_text, pos=start + 1)
    end = next_match.start() if next_match else len(full_text)
    return full_text[start:end].strip()


def _extract_section_lines(block: str, header_label: str) -> List[str]:
    if not block:
        return []
    header_line_re = re.compile(rf"^(?P<header>\s*{re.escape(header_label)})\s*(?P<inline>.*)$", re.IGNORECASE | re.MULTILINE)
    match = header_line_re.search(block)
    if not match:
        return []

    inline = match.group("inline").strip()
    tail = block[match.end() :]
    stop = len(tail)
    next_section = re.search(r"^[A-Za-z0-9].*?:\s*$", tail, re.MULTILINE)
    if next_section:
        stop = min(stop, next_section.start())
    next_rank = _RANK_HEADER_RE.search(tail)
    if next_rank:
        stop = min(stop, next_rank.start())
    body = tail[:stop]

    output: list[str] = []
    if inline:
        output.append(inline)
    output.extend(line.strip() for line in body.splitlines() if _norm(line))
    return output


def _split_items(lines: List[str]) -> List[str]:
    items: list[str] = []
    for line in lines:
        parts = [item.strip(" -•\t") for item in re.split(r"[;,]", line) if item and len(item.strip()) > 1]
        items.extend(parts)
    return [item for item in (_norm(part) for part in items) if item]


def _build_rank_result(
    *,
    det_path: str,
    answer_type: str,
    rank_key: str,
    facts: Dict[str, Any],
    passages: List[Dict[str, Any]],
    confidence: float = 0.95,
) -> DeterministicResult:
    base_facts = {
        "rank": _display_rank(rank_key),
        **facts,
    }
    return build_result(
        det_path=det_path,
        answer_type=answer_type,
        facts=base_facts,
        passages=passages,
        preferred_sources=["nttv rank requirements.txt"],
        confidence=confidence,
        display_hints={"explain": True},
    )


def try_answer_rank_striking(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    ql = _lc(question)
    wants_kicks = any(token in ql for token in ["kick", "kicks", "geri"])
    wants_punches = any(token in ql for token in ["punch", "punches", "tsuki", "ken", "strike", "striking"])
    if not (wants_kicks or wants_punches):
        return None

    cumulative = (
        ("need to know" in ql)
        or any(phrase in ql for phrase in ["need to know by", "up through", "up to", "all kicks for", "everything for", "study list"])
        or re.search(r"\bby\s+\d+(st|nd|rd|th)\s+kyu\b", ql) is not None
    )

    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None

    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None

    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    lines = _extract_section_lines(block, "Striking:")
    if not lines:
        return None

    raw_items = _split_items(lines)
    kicks: list[str] = []
    punches: list[str] = []
    for item in raw_items:
        item_lower = _lc(item)
        if "geri" in item_lower:
            kicks.append(item)
        elif any(token in item_lower for token in ["tsuki", "shuto", "ken", "strike"]):
            punches.append(item)
        else:
            punches.append(item)

    kicks = _dedup(kicks)
    punches = _dedup(punches)
    carry_kicks: list[str] = []

    if cumulative and rank_key != "9th kyu":
        nine_block = _extract_rank_block(rank_text, "9th kyu")
        if nine_block:
            nine_lines = _extract_section_lines(nine_block, "Striking:")
            for item in _split_items(nine_lines):
                if "geri" in _lc(item):
                    carry_kicks.append(item)
        carry_kicks = _dedup([item for item in carry_kicks if _lc(item) not in {_lc(existing) for existing in kicks}])

    kicks_pretty = [_with_kick_aliases(item) for item in kicks]
    punches_pretty = [_with_punch_aliases(item) for item in punches]
    carry_pretty = [_with_kick_aliases(item) for item in carry_kicks]

    if not kicks_pretty and not punches_pretty:
        return None

    return _build_rank_result(
        det_path="rank/striking",
        answer_type="rank_striking",
        rank_key=rank_key,
        facts={
            "kicks": kicks_pretty if wants_kicks else [],
            "strikes": punches_pretty if wants_punches else [],
            "carryover_kicks": carry_pretty if cumulative and wants_kicks else [],
            "cumulative": cumulative,
            "category_label": "striking",
        },
        passages=passages,
        confidence=0.97,
    )


def _build_rank_item_result(
    *,
    question: str,
    passages: List[Dict[str, Any]],
    triggers: List[str],
    section_label: str,
    det_path: str,
    answer_type: str,
    category_label: str,
) -> Optional[DeterministicResult]:
    ql = _lc(question)
    if not any(token in ql for token in triggers):
        return None
    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None
    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None
    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None
    lines = _extract_section_lines(block, section_label)
    if not lines:
        return None
    items = _dedup(_split_items(lines))
    if not items:
        return None
    return _build_rank_result(
        det_path=det_path,
        answer_type=answer_type,
        rank_key=rank_key,
        facts={
            "items": items,
            "category_label": category_label,
        },
        passages=passages,
    )


def try_answer_rank_nage(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    return _build_rank_item_result(
        question=question,
        passages=passages,
        triggers=["nage", "throw", "throws", "nage waza"],
        section_label="Nage waza:",
        det_path="rank/nage",
        answer_type="rank_nage",
        category_label="throws",
    )


def try_answer_rank_jime(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    return _build_rank_item_result(
        question=question,
        passages=passages,
        triggers=["jime", "choke", "chokes", "strangle"],
        section_label="Jime waza:",
        det_path="rank/jime",
        answer_type="rank_jime",
        category_label="chokes",
    )


def try_answer_rank_requirements(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    ql = _lc(question)
    if not any(token in ql for token in ["requirement", "requirements", "what do i need for", "rank checklist"]):
        return None

    rank_key = _rank_key_from_question(question)
    if not rank_key:
        return None
    rank_text = _find_rank_text_from_passages(passages)
    if not rank_text:
        return None
    block = _extract_rank_block(rank_text, rank_key)
    if not block:
        return None

    sections: list[dict[str, Any]] = []

    def add_section(label: str) -> None:
        lines = _extract_section_lines(block, label)
        if not lines:
            return
        if any(sep in " ".join(lines) for sep in [",", ";"]):
            content = ", ".join(_dedup(_split_items(lines)))
        else:
            content = " ".join(lines)
        content = _norm(content)
        if not content:
            return
        sections.append(
            {
                "label": label.rstrip(":"),
                "content": content,
            }
        )

    for label in [
        "Kamae:",
        "Ukemi:",
        "Kaiten:",
        "Taihenjutsu:",
        "Blocking:",
        "Striking:",
        "Kihon Happo:",
        "San Shin no Kata:",
        "Nage waza:",
        "Jime waza:",
        "Kyusho:",
        "Other:",
    ]:
        add_section(label)

    if not sections:
        header_line = re.split(r"\r?\n", block, maxsplit=1)[0].strip()
        sections.append({"label": _display_rank(rank_key), "content": header_line})

    return _build_rank_result(
        det_path="rank/requirements",
        answer_type="rank_requirements",
        rank_key=rank_key,
        facts={
            "sections": sections,
            "category_label": "requirements",
        },
        passages=passages,
        confidence=0.98,
    )


def try_answer_rank_kihon_kata(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    ql = _lc(question)
    if not (("kihon happo" in ql) or ("kihon" in ql and "happo" in ql)):
        return None
    return _build_rank_item_result(
        question=question,
        passages=passages,
        triggers=["kihon happo"],
        section_label="Kihon Happo:",
        det_path="rank/kihon_kata",
        answer_type="rank_kihon_kata",
        category_label="Kihon Happo kata",
    )


def try_answer_rank_sanshin_kata(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    ql = _lc(question)
    if not any(token in ql for token in ["sanshin", "san shin"]):
        return None
    return _build_rank_item_result(
        question=question,
        passages=passages,
        triggers=["sanshin", "san shin"],
        section_label="San Shin no Kata:",
        det_path="rank/sanshin_kata",
        answer_type="rank_sanshin_kata",
        category_label="San Shin no Kata",
    )


def try_answer_rank_ukemi(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    return _build_rank_item_result(
        question=question,
        passages=passages,
        triggers=["ukemi", "roll", "rolls", "breakfall", "breakfalls"],
        section_label="Ukemi:",
        det_path="rank/ukemi",
        answer_type="rank_ukemi",
        category_label="ukemi (rolls and breakfalls)",
    )


def try_answer_rank_taihenjutsu(question: str, passages: List[Dict[str, Any]]) -> Optional[DeterministicResult]:
    return _build_rank_item_result(
        question=question,
        passages=passages,
        triggers=["taihen", "taihenjutsu", "tai sabaki"],
        section_label="Taihenjutsu:",
        det_path="rank/taihenjutsu",
        answer_type="rank_taihenjutsu",
        category_label="Taihenjutsu (body movement)",
    )
