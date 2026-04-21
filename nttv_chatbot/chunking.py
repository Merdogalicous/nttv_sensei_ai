from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

from nttv_chatbot.document_parsing import StructuredDocument, StructuredDocumentElement


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+", re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"[ \t]+")
LINE_WHITESPACE_RE = re.compile(r"[ \t]*\n[ \t]*")
RANK_PATTERNS = [
    "10th kyu",
    "9th kyu",
    "8th kyu",
    "7th kyu",
    "6th kyu",
    "5th kyu",
    "4th kyu",
    "3rd kyu",
    "2nd kyu",
    "1st kyu",
    "shodan",
    "nidan",
    "sandan",
    "yondan",
    "godan",
]
RANK_LABELS = {
    "10th kyu": "10th Kyu",
    "9th kyu": "9th Kyu",
    "8th kyu": "8th Kyu",
    "7th kyu": "7th Kyu",
    "6th kyu": "6th Kyu",
    "5th kyu": "5th Kyu",
    "4th kyu": "4th Kyu",
    "3rd kyu": "3rd Kyu",
    "2nd kyu": "2nd Kyu",
    "1st kyu": "1st Kyu",
    "shodan": "Shodan",
    "nidan": "Nidan",
    "sandan": "Sandan",
    "yondan": "Yondan",
    "godan": "Godan",
}
SCHOOL_TAGS = {
    "Togakure-ryu": ["togakure ryu", "togakure-ryu"],
    "Gyokko-ryu": ["gyokko ryu", "gyokko-ryu"],
    "Koto-ryu": ["koto ryu", "koto-ryu"],
    "Shinden Fudo-ryu": ["shinden fudo ryu", "shinden fudo-ryu"],
    "Kukishinden-ryu": ["kukishinden ryu", "kukishinden-ryu"],
    "Takagi Yoshin-ryu": ["takagi yoshin ryu", "takagi yoshin-ryu"],
    "Gikan-ryu": ["gikan ryu", "gikan-ryu"],
    "Gyokushin-ryu": ["gyokushin ryu", "gyokushin-ryu"],
    "Kumogakure-ryu": ["kumogakure ryu", "kumogakure-ryu"],
}
WEAPON_TAGS = {
    "Hanbo": ["hanbo"],
    "Rokushakubo": ["rokushakubo", "rokushaku bo"],
    "Katana": ["katana"],
    "Tanto": ["tanto"],
    "Shoto": ["shoto"],
    "Kusari Fundo": ["kusari fundo"],
    "Kyoketsu Shoge": ["kyoketsu shoge"],
    "Shuko": ["shuko"],
    "Jutte": ["jutte", "jitte"],
    "Tessen": ["tessen"],
    "Kunai": ["kunai"],
    "Shuriken": ["shuriken", "senban", "shaken"],
}
TECHNIQUE_HINTS = (
    " gyaku",
    " dori",
    " ori",
    " kudaki",
    " gatame",
    " otoshi",
    " nage",
    " jime",
    " keri",
    " ken",
    " tsuki",
    " no kata",
    " kamae",
    " waza",
)


@dataclass(frozen=True)
class ChunkingSettings:
    target_tokens: int = 180
    max_tokens: int = 240
    overlap_tokens: int = 40
    min_tokens: int = 60

    @classmethod
    def from_env(cls) -> "ChunkingSettings":
        settings = cls(
            target_tokens=_int_env("CHUNK_TARGET_TOKENS", 180),
            max_tokens=_int_env("CHUNK_MAX_TOKENS", 240),
            overlap_tokens=_int_env("CHUNK_OVERLAP_TOKENS", 40),
            min_tokens=_int_env("CHUNK_MIN_TOKENS", 60),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("CHUNK_TARGET_TOKENS must be > 0.")
        if self.max_tokens < self.target_tokens:
            raise ValueError("CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS.")
        if self.overlap_tokens < 0:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be >= 0.")
        if self.min_tokens <= 0:
            raise ValueError("CHUNK_MIN_TOKENS must be > 0.")
        if self.min_tokens > self.max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be <= CHUNK_MAX_TOKENS.")


@dataclass
class ChunkUnit:
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    element_type: str
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_token_count(self) -> int:
        return estimate_token_count(self.text)


@dataclass
class ChunkSection:
    source_file: str
    file_type: str
    parser: str
    heading_path: list[str]
    section_title: Optional[str]
    units: list[ChunkUnit] = field(default_factory=list)
    element_types: list[str] = field(default_factory=list)


def build_chunks(
    document: StructuredDocument,
    *,
    source_file: Optional[str] = None,
    settings: Optional[ChunkingSettings] = None,
) -> list[dict[str, Any]]:
    settings = settings or ChunkingSettings.from_env()
    settings.validate()

    resolved_source = source_file or document.source_path
    sections = _build_sections(document, resolved_source)

    chunks: list[dict[str, Any]] = []
    for section_index, section in enumerate(sections):
        chunks.extend(_chunk_section(section, section_index=section_index, settings=settings))

    return [chunk for chunk in chunks if (chunk.get("text") or "").strip()]


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(TOKEN_PATTERN.findall(text))


def _build_sections(document: StructuredDocument, source_file: str) -> list[ChunkSection]:
    sections: list[ChunkSection] = []
    current: Optional[ChunkSection] = None

    for element in document.elements:
        heading_path = [_clean_heading_text(part) for part in element.heading_path if _clean_heading_text(part)]
        element_type = (element.element_type or "paragraph").strip().lower()
        cleaned_text = _clean_text_block(element.text)

        if element_type == "heading":
            if current and current.units:
                sections.append(current)
            section_title = heading_path[-1] if heading_path else _clean_heading_text(cleaned_text)
            current = ChunkSection(
                source_file=source_file,
                file_type=document.file_type,
                parser=document.parser_name,
                heading_path=heading_path or ([section_title] if section_title else []),
                section_title=section_title,
            )
            continue

        if not cleaned_text:
            continue

        if current is None or tuple(current.heading_path) != tuple(heading_path):
            if current and current.units:
                sections.append(current)
            current = ChunkSection(
                source_file=source_file,
                file_type=document.file_type,
                parser=document.parser_name,
                heading_path=heading_path,
                section_title=heading_path[-1] if heading_path else None,
            )

        paragraph_units = _build_units_for_element(cleaned_text, element, current.heading_path)
        if not paragraph_units:
            continue

        current.units.extend(paragraph_units)
        current.element_types.append(element_type)

    if current and current.units:
        sections.append(current)

    return [section for section in sections if section.units]


def _build_units_for_element(
    cleaned_text: str,
    element: StructuredDocumentElement,
    heading_path: list[str],
) -> list[ChunkUnit]:
    paragraphs = _split_into_paragraphs(cleaned_text, element_type=element.element_type)
    units: list[ChunkUnit] = []

    for paragraph in paragraphs:
        normalized = _strip_duplicate_heading(paragraph, heading_path)
        if not normalized:
            continue
        units.append(
            ChunkUnit(
                text=normalized,
                page_start=element.page_start,
                page_end=element.page_end,
                element_type=(element.element_type or "paragraph").strip().lower(),
                raw_metadata=dict(element.raw_metadata),
            )
        )

    return _merge_tiny_units(units)


def _chunk_section(
    section: ChunkSection,
    *,
    section_index: int,
    settings: ChunkingSettings,
) -> list[dict[str, Any]]:
    prefix = _build_heading_prefix(section.heading_path)
    prefix_tokens = estimate_token_count(prefix) if prefix else 0
    body_max_tokens = max(settings.max_tokens - prefix_tokens, max(settings.min_tokens, 1))

    units = _split_large_units(section.units, body_max_tokens)
    if not units:
        return []

    drafts: list[list[ChunkUnit]] = []
    current_units: list[ChunkUnit] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = unit.estimated_token_count
        if not current_units:
            current_units = [unit]
            current_tokens = unit_tokens
            continue

        projected_tokens = prefix_tokens + current_tokens + unit_tokens
        should_flush = False
        if projected_tokens > settings.max_tokens:
            should_flush = True
        elif projected_tokens > settings.target_tokens and (prefix_tokens + current_tokens) >= settings.min_tokens:
            should_flush = True

        if should_flush:
            drafts.append(current_units)
            overlap_units = _select_overlap_units(current_units, settings.overlap_tokens)
            current_units = list(overlap_units)
            current_tokens = sum(item.estimated_token_count for item in current_units)

            while current_units and (prefix_tokens + current_tokens + unit_tokens) > settings.max_tokens:
                current_units.pop(0)
                current_tokens = sum(item.estimated_token_count for item in current_units)

        current_units.append(unit)
        current_tokens = sum(item.estimated_token_count for item in current_units)

    if current_units:
        drafts.append(current_units)

    drafts = _merge_small_tail_chunk(drafts, settings=settings)

    chunks: list[dict[str, Any]] = []
    for local_index, draft_units in enumerate(drafts):
        chunk = _build_chunk_dict(
            section,
            draft_units,
            section_index=section_index,
            chunk_index=local_index,
            prefix=prefix,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


def _build_chunk_dict(
    section: ChunkSection,
    units: list[ChunkUnit],
    *,
    section_index: int,
    chunk_index: int,
    prefix: str,
) -> Optional[dict[str, Any]]:
    if not units:
        return None

    body_text = "\n\n".join(unit.text for unit in units if unit.text.strip()).strip()
    if not body_text:
        return None

    text = body_text
    if prefix and not body_text.startswith(prefix):
        text = f"{prefix}\n\n{body_text}".strip()

    page_start = _min_page(unit.page_start for unit in units)
    page_end = _max_page(unit.page_end for unit in units)
    rank_tag = _detect_rank_tag(section, text)
    school_tag = _detect_school_tag(section, text)
    weapon_tag = _detect_weapon_tag(section, text)
    technique_tag = _detect_technique_tag(
        section,
        text,
        school_tag=school_tag,
        weapon_tag=weapon_tag,
        rank_tag=rank_tag,
    )
    priority = _priority_for_source(section.source_file)
    priority_bucket = _priority_bucket(priority)
    content_type = _content_type_for_chunk(
        section,
        units,
        rank_tag=rank_tag,
        school_tag=school_tag,
        weapon_tag=weapon_tag,
        technique_tag=technique_tag,
    )
    estimated_token_count = estimate_token_count(text)
    char_count = len(text)
    section_title = section.section_title or (section.heading_path[-1] if section.heading_path else None)

    meta: dict[str, Any] = {
        "chunk_id": _stable_chunk_id(
            source_file=section.source_file,
            heading_path=section.heading_path,
            page_start=page_start,
            page_end=page_end,
            text=text,
        ),
        "source": section.source_file,
        "source_file": section.source_file,
        "file_type": section.file_type,
        "parser": section.parser,
        "page": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "heading_path": list(section.heading_path),
        "section_title": section_title,
        "content_type": content_type,
        "rank_tag": rank_tag,
        "school_tag": school_tag,
        "weapon_tag": weapon_tag,
        "technique_tag": technique_tag,
        "priority": priority,
        "priority_bucket": priority_bucket,
        "char_count": char_count,
        "estimated_token_count": estimated_token_count,
        "raw_metadata": {
            "section_index": section_index,
            "chunk_index": chunk_index,
            "unit_count": len(units),
            "element_types": [unit.element_type for unit in units],
            "unit_pages": [
                {"page_start": unit.page_start, "page_end": unit.page_end}
                for unit in units
            ],
        },
    }

    return {
        "chunk_id": meta["chunk_id"],
        "text": text,
        "source": section.source_file,
        "source_file": section.source_file,
        "file_type": section.file_type,
        "parser": section.parser,
        "page_start": page_start,
        "page_end": page_end,
        "heading_path": list(section.heading_path),
        "section_title": section_title,
        "content_type": content_type,
        "rank_tag": rank_tag,
        "school_tag": school_tag,
        "weapon_tag": weapon_tag,
        "technique_tag": technique_tag,
        "priority_bucket": priority_bucket,
        "char_count": char_count,
        "estimated_token_count": estimated_token_count,
        "meta": meta,
    }


def _merge_tiny_units(units: list[ChunkUnit]) -> list[ChunkUnit]:
    if not units:
        return []

    merged: list[ChunkUnit] = []
    small_threshold = 18

    for unit in units:
        if not unit.text.strip():
            continue

        if merged and unit.estimated_token_count <= small_threshold:
            previous = merged.pop()
            merged.append(
                ChunkUnit(
                    text=f"{previous.text}\n\n{unit.text}".strip(),
                    page_start=_min_page([previous.page_start, unit.page_start]),
                    page_end=_max_page([previous.page_end, unit.page_end]),
                    element_type=previous.element_type if previous.element_type == unit.element_type else "mixed",
                    raw_metadata={},
                )
            )
            continue

        merged.append(unit)

    return merged


def _split_large_units(units: list[ChunkUnit], max_tokens: int) -> list[ChunkUnit]:
    output: list[ChunkUnit] = []
    for unit in units:
        if unit.estimated_token_count <= max_tokens:
            output.append(unit)
            continue
        output.extend(_split_unit_by_sentences(unit, max_tokens))
    return output


def _split_unit_by_sentences(unit: ChunkUnit, max_tokens: int) -> list[ChunkUnit]:
    if max_tokens <= 0:
        return [unit]

    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(unit.text) if segment.strip()]
    if len(sentences) <= 1:
        return _split_unit_by_words(unit, max_tokens)

    chunks: list[ChunkUnit] = []
    current_parts: list[str] = []

    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        current_text = " ".join(current_parts).strip()
        current_tokens = estimate_token_count(current_text)

        if current_parts and (current_tokens + sentence_tokens) > max_tokens:
            chunks.append(
                ChunkUnit(
                    text=" ".join(current_parts).strip(),
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    element_type=unit.element_type,
                    raw_metadata=dict(unit.raw_metadata),
                )
            )
            current_parts = [sentence]
            continue

        if not current_parts and sentence_tokens > max_tokens:
            chunks.extend(
                _split_unit_by_words(
                    ChunkUnit(
                        text=sentence,
                        page_start=unit.page_start,
                        page_end=unit.page_end,
                        element_type=unit.element_type,
                        raw_metadata=dict(unit.raw_metadata),
                    ),
                    max_tokens,
                )
            )
            continue

        current_parts.append(sentence)

    if current_parts:
        chunks.append(
            ChunkUnit(
                text=" ".join(current_parts).strip(),
                page_start=unit.page_start,
                page_end=unit.page_end,
                element_type=unit.element_type,
                raw_metadata=dict(unit.raw_metadata),
            )
        )

    return [chunk for chunk in chunks if chunk.text.strip()]


def _split_unit_by_words(unit: ChunkUnit, max_tokens: int) -> list[ChunkUnit]:
    words = unit.text.split()
    if not words:
        return []

    chunks: list[ChunkUnit] = []
    current_words: list[str] = []

    for word in words:
        current_words.append(word)
        if estimate_token_count(" ".join(current_words)) >= max_tokens:
            chunks.append(
                ChunkUnit(
                    text=" ".join(current_words).strip(),
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    element_type=unit.element_type,
                    raw_metadata=dict(unit.raw_metadata),
                )
            )
            current_words = []

    if current_words:
        chunks.append(
            ChunkUnit(
                text=" ".join(current_words).strip(),
                page_start=unit.page_start,
                page_end=unit.page_end,
                element_type=unit.element_type,
                raw_metadata=dict(unit.raw_metadata),
            )
        )

    return chunks


def _select_overlap_units(units: list[ChunkUnit], overlap_tokens: int) -> list[ChunkUnit]:
    if overlap_tokens <= 0:
        return []

    selected: list[ChunkUnit] = []
    total = 0

    for unit in reversed(units):
        unit_tokens = unit.estimated_token_count
        if selected and total >= overlap_tokens:
            break
        selected.insert(0, unit)
        total += unit_tokens

    return selected


def _merge_small_tail_chunk(
    drafts: list[list[ChunkUnit]],
    *,
    settings: ChunkingSettings,
) -> list[list[ChunkUnit]]:
    if len(drafts) < 2:
        return drafts

    tail = drafts[-1]
    tail_text = "\n\n".join(unit.text for unit in tail).strip()
    if estimate_token_count(tail_text) >= settings.min_tokens:
        return drafts

    head = drafts[-2]
    head_text = "\n\n".join(unit.text for unit in head).strip()
    combined_tokens = estimate_token_count(f"{head_text}\n\n{tail_text}".strip())
    if combined_tokens > settings.max_tokens + settings.overlap_tokens:
        return drafts

    deduped_tail = list(tail)
    while deduped_tail and head and deduped_tail[0].text == head[-1].text:
        deduped_tail.pop(0)

    if not deduped_tail:
        return drafts[:-1]

    drafts[-2] = head + deduped_tail
    return drafts[:-1]


def _build_heading_prefix(heading_path: list[str]) -> str:
    clean_parts = [part.strip() for part in heading_path if part and part.strip()]
    if not clean_parts:
        return ""
    return " > ".join(clean_parts)


def _split_into_paragraphs(text: str, *, element_type: str) -> list[str]:
    normalized_type = (element_type or "paragraph").strip().lower()
    if normalized_type in {"table", "code"}:
        return [text.strip()]

    parts = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    if parts:
        return parts
    return [text.strip()]


def _clean_text_block(text: str) -> str:
    if not text:
        return ""

    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned_lines: list[str] = []
    blank_open = False

    for line in lines:
        if not line:
            if cleaned_lines and not blank_open:
                cleaned_lines.append("")
                blank_open = True
            continue
        cleaned_lines.append(line)
        blank_open = False

    cleaned = "\n".join(cleaned_lines).strip()
    return LINE_WHITESPACE_RE.sub("\n", cleaned)


def _strip_duplicate_heading(text: str, heading_path: list[str]) -> str:
    cleaned = _clean_text_block(text)
    if not cleaned or not heading_path:
        return cleaned

    lines = cleaned.splitlines()
    if not lines:
        return ""

    first_line = lines[0].strip().lower()
    candidates = {
        heading_path[-1].strip().lower(),
        " > ".join(heading_path).strip().lower(),
    }
    if first_line in candidates and len(lines) > 1:
        return "\n".join(lines[1:]).strip()
    return cleaned


def _clean_heading_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def _priority_for_source(source_file: str) -> int:
    lower_source = source_file.lower()
    if "glossary" in lower_source:
        return 3
    if "rank" in lower_source:
        return 3
    if "technique description" in lower_source or "technique_descriptions" in lower_source:
        return 3
    if "kihon" in lower_source or "sanshin" in lower_source:
        return 2
    return 1


def _priority_bucket(priority: int) -> str:
    return {3: "p1", 2: "p2", 1: "p3"}.get(priority, "p3")


def _content_type_for_chunk(
    section: ChunkSection,
    units: list[ChunkUnit],
    *,
    rank_tag: Optional[str],
    school_tag: Optional[str],
    weapon_tag: Optional[str],
    technique_tag: Optional[str],
) -> str:
    source_lower = section.source_file.lower()
    if rank_tag:
        return "rank_requirement"
    if weapon_tag:
        return "weapon_profile"
    if school_tag:
        return "school_profile"
    if technique_tag:
        return "technique_description"

    element_types = {unit.element_type for unit in units}
    if len(element_types) == 1:
        only = next(iter(element_types))
        if only in {"table", "code", "list_item"}:
            return only

    if "glossary" in source_lower:
        return "glossary_entry"
    if "leadership" in source_lower:
        return "leadership_note"
    return "section"


def _detect_rank_tag(section: ChunkSection, text: str) -> Optional[str]:
    haystack = _fold(" ".join(section.heading_path + [section.section_title or "", text]))
    for pattern in RANK_PATTERNS:
        if _fold(pattern) in haystack:
            return RANK_LABELS[pattern]
    return None


def _detect_school_tag(section: ChunkSection, text: str) -> Optional[str]:
    haystack = _fold(" ".join(section.heading_path + [section.section_title or "", text]))
    for canonical, aliases in SCHOOL_TAGS.items():
        if any(_fold(alias) in haystack for alias in aliases):
            return canonical
    return None


def _detect_weapon_tag(section: ChunkSection, text: str) -> Optional[str]:
    haystack = _fold(" ".join(section.heading_path + [section.section_title or "", text]))
    for canonical, aliases in WEAPON_TAGS.items():
        if any(_fold(alias) in haystack for alias in aliases):
            return canonical
    return None


def _detect_technique_tag(
    section: ChunkSection,
    text: str,
    *,
    school_tag: Optional[str],
    weapon_tag: Optional[str],
    rank_tag: Optional[str],
) -> Optional[str]:
    if school_tag or weapon_tag or rank_tag:
        return None

    source_lower = _fold(section.source_file)
    candidates = [part for part in section.heading_path if part] + [section.section_title or ""]
    if text:
        first_line = text.splitlines()[0].strip()
        if 0 < len(first_line) <= 80:
            candidates.append(first_line)

    folded_text = _fold(text)
    for candidate in candidates:
        folded = _fold(candidate)
        if not folded:
            continue
        if any(keyword in folded for keyword in TECHNIQUE_HINTS):
            return candidate.strip()
        if "technique" in source_lower and len(candidate.split()) <= 8:
            return candidate.strip()
        if any(marker in folded_text for marker in ["translation:", "definition:", "description:"]) and len(candidate.split()) <= 8:
            return candidate.strip()
    return None


def _stable_chunk_id(
    *,
    source_file: str,
    heading_path: list[str],
    page_start: Optional[int],
    page_end: Optional[int],
    text: str,
) -> str:
    seed = json.dumps(
        {
            "source_file": source_file,
            "heading_path": heading_path,
            "page_start": page_start,
            "page_end": page_end,
            "text": text,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold()
    normalized = normalized.replace("-", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _min_page(values: Any) -> Optional[int]:
    pages = [value for value in values if value is not None]
    return min(pages) if pages else None


def _max_page(values: Any) -> Optional[int]:
    pages = [value for value in values if value is not None]
    return max(pages) if pages else None


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())
