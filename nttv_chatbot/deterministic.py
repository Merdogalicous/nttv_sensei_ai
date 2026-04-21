from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SourceRef:
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    heading_path: list[str] = field(default_factory=list)
    chunk_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class DeterministicResult:
    answered: bool
    det_path: str
    answer_type: str
    facts: dict[str, Any] = field(default_factory=dict)
    source_refs: list[SourceRef] = field(default_factory=list)
    confidence: float = 1.0
    display_hints: dict[str, Any] = field(default_factory=dict)
    followup_suggestions: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.answered

    def to_dict(self) -> dict[str, Any]:
        return {
            "answered": self.answered,
            "det_path": self.det_path,
            "answer_type": self.answer_type,
            "facts": dict(self.facts),
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "confidence": self.confidence,
            "display_hints": dict(self.display_hints),
            "followup_suggestions": list(self.followup_suggestions),
        }


def build_result(
    *,
    det_path: str,
    answer_type: str,
    facts: dict[str, Any],
    passages: Optional[list[dict[str, Any]]] = None,
    preferred_sources: Optional[list[str]] = None,
    source_refs: Optional[list[SourceRef]] = None,
    confidence: float = 0.95,
    display_hints: Optional[dict[str, Any]] = None,
    followup_suggestions: Optional[list[str]] = None,
) -> DeterministicResult:
    refs = list(source_refs or [])
    if not refs:
        refs = source_refs_from_passages(passages or [], preferred_sources=preferred_sources)
    if not refs and preferred_sources:
        refs = [SourceRef(source=preferred_sources[0])]

    return DeterministicResult(
        answered=True,
        det_path=det_path,
        answer_type=answer_type,
        facts=dict(facts),
        source_refs=refs,
        confidence=confidence,
        display_hints=dict(display_hints or {}),
        followup_suggestions=list(followup_suggestions or []),
    )


def source_refs_from_passages(
    passages: list[dict[str, Any]],
    *,
    preferred_sources: Optional[list[str]] = None,
    limit: int = 3,
) -> list[SourceRef]:
    preferred = [item.lower() for item in (preferred_sources or []) if item]

    def matches(source_name: str) -> bool:
        if not preferred:
            return True
        source_lower = source_name.lower()
        source_base = os.path.basename(source_lower)
        for item in preferred:
            item_base = os.path.basename(item)
            if item in source_lower or item_base == source_base:
                return True
        return False

    refs: list[SourceRef] = []
    seen: set[tuple[Any, ...]] = set()

    for passage in passages:
        meta = passage.get("meta") or {}
        source = (
            meta.get("source_file")
            or meta.get("source")
            or passage.get("source_file")
            or passage.get("source")
            or ""
        )
        if not source or not matches(source):
            continue

        page_start = (
            meta.get("page_start")
            or passage.get("page_start")
            or meta.get("page")
            or passage.get("page")
        )
        page_end = meta.get("page_end") or passage.get("page_end") or page_start
        heading_path = meta.get("heading_path") or passage.get("heading_path") or []
        chunk_id = meta.get("chunk_id") or passage.get("chunk_id")

        key = (
            source,
            page_start,
            page_end,
            tuple(heading_path),
            chunk_id,
        )
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            SourceRef(
                source=source,
                page_start=page_start,
                page_end=page_end,
                heading_path=list(heading_path),
                chunk_id=chunk_id,
            )
        )
        if len(refs) >= limit:
            break

    return refs
