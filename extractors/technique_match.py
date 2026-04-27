# extractors/technique_match.py
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from .technique_aliases import TECH_ALIASES, expand_with_aliases


def fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("â€“", "-").replace("â€”", "-")
    s = re.sub(r"[^a-z0-9\s\-']", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def technique_name_variants(name: str) -> List[str]:
    """Generate simple matching variants (strip 'no kata', handle hyphens, spacing)."""
    base = fold(name)
    variants = {base}
    variants.add(re.sub(r"\bno kata\b", "", base).strip())
    variants.add(base.replace(" - ", " ").replace("-", " "))
    return [v for v in variants if v]


def _contains_exact_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def is_single_technique_query(q: str) -> bool:
    qf = fold(q)
    intent = any(
        w in qf
        for w in [
            "what is",
            "explain",
            "describe",
            "define",
            "show me",
            "tell me about",
        ]
    )
    if not intent:
        return False
    return canonical_from_query(q) is not None


def canonical_from_query(q: str) -> Optional[str]:
    """Return the canonical technique name if the query mentions one exactly."""
    qf = fold(q)
    if not qf:
        return None

    candidates: list[tuple[int, int, str]] = []
    for canon, aliases in TECH_ALIASES.items():
        exact_variants = technique_name_variants(canon)
        for phrase in [*exact_variants, *[fold(alias) for alias in aliases]]:
            if not phrase:
                continue
            if _contains_exact_phrase(qf, phrase):
                canonical_priority = 1 if phrase in exact_variants else 0
                candidates.append((len(phrase), canonical_priority, canon))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]

    alias_hits = expand_with_aliases(q)
    for hit in alias_hits:
        folded_hit = fold(hit)
        for canon, aliases in TECH_ALIASES.items():
            if folded_hit in technique_name_variants(canon):
                return canon
            if folded_hit in [fold(alias) for alias in aliases]:
                return canon
    return None
