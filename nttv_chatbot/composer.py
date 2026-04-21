from __future__ import annotations

import re
from typing import Any

from nttv_chatbot.deterministic import DeterministicResult


def compose_deterministic_answer(
    result: DeterministicResult,
    *,
    style: str = "standard",
    output_format: str = "paragraph",
    explanation_mode: bool = True,
    tone: str = "crisp",
) -> str:
    normalized_style = _normalize_style(style)
    normalized_format = _normalize_output_format(output_format)
    normalized_tone = (tone or "crisp").strip().lower()

    if normalized_format == "bullets":
        body = _compose_bullets(result, style=normalized_style, explanation_mode=explanation_mode)
    else:
        body = _compose_paragraph(result, style=normalized_style, explanation_mode=explanation_mode)

    if normalized_tone == "chatty" and normalized_format == "paragraph":
        body = _apply_chatty_tone(body)
    return body.strip()


def _compose_bullets(
    result: DeterministicResult,
    *,
    style: str,
    explanation_mode: bool,
) -> str:
    facts = result.facts
    answer_type = result.answer_type

    if answer_type == "technique":
        lines = [f'{facts.get("technique_name", "Technique")}:']
        if facts.get("translation"):
            lines.append(f'- Translation: {facts["translation"]}')
        if style != "brief" and facts.get("japanese"):
            lines.append(f'- Japanese: {facts["japanese"]}')
        if style != "brief" and facts.get("type"):
            lines.append(f'- Type: {facts["type"]}')
        if style != "brief" and facts.get("rank_context"):
            lines.append(f'- Rank intro: {facts["rank_context"]}')
        if style != "brief" and facts.get("primary_focus"):
            lines.append(f'- Focus: {facts["primary_focus"]}')
        if style == "full" and facts.get("safety"):
            lines.append(f'- Safety: {facts["safety"]}')
        if style == "full" and facts.get("partner_required") is not None:
            lines.append(f'- Partner required: {_bool_label(facts["partner_required"])}')
        if style == "full" and facts.get("solo") is not None:
            lines.append(f'- Solo: {_bool_label(facts["solo"])}')
        if style == "full" and facts.get("tags"):
            lines.append(f'- Tags: {", ".join(facts["tags"])}')
        if facts.get("definition"):
            lines.append(f'- Definition: {_sentence(facts["definition"])}')
        return "\n".join(lines)

    if answer_type == "technique_diff":
        left = facts.get("left", {})
        right = facts.get("right", {})
        lines = [f'Difference between {left.get("technique_name", "Technique A")} and {right.get("technique_name", "Technique B")}:']
        field_order = ["translation", "type", "rank_context", "primary_focus", "safety", "partner_required", "solo", "definition"]
        if style == "brief":
            field_order = ["translation", "type", "definition"]
        elif style == "standard":
            field_order = ["translation", "type", "rank_context", "primary_focus", "definition"]
        for key in field_order:
            left_value = _comparison_value(left.get(key))
            right_value = _comparison_value(right.get(key))
            if not left_value and not right_value:
                continue
            lines.append(
                f'- {_labelize(key)}: '
                f'{left.get("technique_name", "Technique A")} = {left_value or "—"}; '
                f'{right.get("technique_name", "Technique B")} = {right_value or "—"}'
            )
        return "\n".join(lines)

    if answer_type == "school_profile":
        lines = [f'{facts.get("school_name", "School")}:']
        for key, label in [
            ("translation", "Translation"),
            ("type", "Type"),
            ("focus", "Focus"),
            ("weapons", "Weapons"),
            ("notes", "Notes"),
        ]:
            if key == "weapons" and style == "brief":
                continue
            if key == "notes" and style != "full":
                continue
            if facts.get(key):
                lines.append(f'- {label}: {facts[key]}')
        return "\n".join(lines)

    if answer_type == "school_list":
        lines = [facts.get("list_title", "Schools:")]
        for item in facts.get("school_names", []):
            lines.append(f"- {item}")
        return "\n".join(lines)

    if answer_type == "weapon_profile":
        lines = [f'{facts.get("weapon_name", "Weapon")} weapon profile:']
        for key, label in [
            ("weapon_type", "Type"),
            ("kamae", "Kamae"),
            ("core_actions", "Core actions"),
            ("rank_context", "Ranks"),
            ("notes", "Notes"),
        ]:
            if key == "kamae" and style == "brief":
                continue
            if key == "notes" and style != "full":
                continue
            value = facts.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value = ", ".join(value)
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    if answer_type == "weapon_rank":
        return "\n".join(
            [
                f'{facts.get("weapon_name", "This weapon")}:',
                f'- Rank intro: {facts.get("rank_context", "")}',
            ]
        )

    if answer_type == "weapon_parts":
        lines = [facts.get("title", "Weapon parts:")]
        for item in _limit_items(facts.get("parts", []), style):
            term = item.get("term", "")
            description = item.get("description", "")
            lines.append(f"- {term}: {description}")
        return "\n".join(lines)

    if answer_type == "weapon_classification":
        lines = [facts.get("title", "Classification:")]
        for item in _limit_items(facts.get("items", []), style):
            name = item.get("name", "")
            description = item.get("description", "")
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    if answer_type.startswith("rank_"):
        return _compose_rank_bullets(result, style=style)

    if answer_type == "kihon_happo":
        lines = ["Kihon Happo:"]
        if facts.get("definition"):
            lines.append(f'- Definition: {_sentence(facts["definition"])}')
        if style != "brief" and facts.get("kosshi_items"):
            lines.append(f'- Kosshi Kihon Sanpo: {", ".join(facts["kosshi_items"])}')
        if style != "brief" and facts.get("torite_items"):
            lines.append(f'- Torite Goho: {", ".join(facts["torite_items"])}')
        return "\n".join(lines)

    if answer_type == "sanshin_element":
        lines = [facts.get("element_name", "Sanshin element")]
        if facts.get("english_name"):
            lines.append(f'- English: {facts["english_name"]}')
        if facts.get("summary"):
            lines.append(f'- Summary: {_sentence(facts["summary"])}')
        return "\n".join(lines)

    if answer_type == "sanshin_list":
        lines = [facts.get("title", "Sanshin no Kata:")]
        for item in facts.get("items", []):
            lines.append(f"- {item}")
        return "\n".join(lines)

    if answer_type == "sanshin_overview":
        lines = ["Sanshin no Kata:"]
        if facts.get("summary"):
            lines.append(f'- Summary: {_sentence(facts["summary"])}')
        if style != "brief" and facts.get("items"):
            lines.append(f'- Forms: {", ".join(facts["items"])}')
        return "\n".join(lines)

    if answer_type == "glossary_term":
        return "\n".join(
            [
                f'{facts.get("term", "Term")}:',
                f'- Definition: {_sentence(facts.get("definition", ""))}',
            ]
        )

    if answer_type == "kyusho_point":
        return "\n".join(
            [
                f'{facts.get("point_name", "Kyusho point")}:',
                f'- Description: {_sentence(facts.get("description", ""))}',
            ]
        )

    if answer_type == "kyusho_list":
        lines = ["Kyusho points:"]
        for item in _limit_plain_items(facts.get("point_names", []), style):
            lines.append(f"- {item}")
        return "\n".join(lines)

    if answer_type == "leadership":
        lines = [facts.get("school_name", "School leadership:")]
        lines.append(f'- Current soke: {facts.get("soke_name", "")}')
        return "\n".join(lines)

    return _compose_fallback_bullets(result)


def _compose_paragraph(
    result: DeterministicResult,
    *,
    style: str,
    explanation_mode: bool,
) -> str:
    facts = result.facts
    answer_type = result.answer_type

    if answer_type == "technique":
        sentences: list[str] = []
        name = facts.get("technique_name", "This technique")
        translation = facts.get("translation")
        if translation:
            sentences.append(f'{name} translates as "{translation}".')
        else:
            sentences.append(f"{name} is listed in the technique descriptions.")
        if style != "brief" and facts.get("type"):
            sentences.append(f'It is categorized as {facts["type"]}.')
        if style != "brief" and facts.get("rank_context"):
            sentences.append(f'It is introduced at {facts["rank_context"]}.')
        if style != "brief" and facts.get("primary_focus"):
            sentences.append(f'Its primary focus is {facts["primary_focus"]}.')
        if style == "full" and facts.get("safety"):
            sentences.append(f'Safety notes: {facts["safety"]}.')
        if style == "full" and facts.get("partner_required") is not None:
            sentences.append(f'Partner required: {_bool_label(facts["partner_required"])}.')
        if style == "full" and facts.get("solo") is not None:
            sentences.append(f'Solo practice: {_bool_label(facts["solo"])}.')
        if facts.get("definition"):
            sentences.append(_sentence(facts["definition"]))
        return " ".join(sentences)

    if answer_type == "technique_diff":
        left = facts.get("left", {})
        right = facts.get("right", {})
        left_name = left.get("technique_name", "Technique A")
        right_name = right.get("technique_name", "Technique B")
        segments = [f"Here is a direct comparison between {left_name} and {right_name}."]
        field_order = ["translation", "type", "rank_context", "primary_focus", "definition"]
        if style == "brief":
            field_order = ["translation", "type"]
        elif style == "full":
            field_order = ["translation", "type", "rank_context", "primary_focus", "safety", "partner_required", "solo", "definition"]
        for key in field_order:
            left_value = _comparison_value(left.get(key))
            right_value = _comparison_value(right.get(key))
            if not left_value and not right_value:
                continue
            segments.append(
                f'{_labelize(key)}: {left_name} = {left_value or "—"}; {right_name} = {right_value or "—"}.'
            )
        return " ".join(segments)

    if answer_type == "school_profile":
        school_name = facts.get("school_name", "This school")
        sentences = [f"{school_name} is one of the Bujinkan schools."]
        if facts.get("translation"):
            sentences.append(f'Its translation is "{facts["translation"]}."')
        if facts.get("type"):
            sentences.append(f'It is described as {facts["type"]}.')
        if facts.get("focus"):
            sentences.append(f'Its focus is {facts["focus"]}.')
        if style != "brief" and facts.get("weapons"):
            sentences.append(f'Weapons associated with it include {facts["weapons"]}.')
        if style == "full" and facts.get("notes"):
            sentences.append(f'Notes: {facts["notes"]}.')
        return " ".join(sentences)

    if answer_type == "school_list":
        names = facts.get("school_names", [])
        title = facts.get("list_title", "The Nine Schools of the Bujinkan")
        if style == "brief":
            return f'{title}: {", ".join(names)}.'
        return f'{title} are {_join_human(names)}.'

    if answer_type == "weapon_profile":
        weapon_name = facts.get("weapon_name", "This weapon")
        sentences = []
        if facts.get("weapon_type"):
            sentences.append(f'{weapon_name} is {facts["weapon_type"]}.')
        else:
            sentences.append(f"{weapon_name} appears in the weapons reference.")
        core_actions = facts.get("core_actions") or []
        if core_actions:
            sentences.append(f'Core actions include {_join_human(core_actions)}.')
        if style != "brief" and facts.get("kamae"):
            sentences.append(f'Common kamae are {_join_human(facts["kamae"])}.')
        if style != "brief" and facts.get("rank_context"):
            sentences.append(f'Rank context: {facts["rank_context"]}.')
        if style == "full" and facts.get("notes"):
            sentences.append(f'Notes: {facts["notes"]}.')
        return " ".join(sentences)

    if answer_type == "weapon_rank":
        return f'You first study {facts.get("weapon_name", "this weapon")} at {facts.get("rank_context", "")}.'

    if answer_type == "weapon_parts":
        parts = _limit_items(facts.get("parts", []), style)
        if not parts:
            return facts.get("title", "Weapon parts.")
        rendered = [f'{item.get("term", "")} ({item.get("description", "")})' for item in parts]
        return f'{facts.get("title", "Weapon parts")} include {_join_human(rendered)}.'

    if answer_type == "weapon_classification":
        items = _limit_items(facts.get("items", []), style)
        rendered = [f'{item.get("name", "")} ({item.get("description", "")})' for item in items]
        return f'{facts.get("title", "Classification")} include {_join_human(rendered)}.'

    if answer_type == "rank_striking":
        rank = facts.get("rank", "This rank")
        sentences = []
        kicks = facts.get("kicks") or []
        strikes = facts.get("strikes") or []
        if kicks:
            sentences.append(f'For {rank}, the kicks are {_join_human(kicks)}.')
        if facts.get("carryover_kicks"):
            sentences.append(f'Carryover foundational kicks are {_join_human(facts["carryover_kicks"])}.')
        if strikes:
            sentences.append(f'For {rank}, the strikes are {_join_human(strikes)}.')
        return " ".join(sentences)

    if answer_type in {
        "rank_nage",
        "rank_jime",
        "rank_ukemi",
        "rank_taihenjutsu",
        "rank_kihon_kata",
        "rank_sanshin_kata",
    }:
        rank = facts.get("rank", "This rank")
        label = facts.get("category_label", "items")
        items = facts.get("items", [])
        if style == "brief":
            return f'{rank} {label}: {_join_human(items)}.'
        return f'For {rank}, the {label} are {_join_human(items)}.'

    if answer_type == "rank_requirements":
        rank = facts.get("rank", "This rank")
        sections = facts.get("sections", [])
        if style == "brief":
            names = [section.get("label", "") for section in sections]
            return f'{rank} requirements cover {_join_human(names)}.'
        fragments = []
        for section in sections:
            label = section.get("label", "")
            content = section.get("content", "")
            if label and content:
                fragments.append(f"{label}: {content}")
        if style == "full":
            return f'{rank} requirements are as follows. ' + " ".join(_sentence(fragment) for fragment in fragments)
        return f'{rank} requirements include ' + "; ".join(fragments) + "."

    if answer_type == "kihon_happo":
        definition = _sentence(facts.get("definition", ""))
        sentences = [definition] if definition else []
        if style != "brief" and facts.get("kosshi_items"):
            sentences.append(f'Kosshi Kihon Sanpo includes {_join_human(facts["kosshi_items"])}.')
        if style != "brief" and facts.get("torite_items"):
            sentences.append(f'Torite Goho includes {_join_human(facts["torite_items"])}.')
        return " ".join(sentences)

    if answer_type == "sanshin_element":
        name = facts.get("element_name", "This form")
        english_name = facts.get("english_name")
        summary = facts.get("summary", "")
        if english_name:
            return f'{name} is the {english_name}. {_sentence(summary)}'
        return _sentence(summary)

    if answer_type == "sanshin_list":
        return f'{facts.get("title", "Sanshin no Kata")} consists of {_join_human(facts.get("items", []))}.'

    if answer_type == "sanshin_overview":
        summary = _sentence(facts.get("summary", ""))
        items = facts.get("items", [])
        if style == "brief":
            return summary
        return f'{summary} The five forms are {_join_human(items)}.'

    if answer_type == "glossary_term":
        return f'{facts.get("term", "This term")} means {_sentence(facts.get("definition", ""))}'

    if answer_type == "kyusho_point":
        return f'{facts.get("point_name", "That kyusho point")} is {_sentence(facts.get("description", ""))}'

    if answer_type == "kyusho_list":
        names = _limit_plain_items(facts.get("point_names", []), style)
        return f'Kyusho points include {_join_human(names)}.'

    if answer_type == "leadership":
        return f'{facts.get("soke_name", "This person")} is the current soke of {facts.get("school_name", "that school")}.'

    return _compose_fallback_paragraph(result)


def _compose_rank_bullets(result: DeterministicResult, *, style: str) -> str:
    facts = result.facts
    rank = facts.get("rank", "This rank")
    lines = [f"{rank} {facts.get('category_label', 'requirements')}:"] if result.answer_type != "rank_requirements" else [f"{rank} requirements:"]

    if result.answer_type == "rank_requirements":
        for section in facts.get("sections", []):
            label = section.get("label", "")
            content = section.get("content", "")
            if label and content:
                lines.append(f"- {label}: {content}")
        return "\n".join(lines)

    items = list(facts.get("items", []))
    if facts.get("kicks"):
        lines.append(f'- Kicks: {", ".join(facts["kicks"])}')
    if facts.get("carryover_kicks"):
        lines.append(f'- Carryover kicks: {", ".join(facts["carryover_kicks"])}')
    if facts.get("strikes"):
        lines.append(f'- Strikes: {", ".join(facts["strikes"])}')
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _compose_fallback_bullets(result: DeterministicResult) -> str:
    lines = [result.answer_type.replace("_", " ").title() + ":"]
    for key, value in result.facts.items():
        lines.append(f"- {_labelize(key)}: {_comparison_value(value)}")
    return "\n".join(lines)


def _compose_fallback_paragraph(result: DeterministicResult) -> str:
    fragments = [f"{_labelize(key)} = {_comparison_value(value)}" for key, value in result.facts.items()]
    return f'{result.answer_type.replace("_", " ").title()}: ' + "; ".join(fragments) + "."


def _normalize_style(style: str) -> str:
    value = (style or "standard").strip().lower()
    if value not in {"brief", "standard", "full"}:
        return "standard"
    return value


def _normalize_output_format(output_format: str) -> str:
    value = (output_format or "paragraph").strip().lower()
    if value in {"bullet", "bullets"}:
        return "bullets"
    return "paragraph"


def _sentence(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?":
        return cleaned
    return cleaned + "."


def _join_human(items: list[Any]) -> str:
    rendered = [str(item).strip() for item in items if str(item).strip()]
    if not rendered:
        return ""
    if len(rendered) == 1:
        return rendered[0]
    if len(rendered) == 2:
        return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def _comparison_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _bool_label(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value).strip()


def _bool_label(value: bool) -> str:
    return "Yes" if value else "No"


def _labelize(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _limit_items(items: list[dict[str, Any]], style: str) -> list[dict[str, Any]]:
    if style == "brief":
        return items[:3]
    if style == "standard":
        return items[:5]
    return items


def _limit_plain_items(items: list[str], style: str) -> list[str]:
    if style == "brief":
        return items[:5]
    if style == "standard":
        return items[:10]
    return items


def _apply_chatty_tone(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if cleaned.endswith("?"):
        return cleaned
    return cleaned + " If you want, I can keep it concise or unpack one part further."
