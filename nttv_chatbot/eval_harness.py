from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

try:
    import yaml
except Exception:  # pragma: no cover - YAML is optional at runtime
    yaml = None


DEFAULT_PASS_THRESHOLD = 0.75
DEFAULT_CATEGORY_LENGTH_BOUNDS: dict[str, tuple[int, int]] = {
    "deterministic": (30, 1400),
    "retrieval": (40, 1800),
    "ambiguous": (30, 1800),
    "out_of_scope": (15, 500),
}
GENERIC_LENGTH_BOUNDS = (20, 1800)

REFUSAL_PATTERNS = (
    "i don't have enough context",
    "i do not have enough context",
    "not enough context",
    "provided material is incomplete",
    "available material is incomplete",
    "not in the provided material",
    "not in the provided context",
    "outside the provided context",
    "i can't answer",
    "i cannot answer",
    "the material does not say",
    "the provided material does not say",
)


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    category: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_det_path: Optional[str] = None
    expected_sources: list[str] = field(default_factory=list)
    should_refuse: Optional[bool] = None
    min_answer_chars: Optional[int] = None
    max_answer_chars: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        if not isinstance(data, dict):
            raise TypeError("Each eval case must be an object.")

        case_id = str(data.get("id") or "").strip()
        question = str(data.get("question") or "").strip()
        category = str(data.get("category") or "").strip().lower()
        if not case_id:
            raise ValueError("Eval case is missing required field: id")
        if not question:
            raise ValueError(f"Eval case '{case_id}' is missing required field: question")
        if not category:
            raise ValueError(f"Eval case '{case_id}' is missing required field: category")

        return cls(
            id=case_id,
            question=question,
            category=category,
            expected_keywords=_normalize_string_list(data.get("expected_keywords")),
            expected_det_path=_normalize_optional_string(data.get("expected_det_path")),
            expected_sources=_normalize_string_list(data.get("expected_sources")),
            should_refuse=_normalize_optional_bool(data.get("should_refuse")),
            min_answer_chars=_normalize_optional_int(data.get("min_answer_chars")),
            max_answer_chars=_normalize_optional_int(data.get("max_answer_chars")),
        )


@dataclass(frozen=True)
class SourceRecord:
    source: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    heading_path: list[str] = field(default_factory=list)
    chunk_id: Optional[str] = None
    priority_bucket: Optional[str] = None
    score: Optional[float] = None

    @property
    def source_name(self) -> str:
        return os.path.basename(self.source or "").strip()

    def to_display_string(self) -> str:
        label = self.source_name or "(unknown)"
        if self.page_start and self.page_end and self.page_start != self.page_end:
            label += f" pp.{self.page_start}-{self.page_end}"
        elif self.page_start:
            label += f" p.{self.page_start}"
        if self.heading_path:
            label += f" | {' > '.join(self.heading_path)}"
        return label


@dataclass(frozen=True)
class EvalObservation:
    answer: str
    raw_json: str
    det_path: Optional[str]
    deterministic_routing_fired: bool
    deterministic_short_circuit: bool
    route: str
    model_used: str
    model_requested: str
    retrieved_sources: list[SourceRecord] = field(default_factory=list)
    supporting_sources: list[SourceRecord] = field(default_factory=list)

    @property
    def answer_char_count(self) -> int:
        return len((self.answer or "").strip())

    @property
    def observed_source_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for record in [*self.supporting_sources, *self.retrieved_sources]:
            name = record.source_name
            if not name:
                continue
            lowered = name.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            names.append(name)
        return names


@dataclass(frozen=True)
class EvalScore:
    keyword_hits: int
    keyword_total: int
    keyword_hit_rate: float
    source_hits: int
    source_total: int
    source_hit_rate: float
    det_path_match: Optional[bool]
    refusal_detected: bool
    refusal_match: Optional[bool]
    answer_length_ok: bool
    min_answer_chars: int
    max_answer_chars: int
    overall_score: float
    passed: bool
    metric_breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    observation: EvalObservation
    score: EvalScore


def load_eval_cases(path: str | os.PathLike[str]) -> list[EvalCase]:
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    suffix = eval_path.suffix.lower()
    if suffix == ".jsonl":
        raw_cases = []
        for line_number, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw_cases.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {eval_path}: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML eval files.")
        loaded = yaml.safe_load(eval_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_cases = loaded.get("cases") or []
        elif isinstance(loaded, list):
            raw_cases = loaded
        else:
            raise ValueError(f"Unsupported YAML structure in {eval_path}.")
    elif suffix == ".json":
        loaded = json.loads(eval_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_cases = loaded.get("cases") or []
        elif isinstance(loaded, list):
            raw_cases = loaded
        else:
            raise ValueError(f"Unsupported JSON structure in {eval_path}.")
    else:
        raise ValueError(f"Unsupported eval file format: {eval_path.suffix}")

    cases = [EvalCase.from_dict(item) for item in raw_cases]
    _validate_unique_case_ids(cases)
    return cases


def observation_from_pipeline(
    answer: str,
    hits: list[dict[str, Any]],
    raw_json: str,
    retrieval_debug: dict[str, Any],
) -> EvalObservation:
    llm_debug = retrieval_debug.get("llm_routing") or {}
    det_path = extract_det_path(raw_json)
    supporting_sources = extract_source_refs(raw_json)
    retrieved_sources = [_source_record_from_hit(hit) for hit in hits[:8]]
    route = str(llm_debug.get("route") or "")
    deterministic_fired = bool(det_path) or route.startswith("deterministic")

    return EvalObservation(
        answer=(answer or "").strip(),
        raw_json=raw_json or "",
        det_path=det_path,
        deterministic_routing_fired=deterministic_fired,
        deterministic_short_circuit=bool(retrieval_debug.get("deterministic_short_circuit")),
        route=route,
        model_used=str(llm_debug.get("model_used") or ""),
        model_requested=str(llm_debug.get("model_requested") or ""),
        retrieved_sources=[record for record in retrieved_sources if record.source],
        supporting_sources=supporting_sources,
    )


def score_eval_case(
    case: EvalCase,
    observation: EvalObservation,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> EvalScore:
    answer_text = (observation.answer or "").casefold()
    notes: list[str] = []
    metric_breakdown: dict[str, float] = {}

    keyword_hits = sum(1 for keyword in case.expected_keywords if keyword.casefold() in answer_text)
    keyword_total = len(case.expected_keywords)
    keyword_hit_rate = (keyword_hits / keyword_total) if keyword_total else 1.0
    if keyword_total:
        metric_breakdown["keyword_hit_rate"] = keyword_hit_rate
        missing = [keyword for keyword in case.expected_keywords if keyword.casefold() not in answer_text]
        if missing:
            notes.append(f"Missing keywords: {', '.join(missing)}")

    observed_sources = {name.casefold() for name in observation.observed_source_names}
    expected_sources = [os.path.basename(source).strip() for source in case.expected_sources]
    source_hits = sum(1 for source in expected_sources if source.casefold() in observed_sources)
    source_total = len(expected_sources)
    source_hit_rate = (source_hits / source_total) if source_total else 1.0
    if source_total:
        metric_breakdown["source_hit_rate"] = source_hit_rate
        missing_sources = [source for source in expected_sources if source.casefold() not in observed_sources]
        if missing_sources:
            notes.append(f"Missing expected sources: {', '.join(missing_sources)}")

    det_path_match: Optional[bool] = None
    if case.expected_det_path is not None:
        det_path_match = observation.det_path == case.expected_det_path
        metric_breakdown["det_path_match"] = 1.0 if det_path_match else 0.0
        if not det_path_match:
            notes.append(
                f"Expected det_path '{case.expected_det_path}' but observed '{observation.det_path or 'none'}'."
            )

    refusal_detected = detect_refusal(observation.answer)
    refusal_match: Optional[bool] = None
    if case.should_refuse is not None:
        refusal_match = refusal_detected == case.should_refuse
        metric_breakdown["refusal_match"] = 1.0 if refusal_match else 0.0
        if not refusal_match:
            expectation = "refusal" if case.should_refuse else "non-refusal"
            notes.append(f"Expected {expectation} behavior.")

    min_answer_chars, max_answer_chars = resolve_answer_length_bounds(case)
    answer_length_ok = min_answer_chars <= observation.answer_char_count <= max_answer_chars
    metric_breakdown["answer_length_ok"] = 1.0 if answer_length_ok else 0.0
    if not answer_length_ok:
        notes.append(
            f"Answer length {observation.answer_char_count} outside bounds {min_answer_chars}-{max_answer_chars}."
        )

    overall_score = mean(metric_breakdown.values()) if metric_breakdown else 0.0
    passed = overall_score >= pass_threshold

    return EvalScore(
        keyword_hits=keyword_hits,
        keyword_total=keyword_total,
        keyword_hit_rate=keyword_hit_rate,
        source_hits=source_hits,
        source_total=source_total,
        source_hit_rate=source_hit_rate,
        det_path_match=det_path_match,
        refusal_detected=refusal_detected,
        refusal_match=refusal_match,
        answer_length_ok=answer_length_ok,
        min_answer_chars=min_answer_chars,
        max_answer_chars=max_answer_chars,
        overall_score=overall_score,
        passed=passed,
        metric_breakdown=metric_breakdown,
        notes=notes,
    )


def evaluate_cases(
    cases: list[EvalCase],
    answer_fn: Callable[[str], EvalObservation],
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        observation = answer_fn(case.question)
        score = score_eval_case(case, observation, pass_threshold=pass_threshold)
        results.append(EvalResult(case=case, observation=observation, score=score))
    return results


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    total_cases = len(results)
    if total_cases == 0:
        return {
            "total_cases": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "average_score": 0.0,
            "deterministic_count": 0,
            "model_usage": {},
            "category_summary": {},
        }

    category_groups: dict[str, list[EvalResult]] = defaultdict(list)
    model_counts: Counter[str] = Counter()
    deterministic_count = 0

    for result in results:
        category_groups[result.case.category].append(result)
        if result.observation.model_used:
            model_counts[result.observation.model_used] += 1
        if result.observation.deterministic_routing_fired:
            deterministic_count += 1

    category_summary: dict[str, Any] = {}
    for category, category_results in sorted(category_groups.items()):
        category_summary[category] = {
            "cases": len(category_results),
            "pass_count": sum(1 for result in category_results if result.score.passed),
            "pass_rate": _safe_ratio(
                sum(1 for result in category_results if result.score.passed),
                len(category_results),
            ),
            "average_score": mean(result.score.overall_score for result in category_results),
            "deterministic_count": sum(
                1 for result in category_results if result.observation.deterministic_routing_fired
            ),
        }

    pass_count = sum(1 for result in results if result.score.passed)
    return {
        "total_cases": total_cases,
        "pass_count": pass_count,
        "pass_rate": _safe_ratio(pass_count, total_cases),
        "average_score": mean(result.score.overall_score for result in results),
        "deterministic_count": deterministic_count,
        "model_usage": dict(model_counts),
        "category_summary": category_summary,
    }


def render_markdown_report(
    results: list[EvalResult],
    *,
    run_label: str,
    questions_path: str,
    config_snapshot: Optional[dict[str, str]] = None,
) -> str:
    summary = summarize_results(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines = [
        f"# NTTV Eval Report: {run_label}",
        "",
        f"- Generated: `{timestamp}`",
        f"- Questions file: `{questions_path}`",
        f"- Total cases: `{summary['total_cases']}`",
        f"- Average score: `{summary['average_score']:.2f}`",
        f"- Pass rate: `{summary['pass_rate']:.0%}`",
        "",
    ]

    if config_snapshot:
        lines.extend(
            [
                "## Configuration",
                "",
                "| Setting | Value |",
                "|---|---|",
            ]
        )
        for key, value in sorted(config_snapshot.items()):
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")

    lines.extend(
        [
            "## Category Summary",
            "",
            "| Category | Cases | Avg Score | Pass Rate | Deterministic |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, category_summary in summary["category_summary"].items():
        lines.append(
            "| {category} | {cases} | {avg:.2f} | {pass_rate:.0%} | {deterministic} |".format(
                category=category,
                cases=category_summary["cases"],
                avg=category_summary["average_score"],
                pass_rate=category_summary["pass_rate"],
                deterministic=category_summary["deterministic_count"],
            )
        )
    lines.append("")

    if summary["model_usage"]:
        lines.extend(
            [
                "## Model Usage",
                "",
                "| Model | Count |",
                "|---|---:|",
            ]
        )
        for model_name, count in sorted(summary["model_usage"].items()):
            lines.append(f"| `{model_name}` | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Results",
            "",
            "| ID | Category | Score | Route | Model | Det Path | Sources |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for result in results:
        lines.append(
            "| {id} | {category} | {score:.2f} | `{route}` | `{model}` | `{det_path}` | {sources} |".format(
                id=result.case.id,
                category=result.case.category,
                score=result.score.overall_score,
                route=result.observation.route or "n/a",
                model=result.observation.model_used or "n/a",
                det_path=result.observation.det_path or "n/a",
                sources=", ".join(result.observation.observed_source_names[:3]) or "n/a",
            )
        )
    lines.append("")

    needs_review = [result for result in results if not result.score.passed]
    if needs_review:
        lines.extend(["## Needs Review", ""])
        for result in needs_review:
            lines.extend(
                [
                    f"### {result.case.id} ({result.case.category})",
                    "",
                    f"- Score: `{result.score.overall_score:.2f}`",
                    f"- Question: {result.case.question}",
                    f"- Route/model: `{result.observation.route or 'n/a'}` / `{result.observation.model_used or 'n/a'}`",
                    f"- Det path: `{result.observation.det_path or 'n/a'}`",
                    f"- Sources: {', '.join(result.observation.observed_source_names) or 'n/a'}",
                    f"- Notes: {'; '.join(result.score.notes) or 'none'}",
                    "",
                    "**Answer**",
                    "",
                    result.observation.answer or "(empty)",
                    "",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def write_csv_report(results: list[EvalResult], path: str | os.PathLike[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "category",
                "question",
                "overall_score",
                "passed",
                "keyword_hits",
                "keyword_total",
                "keyword_hit_rate",
                "source_hits",
                "source_total",
                "source_hit_rate",
                "det_path",
                "det_path_match",
                "deterministic_routing_fired",
                "deterministic_short_circuit",
                "route",
                "model_used",
                "model_requested",
                "refusal_detected",
                "refusal_match",
                "answer_char_count",
                "answer_length_ok",
                "observed_sources",
                "notes",
                "answer",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "id": result.case.id,
                    "category": result.case.category,
                    "question": result.case.question,
                    "overall_score": f"{result.score.overall_score:.4f}",
                    "passed": result.score.passed,
                    "keyword_hits": result.score.keyword_hits,
                    "keyword_total": result.score.keyword_total,
                    "keyword_hit_rate": f"{result.score.keyword_hit_rate:.4f}",
                    "source_hits": result.score.source_hits,
                    "source_total": result.score.source_total,
                    "source_hit_rate": f"{result.score.source_hit_rate:.4f}",
                    "det_path": result.observation.det_path or "",
                    "det_path_match": result.score.det_path_match,
                    "deterministic_routing_fired": result.observation.deterministic_routing_fired,
                    "deterministic_short_circuit": result.observation.deterministic_short_circuit,
                    "route": result.observation.route,
                    "model_used": result.observation.model_used,
                    "model_requested": result.observation.model_requested,
                    "refusal_detected": result.score.refusal_detected,
                    "refusal_match": result.score.refusal_match,
                    "answer_char_count": result.observation.answer_char_count,
                    "answer_length_ok": result.score.answer_length_ok,
                    "observed_sources": "; ".join(result.observation.observed_source_names),
                    "notes": "; ".join(result.score.notes),
                    "answer": result.observation.answer,
                }
            )


def write_markdown_report(content: str, path: str | os.PathLike[str]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def build_eval_outputs(
    results: list[EvalResult],
    *,
    run_label: str,
    questions_path: str,
    output_dir: str | os.PathLike[str],
    config_snapshot: Optional[dict[str, str]] = None,
) -> tuple[Path, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_label = _slugify(run_label)
    markdown_path = output_root / f"{safe_label}.md"
    csv_path = output_root / f"{safe_label}.csv"
    markdown = render_markdown_report(
        results,
        run_label=run_label,
        questions_path=questions_path,
        config_snapshot=config_snapshot,
    )
    write_markdown_report(markdown, markdown_path)
    write_csv_report(results, csv_path)
    return markdown_path, csv_path


def capture_eval_env(keys: Optional[list[str]] = None) -> dict[str, str]:
    names = keys or [
        "OPENAI_BASE_URL",
        "OPENROUTER_API_BASE",
        "MODEL",
        "SYNTHESIS_MODEL",
        "USE_SYNTHESIS_MODEL",
        "SYNTHESIS_MIN_CONTEXT_CHUNKS",
        "SYNTHESIS_FOR_EXPLANATION_MODE",
        "SYNTHESIS_FOR_DETERMINISTIC_COMPOSER",
        "TOP_K",
        "USE_HYBRID_RETRIEVAL",
        "DENSE_TOP_K",
        "LEXICAL_TOP_K",
        "FUSED_TOP_K",
        "RERANKER_BACKEND",
    ]
    snapshot: dict[str, str] = {}
    for key in names:
        value = os.getenv(key)
        if value is None or value == "":
            continue
        snapshot[key] = value
    return snapshot


def detect_refusal(answer: str) -> bool:
    lowered = (answer or "").casefold()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def extract_det_path(raw_json: str) -> Optional[str]:
    payload = _try_parse_json(raw_json)
    if isinstance(payload, dict):
        det_path = payload.get("det_path")
        if isinstance(det_path, str) and det_path.strip():
            return det_path.strip()

    match = re.search(r'"det_path"\s*:\s*"([^"]+)"', raw_json or "")
    if match:
        return match.group(1)
    return None


def extract_source_refs(raw_json: str) -> list[SourceRecord]:
    payload = _try_parse_json(raw_json)
    if not isinstance(payload, dict):
        return []

    refs = payload.get("source_refs") or []
    records: list[SourceRecord] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        source = str(ref.get("source") or "").strip()
        if not source:
            continue
        records.append(
            SourceRecord(
                source=source,
                page_start=_normalize_optional_int(ref.get("page_start")),
                page_end=_normalize_optional_int(ref.get("page_end")),
                heading_path=_normalize_string_list(ref.get("heading_path")),
                chunk_id=_normalize_optional_string(ref.get("chunk_id")),
            )
        )
    return records


def resolve_answer_length_bounds(case: EvalCase) -> tuple[int, int]:
    defaults = DEFAULT_CATEGORY_LENGTH_BOUNDS.get(case.category, GENERIC_LENGTH_BOUNDS)
    min_chars = case.min_answer_chars if case.min_answer_chars is not None else defaults[0]
    max_chars = case.max_answer_chars if case.max_answer_chars is not None else defaults[1]
    return min_chars, max_chars


def _source_record_from_hit(hit: dict[str, Any]) -> SourceRecord:
    meta = hit.get("meta") or {}
    source = (
        meta.get("source_file")
        or meta.get("source")
        or hit.get("source_file")
        or hit.get("source")
        or ""
    )
    return SourceRecord(
        source=str(source or "").strip(),
        page_start=_normalize_optional_int(meta.get("page_start") or meta.get("page") or hit.get("page_start") or hit.get("page")),
        page_end=_normalize_optional_int(meta.get("page_end") or hit.get("page_end")),
        heading_path=_normalize_string_list(meta.get("heading_path") or hit.get("heading_path")),
        chunk_id=_normalize_optional_string(meta.get("chunk_id") or hit.get("chunk_id")),
        priority_bucket=_normalize_optional_string(meta.get("priority_bucket")),
        score=_normalize_optional_float(hit.get("final_score") or hit.get("rerank_score") or hit.get("score")),
    )


def _try_parse_json(raw_json: str) -> Any:
    if not raw_json:
        return None
    try:
        return json.loads(raw_json)
    except Exception:
        return None


def _validate_unique_case_ids(cases: list[EvalCase]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.id in seen:
            duplicates.append(case.id)
        seen.add(case.id)
    if duplicates:
        raise ValueError(f"Duplicate eval ids found: {', '.join(sorted(set(duplicates)))}")


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        raise TypeError(f"Expected a list of strings, got {type(value).__name__}.")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"Expected a boolean value, got {value!r}.")


def _normalize_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _normalize_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _slugify(value: str) -> str:
    lowered = (value or "eval-run").strip().casefold()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = lowered.strip("-")
    return lowered or "eval-run"
