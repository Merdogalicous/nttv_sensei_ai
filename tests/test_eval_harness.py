from __future__ import annotations

import json

from nttv_chatbot.eval_harness import (
    EvalCase,
    EvalObservation,
    EvalResult,
    SourceRecord,
    build_eval_outputs,
    load_eval_cases,
    score_eval_case,
)


def test_load_eval_cases_from_jsonl(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    rows = [
        {
            "id": "det_1",
            "question": "What kicks do I need to know for 8th kyu?",
            "category": "deterministic",
            "expected_keywords": ["sokuho geri"],
            "expected_det_path": "rank/striking",
            "expected_sources": ["nttv rank requirements.txt"],
        },
        {
            "id": "oos_1",
            "question": "What is the capital of France?",
            "category": "out_of_scope",
            "should_refuse": True,
        },
    ]
    questions_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    cases = load_eval_cases(questions_path)

    assert [case.id for case in cases] == ["det_1", "oos_1"]
    assert cases[0].expected_det_path == "rank/striking"
    assert cases[1].should_refuse is True


def test_score_eval_case_tracks_keywords_sources_det_path_and_length():
    case = EvalCase(
        id="det_technique",
        question="Explain Oni Kudaki.",
        category="deterministic",
        expected_keywords=["oni kudaki", "elbow"],
        expected_det_path="technique/core",
        expected_sources=["Technique Descriptions.md"],
    )
    observation = EvalObservation(
        answer="Oni Kudaki is an elbow control technique from the curriculum.",
        raw_json='{"det_path":"technique/core"}',
        det_path="technique/core",
        deterministic_routing_fired=True,
        deterministic_short_circuit=True,
        route="deterministic_local",
        model_used="deterministic_composer",
        model_requested="",
        retrieved_sources=[],
        supporting_sources=[
            SourceRecord(source="data/Technique Descriptions.md", page_start=4),
        ],
    )

    score = score_eval_case(case, observation)

    assert score.keyword_hits == 2
    assert score.source_hits == 1
    assert score.det_path_match is True
    assert score.answer_length_ok is True
    assert score.overall_score == 1.0
    assert score.passed is True


def test_score_eval_case_checks_refusal_behavior():
    case = EvalCase(
        id="oos_question",
        question="What is the capital of France?",
        category="out_of_scope",
        should_refuse=True,
    )
    observation = EvalObservation(
        answer="I do not have enough context in the provided material to answer that.",
        raw_json="{}",
        det_path=None,
        deterministic_routing_fired=False,
        deterministic_short_circuit=False,
        route="primary",
        model_used="google/gemma-3-27b-it",
        model_requested="google/gemma-3-27b-it",
        retrieved_sources=[],
        supporting_sources=[],
    )

    score = score_eval_case(case, observation)

    assert score.refusal_detected is True
    assert score.refusal_match is True
    assert score.answer_length_ok is True
    assert score.overall_score == 1.0


def test_build_eval_outputs_writes_markdown_and_csv(tmp_path):
    case = EvalCase(
        id="ret_buyu",
        question="What is Buyu?",
        category="retrieval",
        expected_keywords=["martial arts friend"],
        expected_sources=["What is Buyu.txt"],
    )
    observation = EvalObservation(
        answer="Buyu means a martial arts friend.",
        raw_json='{"id":"llm"}',
        det_path=None,
        deterministic_routing_fired=False,
        deterministic_short_circuit=False,
        route="synthesis",
        model_used="google/gemma-3-27b-it",
        model_requested="google/gemma-3-27b-it",
        retrieved_sources=[SourceRecord(source="data/What is Buyu.txt")],
        supporting_sources=[],
    )
    score = score_eval_case(case, observation)
    results = [EvalResult(case=case, observation=observation, score=score)]

    markdown_path, csv_path = build_eval_outputs(
        results,
        run_label="baseline",
        questions_path="evals/questions.jsonl",
        output_dir=tmp_path,
        config_snapshot={"MODEL": "google/gemma-3n-e4b-it"},
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    assert markdown_path.exists()
    assert csv_path.exists()
    assert "NTTV Eval Report: baseline" in markdown
    assert "ret_buyu" in markdown
    assert "google/gemma-3-27b-it" in markdown
    assert "ret_buyu" in csv_text
    assert "martial arts friend" in csv_text
