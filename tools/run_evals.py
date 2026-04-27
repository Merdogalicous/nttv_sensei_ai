from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from nttv_chatbot.eval_harness import (  # noqa: E402
    EvalObservation,
    build_eval_outputs,
    capture_eval_env,
    evaluate_cases,
    load_eval_cases,
    observation_from_pipeline,
    summarize_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NTTV chatbot evals against the local pipeline.")
    parser.add_argument(
        "--questions",
        default=str(ROOT / "evals" / "questions.jsonl"),
        help="Path to the eval question set (.jsonl, .json, .yaml, .yml).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "evals" / "results"),
        help="Directory where markdown/csv reports should be written.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Run label used for output filenames and report headings.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Optional category filter. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of cases to run after filtering.",
    )
    parser.add_argument(
        "--format",
        default="Paragraph",
        choices=["Bullets", "Paragraph"],
        help="Deterministic output format during eval runs.",
    )
    parser.add_argument(
        "--tone",
        default="Crisp",
        choices=["Crisp", "Chatty"],
        help="Deterministic tone during eval runs.",
    )
    parser.add_argument(
        "--technique-detail",
        default="Standard",
        choices=["Brief", "Standard", "Full"],
        help="Technique detail mode during eval runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = args.label.strip() or datetime.now().strftime("eval-%Y%m%d-%H%M%S")

    os.environ.setdefault("STREAMLIT_BROWSER_GATHERUSAGESTATS", "false")
    os.environ.setdefault("STREAMLIT_SUPPRESS_CONFIG_WARNINGS", "true")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

    with redirect_stderr(io.StringIO()):
        import app  # noqa: WPS433,E402

    app.output_style = args.format
    app.tone_style = args.tone
    app.TECH_DETAIL_MODE = args.technique_detail

    cases = load_eval_cases(args.questions)
    if args.category:
        wanted = {value.strip().lower() for value in args.category if value.strip()}
        cases = [case for case in cases if case.category in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]

    if not cases:
        print("No eval cases matched the requested filters.", file=sys.stderr)
        return 1

    def answer_fn(question: str) -> EvalObservation:
        with redirect_stderr(io.StringIO()):
            answer, hits, raw_json, retrieval_debug = app.answer_with_rag(question)
        return observation_from_pipeline(answer, hits, raw_json, retrieval_debug)

    results = evaluate_cases(cases, answer_fn)
    config_snapshot = capture_eval_env()
    config_snapshot["OUTPUT_FORMAT"] = args.format
    config_snapshot["TONE"] = args.tone
    config_snapshot["TECHNIQUE_DETAIL"] = args.technique_detail

    markdown_path, csv_path = build_eval_outputs(
        results,
        run_label=label,
        questions_path=args.questions,
        output_dir=args.out_dir,
        config_snapshot=config_snapshot,
    )
    summary = summarize_results(results)

    print(f"Eval run: {label}")
    print(f"Cases: {summary['total_cases']}")
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"Pass rate: {summary['pass_rate']:.0%}")
    print(f"Markdown report: {markdown_path}")
    print(f"CSV report: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
