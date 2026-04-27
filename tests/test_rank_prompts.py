import glob
import os
import re

import pytest

from extractors import try_extract_answer
from tests.helpers import render_result

ROOT = os.path.dirname(os.path.dirname(__file__))
RANK_PATH = os.path.join(ROOT, "data", "nttv rank requirements.txt")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _read_rank_text() -> str:
    with open(RANK_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_prompt_blocks(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()

    blocks = [block.strip() for block in re.split(r"^\s*---\s*$", raw, flags=re.M) if block.strip()]
    cases = []
    for block in blocks:
        obj = {"QUESTION": "", "EXPECT_ALL": [], "EXPECT_ANY": [], "EXPECT_NOT": []}
        for line in block.splitlines():
            if not line.strip() or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "QUESTION":
                obj["QUESTION"] = value
            elif key == "EXPECT_ALL":
                obj["EXPECT_ALL"] = [token.strip() for token in value.split(",") if token.strip()]
            elif key == "EXPECT_ANY":
                obj["EXPECT_ANY"] = [token.strip() for token in re.split(r"\|", value) if token.strip()]
            elif key == "EXPECT_NOT":
                obj["EXPECT_NOT"] = [token.strip() for token in value.split(",") if token.strip()]
        if obj["QUESTION"]:
            cases.append(obj)
    return cases


def _collect_cases():
    files = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
    cases = []
    for path in files:
        for case in _load_prompt_blocks(path):
            cases.append((os.path.basename(path), case))
    return cases


@pytest.mark.parametrize("source_file,case", _collect_cases())
def test_rank_prompt_cases(source_file, case):
    passages = [
        {
            "text": _read_rank_text(),
            "source": "nttv rank requirements.txt",
            "meta": {"priority": 3},
        }
    ]

    answer = try_extract_answer(case["QUESTION"], passages)
    assert answer and answer.answered, f"No answer for: {case['QUESTION']}"

    rendered = render_result(answer, style="full", output_format="paragraph").lower()

    for token in case.get("EXPECT_ALL", []):
        assert token.lower() in rendered, f"Missing token '{token}' in answer: {rendered}"

    any_list = case.get("EXPECT_ANY", [])
    if any_list:
        assert any(token.lower() in rendered for token in any_list), (
            f"None of EXPECT_ANY tokens {any_list} found in answer: {rendered}"
        )

    for token in case.get("EXPECT_NOT", []):
        assert token.lower() not in rendered, f"Forbidden token '{token}' present in answer: {rendered}"
