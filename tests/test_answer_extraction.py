"""Short, dependency-light sanity tests for the GSM8K answer parser.

Runs in under a second. Exits non-zero on failure so it can be used as a
pre-flight check before launching the Colab / SageMaker notebook.

    python tests/test_answer_extraction.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probe_pipeline.gsm8k_eval import (  # noqa: E402
    extract_gsm8k_answer,
    extract_gsm8k_ground_truth,
    is_correct,
)


def _check(name: str, got, expected) -> bool:
    ok = got == expected
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}: got={got!r} expected={expected!r}")
    return ok


def main() -> int:
    failures = 0

    print("extract_gsm8k_answer:")
    cases = [
        ("plain_delim", "#### 42", "42"),
        ("thousands_comma", "He has #### 1,234 apples.", "1234"),
        ("negative", "The net change is #### -7", "-7"),
        ("decimal_trailing_zero", "So the answer is #### 3.50", "3.50"),
        ("no_delim_fallback", "...so the total is 18 apples.", "18"),
        ("ground_truth_field", "Jane has 5. Final: #### 5", "5"),
        ("empty_string", "", None),
    ]
    for name, text, expected in cases:
        if not _check(name, extract_gsm8k_answer(text), expected):
            failures += 1

    print("\nextract_gsm8k_ground_truth (shares parser):")
    if not _check(
        "gt_with_reasoning",
        extract_gsm8k_ground_truth("A chain of reasoning...\n#### 72"),
        "72",
    ):
        failures += 1

    print("\nis_correct:")
    correct_cases = [
        ("float_vs_int", is_correct("3.0", "3"), True),
        ("comma_vs_plain", is_correct("1234", "1,234"), True),
        ("none_predicted", is_correct(None, "5"), False),
        ("none_gt", is_correct("5", None), False),
        ("mismatch", is_correct("4", "5"), False),
        ("whitespace", is_correct(" 12 ", "12"), True),
    ]
    for name, got, expected in correct_cases:
        if not _check(name, got, expected):
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures} case(s) did not match expected output.")
        return 1
    print("All answer-extraction tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
