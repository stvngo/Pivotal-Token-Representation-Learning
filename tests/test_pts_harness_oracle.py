"""Our GSM8K oracle must agree with upstream's, case for case.

A wrong oracle does not fail loudly -- it mislabels every event. So this is
checked against `pts.oracle.MathOracle` directly whenever the clone is
present, rather than trusting a reimplementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pts_harness.oracle import (
    GSM8KOracle,
    answers_match,
    extract_answer,
    extract_gsm8k_answer,
    normalize_answer,
)

CASES = [
    "The answer is 42.",
    "So we get #### 18",
    "#### 1,234",
    "Thus \\boxed{72} follows.",
    "answer: 3.5",
    "= 17",
    "no numbers at all here",
    "",
    "step one\nstep two\n99",
    "#### -5",
    "The total is 1,234.56 dollars",
]


def test_gsm8k_marker_wins_over_boxed():
    assert extract_answer("\\boxed{7} but #### 9") == "9"


def test_commas_are_stripped_from_gsm8k_answers():
    assert extract_gsm8k_answer("#### 1,234") == "1234"


def test_falls_back_to_last_nonempty_line():
    assert extract_answer("alpha\n\nbeta") == "beta"


def test_numeric_comparison_uses_a_tolerance():
    assert answers_match("18", "18")
    assert answers_match("18.0000001", "18", tolerance=1e-3)
    assert not answers_match("19", "18")


def test_missing_answer_is_a_failure_not_an_error():
    assert not answers_match(None, "18")
    assert not answers_match("", "18")


def test_oracle_only_credits_known_queries():
    oracle = GSM8KOracle({"q": "42"})
    assert oracle("q", "#### 42")
    assert not oracle("q", "#### 41")
    assert not oracle("unknown question", "#### 42")


def test_normalization_strips_decorators_and_spacing():
    assert normalize_answer("4 2") == "42"
    assert normalize_answer("1,5") == "1.5"
    # Upstream's decorator regex alternates (bf|text|rm|cal), so on
    # "\\textbf" the "text" branch matches first and "bf" survives. We
    # reproduce that quirk deliberately rather than "fixing" it, because
    # diverging would change which rollouts count as successes.
    assert normalize_answer("\\textbf{5}") == "bf{5}"


@pytest.fixture(scope="module")
def upstream():
    """`pts.oracle.MathOracle`, or skip if the clone is absent."""
    root = Path(__file__).resolve().parent.parent / "pts"
    if not (root / "pts" / "oracle.py").exists():
        pytest.skip("upstream pts clone not present")
    sys.path.insert(0, str(root))
    try:
        from pts.oracle import MathOracle
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot import upstream oracle: {exc}")
    return MathOracle


@pytest.mark.parametrize("response", CASES)
def test_extraction_matches_upstream(upstream, response):
    theirs = upstream(answers={}, dataset_format="gsm8k")
    assert extract_answer(response, dataset_format="gsm8k") == theirs.extract_answer(response)


@pytest.mark.parametrize("response", CASES)
@pytest.mark.parametrize("expected", ["42", "18", "1234"])
def test_success_verdict_matches_upstream(upstream, response, expected):
    query = "the question"
    theirs = upstream(answers={query: expected}, dataset_format="gsm8k")
    ours = GSM8KOracle({query: expected})
    assert ours.check_success(query, response) == theirs.check_success(query, response)
