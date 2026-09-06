"""The MATH oracle, and specifically the ways it can silently be wrong.

GSM8K answers are integers and its oracle can lean on numeric comparison.
MATH answers are LaTeX, so the failure modes are different in kind: a
regex that stops at the first closing brace truncates nested expressions,
and two spellings of one answer must compare equal or accuracy is
understated everywhere.
"""

from __future__ import annotations

import pytest

from pts_harness.oracle import (
    MathOracle,
    extract_boxed,
    math_answers_from_dataset,
    math_answers_match,
    normalize_math_answer,
)


class TestExtractBoxed:
    def test_nested_braces_are_not_truncated(self) -> None:
        """The bug a naive regex has: \\boxed{\\frac{1}{2}} -> '\\frac{1'."""
        assert extract_boxed(r"thus \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
        assert extract_boxed(r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"

    def test_takes_the_last_box(self) -> None:
        """A model that explores candidates boxes more than once; the final
        one is its claim."""
        assert extract_boxed(r"maybe \boxed{3}, no: \boxed{4}") == "4"

    def test_missing_box_is_none_not_a_guess(self) -> None:
        assert extract_boxed("the answer is clearly 42") is None
        assert extract_boxed("") is None

    def test_unbalanced_box_is_none(self) -> None:
        """Better to score a failure than to invent an answer from a truncated
        generation."""
        assert extract_boxed(r"\boxed{\frac{1}{2}") is None

    def test_bare_token_after_boxed(self) -> None:
        assert extract_boxed(r"\boxed 42 and more") == "42"


class TestNormalise:
    @pytest.mark.parametrize("a,b", [
        (r"\left(0,1\right)", "(0,1)"),
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"\tfrac{1}{2}", r"\frac{1}{2}"),
        (r"2\,\pi", r"2\pi"),
        ("1,000", "1000"),
        (r"\text{5}", "5"),
        ("5.", "5"),
        ("45^\\circ", "45"),
    ])
    def test_equivalent_spellings_normalise_together(self, a: str, b: str) -> None:
        assert normalize_math_answer(a) == normalize_math_answer(b)

    def test_distinct_answers_stay_distinct(self) -> None:
        assert normalize_math_answer("(0,1)") != normalize_math_answer("(0,2)")
        assert normalize_math_answer(r"\frac{1}{2}") != normalize_math_answer(r"\frac{1}{3}")


class TestMatching:
    def test_fraction_against_decimal(self) -> None:
        assert math_answers_match(r"\frac{1}{2}", "0.5")
        assert math_answers_match("0.5", r"\frac{1}{2}")
        assert not math_answers_match(r"\frac{1}{3}", "0.5")

    def test_no_answer_never_matches(self) -> None:
        assert not math_answers_match(None, "3")
        assert not math_answers_match("", "3")

    def test_zero_denominator_does_not_raise(self) -> None:
        assert not math_answers_match(r"\frac{1}{0}", "5")


class TestOracle:
    def test_scores_against_the_table(self) -> None:
        o = MathOracle({"q": r"\frac{1}{2}"})
        assert o("q", r"so \boxed{\dfrac{1}{2}}")
        assert o("q", r"\boxed{0.5}")
        assert not o("q", r"\boxed{\frac{1}{3}}")

    def test_unknown_query_is_a_failure(self) -> None:
        assert not MathOracle({})("q", r"\boxed{1}")

    def test_no_fallback_to_last_number(self) -> None:
        """The GSM8K oracle falls back to the last number; on MATH that turns
        a model which never committed into one that guesses right often,
        because solutions are full of intermediate quantities."""
        o = MathOracle({"q": "5"})
        assert not o("q", "we compute 2 + 3 and obtain 5")

    def test_dataset_table_handles_both_shapes(self) -> None:
        rows = [
            {"problem": "p1", "solution": r"... hence \boxed{7}."},
            {"problem": "p2", "answer": r"\frac{3}{4}"},
            {"problem": "p3", "solution": "no box anywhere"},
        ]
        table = math_answers_from_dataset(rows)
        assert table == {"p1": "7", "p2": r"\frac{3}{4}"}
