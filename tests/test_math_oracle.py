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


# ---------------------------------------------------------------------------
# Divergence from upstream, documented rather than assumed
# ---------------------------------------------------------------------------
#
# The house rule is to reimplement upstream behaviour and check it against
# the clone. For GSM8K the goal is to match. For MATH it is deliberately
# *not*, and these tests pin why, so nobody later "fixes" our oracle back
# towards upstream's.


def _upstream_math_oracle(answers):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "pts"
    if not (root / "pts" / "oracle.py").exists():
        pytest.skip("upstream pts clone not present")
    sys.path.insert(0, str(root))
    try:
        from pts.oracle import MathOracle as Upstream
    except Exception as exc:                                   # pragma: no cover
        pytest.skip(f"upstream not importable: {exc}")
    return Upstream(answers=answers, dataset_format="math")


class TestDivergenceFromUpstream:
    """Upstream's MathOracle is unusable for MATH, in three separate ways."""

    def test_upstream_cannot_separate_two_fractions(self) -> None:
        """Upstream is wrong here whichever way it is configured.

        The default regex ``\\boxed{([^}]+)}`` stops at the first closing
        brace, so ``\\boxed{\\frac{1}{2}}`` extracts ``\\frac{1``. Without
        math_verify the comparison then asks whether the extracted string is
        contained in the expected one, and ``\\frac{1`` is contained in
        ``\\frac{1}{2}`` -- so every fraction with the right numerator passes
        regardless of denominator. With math_verify installed the truncated
        expression fails to parse and *correct* answers are rejected instead.

        Rather than pin one failure mode, assert the invariant that holds in
        both: upstream cannot tell the right fraction from the wrong one.
        """
        gold = r"\frac{1}{2}"
        up = _upstream_math_oracle({"Q": gold})
        right = up.check_success("Q", r"\boxed{\frac{1}{2}}")
        wrong = up.check_success("Q", r"\boxed{\frac{1}{3}}")
        assert right == wrong, (
            "upstream separated them; the extraction bug may be fixed "
            f"(right={right}, wrong={wrong})"
        )

        ours = MathOracle({"Q": gold})
        assert ours.check_success("Q", r"\boxed{\frac{1}{2}}") is True
        assert ours.check_success("Q", r"\boxed{\frac{1}{3}}") is False

    def test_ours_rejects_them(self) -> None:
        """The same cases, without needing the clone."""
        for resp, gold in [(r"\boxed{\frac{1}{3}}", r"\frac{1}{2}"),
                           (r"\boxed{\frac{1}{99}}", r"\frac{1}{2}"),
                           (r"\boxed{\frac{2}{3}}", r"\frac{2}{5}")]:
            assert not MathOracle({"Q": gold}).check_success("Q", resp)
        assert MathOracle({"Q": r"\frac{1}{2}"}).check_success("Q", r"\boxed{\frac{1}{2}}")

    def test_upstream_takes_the_first_box_we_take_the_last(self) -> None:
        up = _upstream_math_oracle({"Q": "4"})
        assert up.check_success("Q", r"maybe \boxed{3}, actually \boxed{4}") is False
        assert MathOracle({"Q": "4"}).check_success("Q", r"maybe \boxed{3}, actually \boxed{4}")

    def test_upstream_falls_back_to_the_last_line(self) -> None:
        """On MATH this rewards a model that never committed to an answer."""
        up = _upstream_math_oracle({"Q": "5"})
        assert up.check_success("Q", "we compute 2 + 3 and obtain 5") is True
        assert MathOracle({"Q": "5"}).check_success("Q", "we compute 2 + 3 and obtain 5") is False
