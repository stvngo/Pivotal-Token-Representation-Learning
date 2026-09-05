"""Success oracle for GSM8K, matching upstream ``pts.oracle.MathOracle``.

Reimplemented rather than imported so the harness has no hard dependency on
a sibling clone, but the behaviour is meant to be identical and
``tests/test_pts_harness_oracle.py`` checks it against upstream directly
when the clone is present.

The oracle is the single most likely thing to be wrong in a PTS run, and a
wrong oracle silently mislabels every event rather than failing. That is why
the harness also stores raw rollouts by default: re-scoring is cheap,
regenerating is not.
"""

from __future__ import annotations

import re
from typing import Mapping

_BOXED = re.compile(r"\\boxed{([^}]+)}", re.IGNORECASE)
_GSM8K = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_ANSWER = re.compile(
    r"(?:answer(?:\s+)?(?:is|:)?(?:\s+)?|=(?:\s+)?)(?P<answer>[^.,]+)", re.IGNORECASE
)


def extract_gsm8k_answer(response: str) -> str | None:
    m = _GSM8K.search(response)
    return m.group(1).replace(",", "") if m else None


def extract_answer(response: str, *, dataset_format: str | None = "gsm8k") -> str | None:
    """Upstream's cascade: #### -> \\boxed -> #### -> 'answer is X' -> last line."""
    if dataset_format == "gsm8k":
        found = extract_gsm8k_answer(response)
        if found:
            return found

    boxed = _BOXED.search(response)
    if boxed:
        return boxed.group(1).strip()

    found = extract_gsm8k_answer(response)
    if found:
        return found

    m = _ANSWER.search(response)
    if m:
        return m.group("answer").strip()

    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    return lines[-1] if lines else None


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    out = answer.lower().replace(" ", "")
    out = re.sub(r"\\(?:math)?(?:bf|text|rm|cal)", "", out)
    out = re.sub(r"(\d),(\d)", r"\1.\2", out)
    return out


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def answers_match(extracted: str | None, expected: str, *, tolerance: float = 1e-6) -> bool:
    a, b = normalize_answer(extracted), normalize_answer(expected)
    if not a:
        return False
    if _is_numeric(a) and _is_numeric(b):
        return abs(float(a) - float(b)) < tolerance
    return a == b or b in a


class GSM8KOracle:
    """``oracle(query, response) -> bool`` over a query->answer table."""

    def __init__(self, answers: Mapping[str, str], *, tolerance: float = 1e-6) -> None:
        self.answers = dict(answers)
        self.tolerance = tolerance

    def __call__(self, query: str, response: str) -> bool:
        return self.check_success(query, response)

    def check_success(self, query: str, response: str) -> bool:
        expected = self.answers.get(query)
        if expected is None:
            return False
        return answers_match(
            extract_answer(response, dataset_format="gsm8k"),
            expected,
            tolerance=self.tolerance,
        )


def gsm8k_answers_from_dataset(rows) -> dict[str, str]:
    """Build the query->answer table from a GSM8K split."""
    out: dict[str, str] = {}
    for row in rows:
        q = row["question"] if "question" in row else row.get("query", "")
        gold = extract_gsm8k_answer(row.get("answer", ""))
        if q and gold is not None:
            out[q] = gold
    return out
