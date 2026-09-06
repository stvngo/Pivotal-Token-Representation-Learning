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


# ---------------------------------------------------------------------------
# MATH (Hendrycks et al.)
# ---------------------------------------------------------------------------
#
# GSM8K answers are integers, so the GSM8K oracle can lean on numeric
# comparison. MATH answers are LaTeX expressions -- fractions, radicals,
# intervals, matrices -- and the failure modes are different: `\dfrac{1}{2}`
# and `\frac{1}{2}` are the same answer, `\left(0,1\right)` and `(0,1)` are
# the same answer, and a regex that stops at the first `}` truncates
# `\boxed{\frac{1}{2}}` to `\frac{1`.
#
# That last one matters most, and it is why extraction here is a brace
# matcher rather than a pattern. The module-level ``_BOXED`` regex above is
# kept as-is because it reproduces upstream behaviour on GSM8K, where the
# nesting does not arise.


def extract_boxed(text: str) -> str | None:
    """Return the contents of the LAST ``\\boxed{...}``, matching braces.

    The last, not the first: a model that restates the problem or works
    through candidate answers will emit earlier boxes, and the final one is
    the claim. Nested braces are counted, so ``\\boxed{\\frac{1}{2}}``
    returns ``\\frac{1}{2}`` rather than ``\\frac{1``.
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text):
        return None
    if text[i] != "{":
        # \boxed12 -- rare, but upstream tolerates a bare token.
        j = i
        while j < len(text) and not text[j].isspace() and text[j] != "$":
            j += 1
        return text[i:j] or None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None            # unbalanced; treat as no answer rather than guess


_MATH_STRIP = (
    (r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\;", ""),
    (r"\$", ""), ("$", ""), (r"\%", ""), ("%", ""), (r"\ ", ""),
    ("dfrac", "frac"), ("tfrac", "frac"), (r"^\circ", ""), (r"\circ", ""),
)


def normalize_math_answer(answer: str | None) -> str:
    """Canonicalise a LaTeX answer enough to compare two spellings of it.

    Deliberately conservative: it removes formatting that never changes
    meaning (spacing macros, \\left/\\right, dollar signs, dfrac vs frac)
    and does not attempt symbolic equivalence. Two answers that differ by
    algebra rather than by spelling are treated as different, which
    understates accuracy rather than overstating it.
    """
    if answer is None:
        return ""
    out = answer.strip()
    if out.startswith("\\boxed"):
        out = extract_boxed(out) or out
    for a, b in _MATH_STRIP:
        out = out.replace(a, b)
    out = re.sub(r"\\text\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\\mbox\{([^}]*)\}", r"\1", out)
    out = re.sub(r"\s+", "", out)
    out = out.rstrip(".")
    out = re.sub(r"^\{(.*)\}$", r"\1", out)          # a stray outer brace
    out = re.sub(r"(\d),(\d{3})\b", r"\1\2", out)     # 1,000 -> 1000
    # A trailing units word ("cm", "degrees") is formatting, not an answer.
    out = re.sub(r"(?:\\?(?:text|mathrm)\{)?(cm|m|km|inches|in|ft|units?|"
                 r"degrees?|dollars?)\}?$", "", out)
    return out.lower()


def math_answers_match(extracted: str | None, expected: str) -> bool:
    a, b = normalize_math_answer(extracted), normalize_math_answer(expected)
    if not a:
        return False
    if a == b:
        return True
    if _is_numeric(a) and _is_numeric(b):
        return abs(float(a) - float(b)) < 1e-6
    # \frac{a}{b} against a decimal, which models produce constantly.
    m = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", a)
    n = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", b)
    try:
        if m and _is_numeric(b):
            return abs(float(m[1]) / float(m[2]) - float(b)) < 1e-6
        if n and _is_numeric(a):
            return abs(float(n[1]) / float(n[2]) - float(a)) < 1e-6
    except ZeroDivisionError:
        return False
    return False


class MathOracle:
    """``oracle(query, response) -> bool`` for MATH-style boxed answers.

    Unlike the GSM8K oracle this does **not** fall back to "the last number
    anywhere" or "the last line". On MATH those fallbacks are actively
    harmful: solutions are long and full of intermediate quantities, so a
    fallback turns a model that never committed to an answer into one that
    frequently guesses right. A response with no ``\\boxed`` is scored a
    failure, which is what it is.
    """

    def __init__(self, answers: Mapping[str, str]) -> None:
        self.answers = dict(answers)

    def __call__(self, query: str, response: str) -> bool:
        return self.check_success(query, response)

    def check_success(self, query: str, response: str) -> bool:
        expected = self.answers.get(query)
        if expected is None:
            return False
        return math_answers_match(extract_boxed(response), expected)


def math_answers_from_dataset(rows) -> dict[str, str]:
    """Build the query->answer table from a MATH split.

    Handles both shapes in circulation: a ``solution`` field with the answer
    boxed inside it (Hendrycks' original), and a pre-extracted ``answer``
    field (the MATH-500 repackagings).
    """
    out: dict[str, str] = {}
    for row in rows:
        q = row.get("problem") or row.get("question") or row.get("query", "")
        gold = row.get("answer")
        if not gold:
            gold = extract_boxed(row.get("solution", "") or "")
        if q and gold:
            out[q] = str(gold)
    return out
