"""Tests for probe-row construction.

No model, no GPU, no HF cache -- a stub tokenizer stands in, including one
that reproduces the BPE merge behaviour that made the original index
arithmetic fragile.
"""

from __future__ import annotations

import pytest

from probe_pipeline.probe_dataset import (
    LABEL_NON_PIVOTAL,
    LABEL_PIVOTAL,
    LABEL_UNUSED,
    BuildStats,
    build_probe_dataset,
    build_rows_from_group,
    locate_pivot,
    split_by_query,
)


class WordTokenizer:
    """Whitespace tokenizer: one id per distinct whitespace-delimited word.

    Deliberately simple, so the index arithmetic under test is the only thing
    that can be wrong.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab) + 1000)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(t) for t in text.split()]


class MergingTokenizer(WordTokenizer):
    """Like WordTokenizer, but a trailing '.' fuses with the previous word.

    This is the real BPE hazard in miniature: encode(a) + encode(b) is not
    encode(a + b), so ``len(encode(context))`` is the wrong index and the
    token there is not the pivot.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        parts = text.split()
        merged: list[str] = []
        for part in parts:
            if part == "." and merged:
                merged[-1] = merged[-1] + "."
            else:
                merged.append(part)
        return [self._id(t) for t in merged]


def build_row_from_group(rows, tokenizer, **kw):
    """Single-row helper: most tests use groups that form exactly one branch."""
    out = build_rows_from_group(rows, tokenizer, **kw)
    return out[0] if out else None


def pts_row(qid, query, context, token, token_id=None, delta=0.5, positive=True):
    return {
        "dataset_item_id": qid,
        "query": query,
        "pivot_context": context,
        "pivot_token": token,
        "pivot_token_id": token_id,
        "prob_delta": delta,
        "is_positive": positive,
    }


QUERY = "a b c"                      # 3 query tokens: indices 0,1,2
ANSWER = "d e f g h i j"             # answer tokens start at index 3


# -- locate_pivot ----------------------------------------------------------


def test_locate_pivot_finds_the_index():
    tok = WordTokenizer()
    ctx = f"{QUERY} d e"
    assert locate_pivot(tok, ctx, " f") == 5  # a b c d e -> f is index 5


def test_locate_pivot_verifies_against_token_id():
    tok = WordTokenizer()
    ctx = f"{QUERY} d e"
    correct = tok.encode(f"{ctx} f")[5]
    assert locate_pivot(tok, ctx, " f", correct) == 5
    assert locate_pivot(tok, ctx, " f", correct + 999) is None


def test_locate_pivot_rejects_a_merging_boundary():
    """The exact case the old `len(encode(context)) - 1` arithmetic got wrong."""
    tok = MergingTokenizer()
    ctx = f"{QUERY} d e"
    # "e" + "." fuses into "e.", so the sequence does not grow.
    assert locate_pivot(tok, ctx, " .") is None


def test_locate_pivot_rejects_when_prefix_retokenizes():
    tok = MergingTokenizer()
    assert locate_pivot(tok, "a b", " .") is None


# -- the t-1 contract ------------------------------------------------------


def test_positive_label_is_immediately_before_the_pivot():
    """The single most important invariant in the pipeline."""
    tok = WordTokenizer()
    ctx = f"{QUERY} d e"           # pivot "f" sits at index 5
    row = build_row_from_group([pts_row("q1", QUERY, ctx, " f")], tok)

    assert row is not None
    assert row.positions(LABEL_PIVOTAL) == [4], "label must be at pivot_index - 1"
    assert row.token_ids[5] == tok.encode(" f")[0], "index 5 is the pivot itself"


def test_multiple_pivots_in_one_question():
    tok = WordTokenizer()
    rows = [
        pts_row("q1", QUERY, f"{QUERY} d", " e"),          # pivot at 4 -> label 3
        pts_row("q1", QUERY, f"{QUERY} d e f", " g"),      # pivot at 6 -> label 5
        pts_row("q1", QUERY, f"{QUERY} d e f g h", " i"),  # pivot at 8 -> label 7
    ]
    row = build_row_from_group(rows, tok, negative_to_positive_ratio=0.0)
    assert row is not None
    assert row.positions(LABEL_PIVOTAL) == [3, 5, 7]


def test_diverging_rollouts_each_become_their_own_row():
    """PTS samples several rollouts per question and they diverge. Collapsing
    to one 'longest' sequence discarded 74% of real pivots on the codelion
    data; each branch must survive as its own row."""
    tok = WordTokenizer()
    stats = BuildStats()
    rows = [
        pts_row("q1", QUERY, f"{QUERY} d e f g h", " i"),  # branch A
        pts_row("q1", QUERY, f"{QUERY} X Y", " Z"),        # branch B, diverges at once
    ]
    built = build_rows_from_group(rows, tok, negative_to_positive_ratio=0.0, stats=stats)

    assert len(built) == 2, "both rollouts must be represented"
    assert stats.branches == 2
    # branch A "a b c d e f g h i": pivot 'i' at 8 -> label 7
    # branch B "a b c X Y Z":       pivot 'Z' at 5 -> label 4
    assert {r.positions(LABEL_PIVOTAL)[0] for r in built} == {7, 4}
    assert stats.pivotal_positions == 2


def test_a_shared_prefix_is_labelled_only_once():
    """Branches share a prefix, and the residual stream at a shared position
    is bit-identical, so labelling it on both branches would duplicate the
    same training vector."""
    tok = WordTokenizer()
    rows = [
        # Same pivot at index 4 reachable from two different continuations.
        pts_row("q1", QUERY, f"{QUERY} d", " e"),
        pts_row("q1", QUERY, f"{QUERY} d e f", " g"),
        pts_row("q1", QUERY, f"{QUERY} d e X", " Y"),
    ]
    built = build_rows_from_group(rows, tok, negative_to_positive_ratio=0.0)

    labelled = [(r.token_ids[: p + 1]) for r in built for p in r.positions(LABEL_PIVOTAL)]
    assert len(labelled) == len({tuple(c) for c in labelled}), "duplicate context labelled"


def test_pivot_at_index_zero_is_dropped():
    """There is no t-1 inside the sequence for a pivot at position 0."""
    tok = WordTokenizer()
    stats = BuildStats()
    row = build_row_from_group([pts_row("q1", "", "", "a")], tok, stats=stats)
    assert row is None
    assert stats.dropped_pivot_at_zero == 1


# -- negatives -------------------------------------------------------------


def test_default_ratio_is_balanced():
    """User intent is a 1:1 set; the old code defaulted to 2:1."""
    tok = WordTokenizer()
    rows = [
        pts_row("q1", QUERY, f"{QUERY} d", " e"),
        pts_row("q1", QUERY, f"{QUERY} d e f g h i j k l", " m"),
    ]
    row = build_row_from_group(rows, tok)
    assert row is not None
    assert row.n_non_pivotal == row.n_pivotal


@pytest.mark.parametrize("ratio", [0.0, 1.0, 2.0, 3.0])
def test_ratio_is_honoured(ratio):
    tok = WordTokenizer()
    long_answer = " ".join(f"w{i}" for i in range(40))
    rows = [pts_row("q1", QUERY, f"{QUERY} {long_answer}", " END")]
    row = build_row_from_group(rows, tok, negative_to_positive_ratio=ratio)
    assert row is not None
    assert row.n_non_pivotal == int(round(row.n_pivotal * ratio))


def test_negatives_never_land_on_a_pivot_or_its_predecessor():
    tok = WordTokenizer()
    long_answer = " ".join(f"w{i}" for i in range(30))
    rows = [
        pts_row("q1", QUERY, f"{QUERY} {long_answer}", " END"),
        pts_row("q1", QUERY, f"{QUERY} w0 w1", " w2"),
    ]
    row = build_row_from_group(rows, tok, negative_to_positive_ratio=5.0)
    assert row is not None
    pivotal = set(row.positions(LABEL_PIVOTAL))
    negatives = set(row.positions(LABEL_NON_PIVOTAL))
    assert not (pivotal & negatives)
    # nor on the pivot tokens themselves
    assert not ({p + 1 for p in pivotal} & negatives)


def test_negatives_exclude_the_query_span_but_keep_its_last_token():
    """The last query token is a real t-1: the first answer token can be pivotal."""
    tok = WordTokenizer()
    long_answer = " ".join(f"w{i}" for i in range(30))
    rows = [pts_row("q1", QUERY, f"{QUERY} {long_answer}", " END")]
    row = build_row_from_group(rows, tok, negative_to_positive_ratio=20.0)
    assert row is not None
    assert row.answer_start == 2, "len(query)-1, so index 2 is eligible"
    assert min(row.positions(LABEL_NON_PIVOTAL)) >= 2
    assert all(row.labels[i] == LABEL_UNUSED for i in range(0, 2))


def test_negatives_are_not_correlated_across_questions():
    """The old builder reset the RNG per query with a constant seed, so every
    question drew the same offsets into its candidate list."""
    tok = WordTokenizer()
    long_answer = " ".join(f"w{i}" for i in range(40))
    picks = []
    for qid in ("q1", "q2", "q3", "q4"):
        rows = [pts_row(qid, QUERY, f"{QUERY} {long_answer}", " END")]
        row = build_row_from_group(rows, tok, negative_to_positive_ratio=4.0)
        assert row is not None
        picks.append(tuple(row.positions(LABEL_NON_PIVOTAL)))
    assert len(set(picks)) > 1, "identical negative offsets across questions"


def test_building_is_reproducible():
    tok = WordTokenizer()
    long_answer = " ".join(f"w{i}" for i in range(40))
    rows = [pts_row("q1", QUERY, f"{QUERY} {long_answer}", " END")]
    a = build_row_from_group(rows, WordTokenizer(), negative_to_positive_ratio=4.0)
    b = build_row_from_group(rows, tok, negative_to_positive_ratio=4.0)
    assert a is not None and b is not None
    assert a.labels == b.labels


# -- metadata --------------------------------------------------------------


def test_prob_delta_and_sign_are_carried_through():
    """The signed probe needs these; the original pipeline dropped them."""
    tok = WordTokenizer()
    rows = [
        pts_row("q1", QUERY, f"{QUERY} d", " e", delta=+0.42, positive=True),
        pts_row("q1", QUERY, f"{QUERY} d e f", " g", delta=-0.31, positive=False),
    ]
    row = build_row_from_group(rows, tok, negative_to_positive_ratio=0.0)
    assert row is not None
    assert row.prob_delta[3] == pytest.approx(0.42)
    assert row.is_positive[3] is True
    assert row.prob_delta[5] == pytest.approx(-0.31)
    assert row.is_positive[5] is False


def test_token_ids_are_authoritative_no_text_round_trip():
    """Storing decoded text and re-encoding it downstream is what shifted
    labels in the original pipeline; the row carries ids instead."""
    tok = WordTokenizer()
    rows = [pts_row("q1", QUERY, f"{QUERY} d e", " f")]
    row = build_row_from_group(rows, tok)
    assert row is not None
    assert row.token_ids == tok.encode(f"{QUERY} d e f")
    assert len(row.labels) == len(row.token_ids)
    assert not hasattr(row, "text")


# -- dataset level ---------------------------------------------------------


def test_build_dataset_groups_and_reports():
    tok = WordTokenizer()
    q2 = "p q r"
    rows = [
        pts_row("q1", QUERY, f"{QUERY} d e", " f"),
        pts_row("q1", QUERY, f"{QUERY} d", " e"),
        pts_row("q2", q2, f"{q2} d e f", " g"),
    ]
    built, stats = build_probe_dataset(rows, tok)
    assert len(built) == 2
    assert stats.groups == 2
    assert stats.rows_in == 3
    assert stats.as_dict()["drop_rate"] == 0.0


def test_questions_are_grouped_by_text_not_by_item_id():
    """codelion/Qwen3-0.6B-pts has one question under two ids (41 and 277).
    Keyed by id it would straddle the train/test split."""
    tok = WordTokenizer()
    rows = [
        pts_row("41", QUERY, f"{QUERY} d e", " f"),
        pts_row("277", QUERY, f"{QUERY} d", " e"),
    ]
    built, stats = build_probe_dataset(rows, tok)
    assert stats.groups == 1, "same question text must form one group"
    assert {r.query_id for r in built} == {"277+41"}


def test_one_id_covering_two_questions_is_split_apart():
    """The same data has 104 ids over 107 question texts, so an id can span
    two questions -- merging them would compute answer_start from the wrong
    query."""
    tok = WordTokenizer()
    rows = [
        pts_row("dup", "a b c", "a b c d", " e"),
        pts_row("dup", "x y", "x y z", " w"),
    ]
    _, stats = build_probe_dataset(rows, tok)
    assert stats.groups == 2


def test_split_is_at_the_question_level():
    """Splitting at the activation level would leak: positions within one
    question are not independent."""
    tok = WordTokenizer()
    rows = [
        pts_row(f"q{i}", f"q{i} a b", f"q{i} a b d e", " f") for i in range(10)
    ]
    built, _ = build_probe_dataset(rows, tok)
    train, test = split_by_query(built, test_size=0.3, seed=0)

    train_ids = {r.query_id for r in train}
    test_ids = {r.query_id for r in test}
    assert train_ids and test_ids
    assert not (train_ids & test_ids)
    assert len(train_ids | test_ids) == 10


# -- our own harness output ------------------------------------------------


def harness_event(uid, gen, position, seq, prompt_len, delta=0.5):
    return {
        "query_uid": uid,
        "generation_index": gen,
        "position": position,
        "token_id": seq[position],
        "sequence_token_ids": seq,
        "prompt_len": prompt_len,
        "prob_delta": delta,
        "is_positive": delta > 0,
    }


def test_harness_events_need_no_retokenization():
    """Our own events carry an absolute position, so t-1 is position-1
    exactly -- none of the BPE-merge recovery applies."""
    from probe_pipeline.probe_dataset import build_rows_from_harness_events

    seq = list(range(200, 220))
    rows, stats = build_rows_from_harness_events(
        [harness_event("q1", 0, 12, seq, prompt_len=4)]
    )
    assert len(rows) == 1
    assert rows[0].positions(LABEL_PIVOTAL) == [11]
    assert rows[0].token_ids[12] == seq[12]
    assert stats.located_from_position == 1
    assert stats.dropped_unlocatable == 0


def test_harness_rollouts_of_one_question_share_a_query_id():
    """Otherwise branches of the same question would straddle the split."""
    from probe_pipeline.probe_dataset import build_rows_from_harness_events

    seq_a, seq_b = list(range(200, 220)), list(range(300, 320))
    rows, stats = build_rows_from_harness_events(
        [
            harness_event("q1", 0, 12, seq_a, 4),
            harness_event("q1", 1, 9, seq_b, 4),
        ]
    )
    assert len(rows) == 2
    assert {r.query_id for r in rows} == {"q1"}
    assert {r.branch for r in rows} == {0, 1}
    assert stats.branches == 2


def test_harness_negatives_respect_the_ratio_and_exclusions():
    from probe_pipeline.probe_dataset import build_rows_from_harness_events

    seq = list(range(200, 260))
    rows, _ = build_rows_from_harness_events(
        [harness_event("q1", 0, 30, seq, 4)], negative_to_positive_ratio=3.0
    )
    row = rows[0]
    assert row.n_non_pivotal == 3
    negatives = set(row.positions(LABEL_NON_PIVOTAL))
    assert 29 not in negatives and 30 not in negatives
    assert min(negatives) >= 3  # prompt_len - 1


def test_harness_carries_sign_through():
    from probe_pipeline.probe_dataset import build_rows_from_harness_events

    seq = list(range(200, 220))
    rows, _ = build_rows_from_harness_events(
        [harness_event("q1", 0, 12, seq, 4, delta=-0.31)]
    )
    assert rows[0].prob_delta[11] == pytest.approx(-0.31)
    assert rows[0].is_positive[11] is False
