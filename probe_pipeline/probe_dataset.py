"""Build labelled probe rows from PTS events.

The probe predicts, from the residual stream at position *t-1*, whether the
token at *t* will be pivotal. So the label lives one position **before** the
pivot. That is what makes the probe usable during decoding: at step *t-1* the
prediction is available before the forward pass that commits to *t*.

Label convention, per token position of a canonical sequence:

===== =========================================================
   1  immediately precedes a pivotal token  (the positive class)
  -1  a sampled non-pivotal answer position (the negative class)
   0  unused -- present in the sequence, excluded from training
===== =========================================================

Two ways in, depending on where the events came from:

* **Native** (we ran PTS ourselves): events carry ``position``, the absolute
  prompt-inclusive index of the pivot token, so *t-1* is ``position - 1`` and
  nothing has to be inferred. See ``docs/pts_semantics.md`` section 4.1.
* **Legacy** (the released ``codelion/*-pts`` datasets): no ``position``
  field, so the index is recovered by re-tokenizing ``pivot_context``. That
  recovery is verified against ``pivot_token_id`` rather than assumed --
  see :func:`locate_pivot`, which is where the old pipeline silently went
  wrong on BPE merge boundaries.

What this module deliberately does differently from the original
``preprocess.py``:

* it emits **token ids**, never decoded text, so nothing downstream has to
  re-tokenize and risk a shifted alignment;
* it carries ``prob_delta`` through, which the signed probe needs and the
  old pipeline dropped;
* it seeds the RNG **per query**, so negatives are not correlated across
  questions;
* it counts what it drops instead of discarding silently.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

# Positions labelled 1 / -1 / 0.
LABEL_PIVOTAL = 1
LABEL_NON_PIVOTAL = -1
LABEL_UNUSED = 0


class TokenizerLike(Protocol):
    """The only tokenizer surface this module needs."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


# --------------------------------------------------------------------------
# schema normalization
# --------------------------------------------------------------------------

# Three shapes exist in the wild and we have to read all of them:
#
#   v1        pivot_context / pivot_token / pivot_token_id, no position.
#             What the original notebooks consumed.
#   v2-migrated  context / label / token_id, position is NULL, and the v1
#             fields survive under metadata. This is what
#             `codelion/Qwen3-0.6B-pts` serves today -- the same 1,245 token
#             events as v1, re-wrapped (metadata.migrated_from ==
#             "v1_pivotal_token"). Because position is null, the index still
#             has to be recovered by re-tokenizing.
#   v2-native  context / label / token_id with a real integer position. What
#             our own PTS runs emit, and the only shape where the pivot index
#             needs no inference at all.


def normalize_pts_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Map any PTS row shape onto a common set of fields.

    Returns ``None`` for rows that are not token-granularity events -- the
    published datasets interleave latent and sentence events in the same
    split, and their ``prob_delta`` is null by design.
    """
    granularity = row.get("granularity")
    event_type = row.get("event_type")
    if granularity is not None and granularity != "token":
        return None
    if event_type is not None and event_type not in ("pivotal_token", None):
        return None

    meta = row.get("metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {}

    context = row.get("pivot_context") or meta.get("pivot_context") or row.get("context")
    token = row.get("pivot_token")
    if token is None:
        token = meta.get("pivot_token")
    if token is None:
        token = row.get("label")
    if context is None or token is None:
        return None

    token_id = row.get("pivot_token_id")
    if token_id is None:
        token_id = row.get("token_id")

    position = row.get("position")

    return {
        "dataset_item_id": str(row.get("dataset_item_id", "unknown")),
        "query": row.get("query", ""),
        "pivot_context": context,
        "pivot_token": token,
        "pivot_token_id": int(token_id) if token_id is not None else None,
        "position": int(position) if position is not None else None,
        "prob_delta": float(row.get("prob_delta") or 0.0),
        "prob_before": float(row.get("prob_before") or 0.0),
        "prob_after": float(row.get("prob_after") or 0.0),
        "is_positive": bool(row.get("is_positive", False)),
    }


def normalize_pts_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and keep only token-granularity events."""
    out = []
    for row in rows:
        norm = normalize_pts_row(row)
        if norm is not None:
            out.append(norm)
    return out


# --------------------------------------------------------------------------
# locating the pivot
# --------------------------------------------------------------------------


def locate_pivot(
    tokenizer: TokenizerLike,
    pivot_context: str,
    pivot_token: str,
    pivot_token_id: int | None = None,
) -> int | None:
    """Absolute index of ``pivot_token`` within ``encode(context + token)``.

    Returns ``None`` when the index cannot be established, which happens for a
    real and non-rare reason: BPE does not guarantee
    ``encode(a + b) == encode(a) + encode(b)``. A pivot token like ``"."`` or
    ``"'"`` appended to a context ending in a word frequently re-merges with
    it, so the naive ``len(encode(context))`` is then off by one *and* points
    at a token that contains the pivot rather than being it.

    Rather than assume, we verify three things: the prefix must survive
    re-tokenization intact, the sequence must actually grow, and -- when the
    dataset gives us ``pivot_token_id``, which the old pipeline never read --
    the token at the computed index must be the pivot.
    """
    ctx_ids = list(tokenizer.encode(pivot_context, add_special_tokens=False))
    full_ids = list(tokenizer.encode(pivot_context + pivot_token, add_special_tokens=False))

    if len(full_ids) <= len(ctx_ids):
        return None  # the token vanished into a merge
    if full_ids[: len(ctx_ids)] != ctx_ids:
        return None  # appending re-tokenized the prefix

    position = len(ctx_ids)
    if pivot_token_id is not None and full_ids[position] != int(pivot_token_id):
        return None  # index does not point at the pivot
    return position


# --------------------------------------------------------------------------
# the row
# --------------------------------------------------------------------------


@dataclass
class ProbeRow:
    """One canonical sequence with per-position labels.

    ``token_ids`` is authoritative. Nothing downstream should re-tokenize.
    """

    query_id: str
    token_ids: list[int]
    labels: list[int]
    answer_start: int
    prob_delta: list[float]
    is_positive: list[bool]
    branch: int = 0

    def positions(self, label: int) -> list[int]:
        return [i for i, v in enumerate(self.labels) if v == label]

    @property
    def n_pivotal(self) -> int:
        return sum(1 for v in self.labels if v == LABEL_PIVOTAL)

    @property
    def n_non_pivotal(self) -> int:
        return sum(1 for v in self.labels if v == LABEL_NON_PIVOTAL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "branch": self.branch,
            "token_ids": self.token_ids,
            "labels": self.labels,
            "answer_start": self.answer_start,
            "prob_delta": self.prob_delta,
            "is_positive": self.is_positive,
        }


@dataclass
class BuildStats:
    """What the builder saw and what it threw away.

    The old pipeline dropped divergent rows silently, so nobody knew whether
    the rate was 1% or 30%. Now it is counted, and a high rate is a signal
    that the source data does not match this builder's assumptions.
    """

    groups: int = 0
    branches: int = 0
    rows_in: int = 0
    rows_used: int = 0
    located_from_position: int = 0
    located_by_retokenizing: int = 0
    dropped_unlocatable: int = 0
    dropped_not_a_prefix: int = 0
    dropped_pivot_at_zero: int = 0
    groups_empty: int = 0
    pivotal_positions: int = 0
    non_pivotal_positions: int = 0
    per_query_pivots: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "per_query_pivots"}
        d["mean_pivots_per_query"] = (
            float(np.mean(self.per_query_pivots)) if self.per_query_pivots else 0.0
        )
        d["drop_rate"] = (
            (self.dropped_unlocatable + self.dropped_not_a_prefix + self.dropped_pivot_at_zero)
            / self.rows_in
            if self.rows_in
            else 0.0
        )
        return d


def _query_rng(seed: int, query_id: str) -> np.random.Generator:
    """A generator unique to this query.

    The original builder constructed ``default_rng(seed)`` inside the
    per-query function with the same seed every time, so every question drew
    the same offsets into its candidate list and the negatives were
    systematically correlated across the dataset.
    """
    digest = hashlib.blake2b(f"{seed}|{query_id}".encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def build_rows_from_group(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: TokenizerLike,
    *,
    negative_to_positive_ratio: float = 1.0,
    seed: int = 42,
    stats: BuildStats | None = None,
    query_id: str | None = None,
) -> list[ProbeRow]:
    """Build one :class:`ProbeRow` per independent rollout branch.

    PTS samples ``max_generations`` rollouts per question, and they diverge
    from one another after a few tokens. Collapsing a question to a single
    "longest" sequence -- what the original builder did -- therefore discards
    every pivot that lives on a diverging branch. Measured on
    ``codelion/Qwen3-0.6B-pts``: 104 questions carry 1,245 pivotal tokens
    across 447 distinct branches, and keeping only the longest branch throws
    away **74%** of them.

    So the unit here is the branch: every maximal token sequence in the group
    becomes its own row, and a pivot is labelled on each branch it lies on.

    Deduplication matters because branches share prefixes. The residual
    stream at position *p* depends only on ``token_ids[:p+1]``, so the same
    prefix on two branches yields a bit-identical activation. Each distinct
    context is therefore labelled exactly once, keyed by the prefix itself --
    otherwise shared early positions would be silently duplicated into the
    training set and over-weighted.
    """
    stats = stats if stats is not None else BuildStats()
    if not rows:
        return []

    if query_id is None:
        query_id = "+".join(
            sorted({str(r.get("dataset_item_id", "unknown")) for r in rows})
        )

    # Locate every pivot, keeping the encodings we already paid for.
    located: list[tuple[int, list[int], Mapping[str, Any]]] = []
    for row in rows:
        # Native v2 events carry the absolute index, so nothing is inferred.
        # Migrated and v1 events do not, and must be recovered by
        # re-tokenizing -- verified, not assumed. See docs/pts_semantics.md.
        pos = row.get("position")
        if pos is not None:
            pos = int(pos)
            stats.located_from_position += 1
        else:
            pos = locate_pivot(
                tokenizer,
                row["pivot_context"],
                row["pivot_token"],
                row.get("pivot_token_id"),
            )
            if pos is not None:
                stats.located_by_retokenizing += 1
        if pos is None:
            stats.dropped_unlocatable += 1
            continue
        if pos == 0:
            # No t-1 exists inside the sequence for a pivot at index 0.
            stats.dropped_pivot_at_zero += 1
            continue
        full_ids = list(
            tokenizer.encode(row["pivot_context"] + row["pivot_token"], add_special_tokens=False)
        )
        located.append((pos, full_ids, row))

    if not located:
        stats.groups_empty += 1
        return []

    # Maximal branches: sequences that are not a proper prefix of another.
    all_seqs = [tuple(ids) for _, ids, _ in located]
    branches: list[tuple[int, ...]] = []
    for seq in dict.fromkeys(all_seqs):  # dedupe, preserve order
        if not any(other != seq and other[: len(seq)] == seq for other in all_seqs):
            branches.append(seq)
    # Deterministic order: longest first, then by content, so dedupe is stable.
    branches.sort(key=lambda s: (-len(s), s))

    query_ids = list(tokenizer.encode(rows[0].get("query", ""), add_special_tokens=False))
    # The last query token is a legitimate t-1: PTS rows exist where the very
    # first answer token is pivotal, and its predecessor is that token.
    answer_start = max(0, len(query_ids) - 1)

    rng = _query_rng(seed, query_id)
    seen_contexts: set[tuple[int, ...]] = set()
    out: list[ProbeRow] = []

    for branch_idx, branch in enumerate(branches):
        # Pivots lying on this branch: their whole sequence must be a prefix.
        pivot_meta: dict[int, tuple[float, bool]] = {}
        for pos, ids, row in located:
            if branch[: len(ids)] != tuple(ids):
                continue
            pivot_meta[pos] = (
                float(row.get("prob_delta", 0.0) or 0.0),
                bool(row.get("is_positive", False)),
            )

        if not pivot_meta:
            continue

        pivot_positions = set(pivot_meta)
        t_minus_1 = {p - 1 for p in pivot_positions if p >= 1}

        # Label a context only once across branches: the activation at p
        # depends solely on branch[:p+1], so a shared prefix is the same
        # vector and would otherwise enter training several times.
        fresh_labels = {p for p in t_minus_1 if branch[: p + 1] not in seen_contexts}
        if not fresh_labels:
            continue
        for p in fresh_labels:
            seen_contexts.add(branch[: p + 1])

        excluded = t_minus_1 | pivot_positions
        candidates = [
            i
            for i in range(answer_start, len(branch))
            if i not in excluded and branch[: i + 1] not in seen_contexts
        ]

        n_take = min(max(int(round(len(fresh_labels) * negative_to_positive_ratio)), 0), len(candidates))
        sampled: set[int] = set()
        if n_take > 0:
            sampled = {int(i) for i in rng.choice(np.array(candidates), size=n_take, replace=False)}
            for p in sampled:
                seen_contexts.add(branch[: p + 1])

        labels = [LABEL_UNUSED] * len(branch)
        prob_delta = [0.0] * len(branch)
        is_positive = [False] * len(branch)

        for p in sorted(fresh_labels):
            labels[p] = LABEL_PIVOTAL
            delta, positive = pivot_meta.get(p + 1, (0.0, False))
            prob_delta[p] = delta
            is_positive[p] = positive
        for p in sampled:
            if labels[p] != LABEL_PIVOTAL:
                labels[p] = LABEL_NON_PIVOTAL

        row_out = ProbeRow(
            query_id=query_id,
            token_ids=list(branch),
            labels=labels,
            answer_start=answer_start,
            prob_delta=prob_delta,
            is_positive=is_positive,
            branch=branch_idx,
        )
        stats.rows_used += len(fresh_labels)
        stats.pivotal_positions += row_out.n_pivotal
        stats.non_pivotal_positions += row_out.n_non_pivotal
        stats.per_query_pivots.append(row_out.n_pivotal)
        stats.branches += 1
        out.append(row_out)

    if not out:
        stats.groups_empty += 1
    return out


def group_by_query(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Group PTS rows by the **question text**, not by ``dataset_item_id``.

    The id is not a reliable key in the published data. On
    ``codelion/Qwen3-0.6B-pts`` there are 104 distinct ``dataset_item_id``
    values but 107 distinct question texts, and one question appears under
    two different ids (41 and 277). Both directions are harmful:

    * one text under two ids puts the *same* question on both sides of a
      train/test split -- a leak, and it is how a duplicated labelled context
      first showed up;
    * one id over two texts merges two different questions into one group,
      whose ``answer_start`` would then be computed from the wrong query.

    Keying on the normalized text fixes both. The returned key is the joined
    set of source ids so grouping stays traceable in logs.
    """
    by_text: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_text[" ".join(str(row.get("query", "")).split())].append(row)

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for text, group in sorted(by_text.items()):
        ids = sorted({str(r.get("dataset_item_id", "unknown")) for r in group})
        key = "+".join(ids)
        # Two different questions can share an id, so the joined id is not
        # unique on its own; disambiguate rather than silently overwrite.
        if key in grouped:
            suffix = 2
            while f"{key}#{suffix}" in grouped:
                suffix += 1
            key = f"{key}#{suffix}"
        grouped[key] = group
    return grouped


def build_probe_dataset(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: TokenizerLike,
    *,
    negative_to_positive_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[list[ProbeRow], BuildStats]:
    """Build labelled rows for every question in ``rows``."""
    grouped = group_by_query(rows)
    stats = BuildStats(groups=len(grouped), rows_in=sum(len(v) for v in grouped.values()))

    out: list[ProbeRow] = []
    for query_id in sorted(grouped):
        out.extend(
            build_rows_from_group(
                grouped[query_id],
                tokenizer,
                negative_to_positive_ratio=negative_to_positive_ratio,
                seed=seed,
                stats=stats,
                query_id=query_id,
            )
        )
    return out, stats


def split_by_query(
    rows: Sequence[ProbeRow],
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[ProbeRow], list[ProbeRow]]:
    """Split at the **question** level.

    Positions drawn from one question are not independent, so splitting at the
    activation level would leak. This is the one thing the original pipeline
    got right and it is preserved deliberately.
    """
    ids = sorted({r.query_id for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)  # type: ignore[arg-type]
    n_test = max(1, int(round(len(ids) * test_size))) if ids else 0
    test_ids = set(ids[:n_test])
    train = [r for r in rows if r.query_id not in test_ids]
    test = [r for r in rows if r.query_id in test_ids]
    return train, test


# --------------------------------------------------------------------------
# our own harness output
# --------------------------------------------------------------------------


def build_rows_from_harness_events(
    events: Iterable[Mapping[str, Any]],
    *,
    negative_to_positive_ratio: float = 1.0,
    seed: int = 42,
    answer_start_from_prompt: bool = True,
    label_offset: int = -1,
    negatives_within_pivot_span: bool = True,
) -> tuple[list[ProbeRow], BuildStats]:
    """Build probe rows from ``pts_harness`` events.

    Simpler and exact, compared with the released-dataset path: our events
    carry ``sequence_token_ids`` and an absolute ``position``, so t-1 is
    ``position - 1`` and nothing is recovered by re-tokenizing. No BPE merge
    hazard, no string prefix matching, no dropped rows.

    One row per (question, searched rollout). Rollouts of the same question
    keep the same ``query_id`` so the train/test split stays question-level.

    ``negatives_within_pivot_span`` restricts negatives to the span where
    pivots actually occur, and defaults to on because the alternative
    silently rigs every baseline comparison. Our rollouts run to the token
    limit, so sampling negatives from the whole sequence draws most of them
    from the fluent tail *after* the model has committed to an answer:
    measured on this data, 63% of negatives landed after the last pivot,
    with mean next-token entropy 0.275 against 0.476 before it. That inflates
    the entropy baseline from a +0.44 to a +1.14 separation and confounds
    position with label (negative median position 232 vs pivot median 146).
    The released datasets do not have this problem only because their stored
    sequence stops at the deepest pivot.

    ``label_offset`` selects which position carries the positive label,
    relative to the pivot. ``-1`` (the default) is the prediction problem:
    read at *t-1*, before the token exists, which is the only version usable
    during decoding. ``0`` reads at the pivot itself, which is a post-hoc
    classifier and cannot gate anything -- useful only as a diagnostic, to
    see how much of the signal is in the token rather than its context.
    """
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for ev in events:
        grouped[(str(ev["query_uid"]), int(ev.get("generation_index", 0)))].append(ev)

    stats = BuildStats(
        groups=len({k[0] for k in grouped}),
        rows_in=sum(len(v) for v in grouped.values()),
    )
    out: list[ProbeRow] = []

    for (uid, gen_idx), evs in sorted(grouped.items()):
        tokens = list(evs[0]["sequence_token_ids"])
        prompt_len = int(evs[0].get("prompt_len", 0))
        # The last prompt token is a legitimate t-1: the first generated
        # token can itself be pivotal.
        answer_start = max(0, prompt_len - 1) if answer_start_from_prompt else 0

        pivot_positions: set[int] = set()
        meta: dict[int, tuple[float, bool]] = {}
        for ev in evs:
            pos = int(ev["position"])
            if pos < 1 or pos >= len(tokens):
                stats.dropped_pivot_at_zero += 1
                continue
            pivot_positions.add(pos)
            meta[pos] = (float(ev.get("prob_delta", 0.0)), bool(ev.get("is_positive", False)))
            stats.located_from_position += 1
            stats.rows_used += 1

        t_minus_1 = {
            p + label_offset
            for p in pivot_positions
            if 0 <= p + label_offset < len(tokens)
        }
        if not t_minus_1:
            stats.groups_empty += 1
            continue

        excluded = t_minus_1 | pivot_positions
        span_end = (max(pivot_positions) + 1) if negatives_within_pivot_span else len(tokens)
        candidates = [i for i in range(answer_start, span_end) if i not in excluded]
        rng = _query_rng(seed, f"{uid}#{gen_idx}")
        n_take = min(max(int(round(len(t_minus_1) * negative_to_positive_ratio)), 0), len(candidates))
        sampled = (
            {int(i) for i in rng.choice(np.array(candidates), size=n_take, replace=False)}
            if n_take > 0
            else set()
        )

        labels = [LABEL_UNUSED] * len(tokens)
        prob_delta = [0.0] * len(tokens)
        is_positive = [False] * len(tokens)
        for p in sorted(t_minus_1):
            labels[p] = LABEL_PIVOTAL
            prob_delta[p], is_positive[p] = meta.get(p - label_offset, (0.0, False))
        for p in sampled:
            if labels[p] != LABEL_PIVOTAL:
                labels[p] = LABEL_NON_PIVOTAL

        row = ProbeRow(
            query_id=uid,
            token_ids=tokens,
            labels=labels,
            answer_start=answer_start,
            prob_delta=prob_delta,
            is_positive=is_positive,
            branch=gen_idx,
        )
        stats.branches += 1
        stats.pivotal_positions += row.n_pivotal
        stats.non_pivotal_positions += row.n_non_pivotal
        stats.per_query_pivots.append(row.n_pivotal)
        out.append(row)

    return out, stats
