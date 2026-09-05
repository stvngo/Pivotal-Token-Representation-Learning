"""Token PTS bisection, as an explicit state machine.

Upstream expresses the search as recursion (``TokenPTSSearcher.subdivide_sequence``),
which forces every probability estimate to block. That caps a GPU at the
~50 rollouts of a single node. Re-expressing it as a state machine lets a
scheduler collect the ready work of *many* queries into one batch, which is
where the throughput comes from -- the tree is sequential within a query but
queries are independent.

Semantics are matched to upstream at commit 8334808 (see
``docs/pts_semantics.md``):

* screen the query, skip unless ``min_prob <= p_base <= max_prob``;
* for each sampled rollout, bisect the generated span:
  a segment shorter than 2 tokens is terminal and is never scored during
  subdivision; otherwise score the segment's endpoints and split at the
  midpoint only if the probability moved by at least ``prob_threshold``;
* walk the terminal segments in order and emit an event for each
  single-token segment whose own delta clears the threshold.

Because a node's endpoints are its parent's, and estimates are keyed by
prefix, each bisection node costs exactly **one** new estimate -- its
midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence

from .backends.base import RolloutRequest
from .probability import ProbEstimate, node_key, node_seed


class Phase(str, Enum):
    BASELINE = "baseline"
    GENERATE = "generate"
    BISECT = "bisect"
    DONE = "done"


@dataclass(frozen=True)
class SearchConfig:
    num_samples: int = 40
    prob_threshold: float = 0.2
    min_prob: float = 0.2
    max_prob: float = 0.8
    max_generations: int = 1
    max_new_tokens: int = 320
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_generation_len: int = 5      # upstream skips rollouts shorter than this
    run_seed: int = 42
    significance_alpha: float | None = None


@dataclass
class PivotEvent:
    """One pivotal token, with the statistics behind the judgement."""

    query_uid: str
    generation_index: int
    position: int                 # absolute, prompt-inclusive
    token_id: int
    prefix_token_ids: list[int]
    sequence_token_ids: list[int]   # prompt + the whole searched rollout
    prompt_len: int
    before: ProbEstimate
    after: ProbEstimate
    baseline_p: float

    @property
    def prob_delta(self) -> float:
        return self.after.p - self.before.p

    @property
    def is_positive(self) -> bool:
        return self.prob_delta > 0

    def to_dict(self) -> dict:
        lo_b, hi_b = self.before.wilson_ci()
        lo_a, hi_a = self.after.wilson_ci()
        return {
            "query_uid": self.query_uid,
            "generation_index": self.generation_index,
            "position": self.position,
            "token_id": self.token_id,
            "prefix_len": len(self.prefix_token_ids),
            # The whole rollout, so downstream builds probe rows from exact
            # indices instead of recovering them by re-tokenizing text.
            "sequence_token_ids": self.sequence_token_ids,
            "prompt_len": self.prompt_len,
            "prob_before": self.before.p,
            "prob_after": self.after.p,
            "prob_delta": self.prob_delta,
            "is_positive": self.is_positive,
            "baseline_prob": self.baseline_p,
            # Sufficient statistics, so the threshold stays post-hoc.
            "n_before": self.before.n,
            "s_before": self.before.n_success,
            "n_after": self.after.n,
            "s_after": self.after.n_success,
            "ci_before": [lo_b, hi_b],
            "ci_after": [lo_a, hi_a],
        }


@dataclass
class _Interval:
    """Half-open span ``[lo, hi)`` of the generated sequence."""

    lo: int
    hi: int

    def __len__(self) -> int:
        return self.hi - self.lo


@dataclass
class QueryState:
    """Drives one query from screening to emitted events."""

    uid: str
    prompt_token_ids: list[int]
    cfg: SearchConfig
    model_key: str = "model"

    phase: Phase = Phase.BASELINE
    baseline: ProbEstimate | None = None
    generation_index: int = -1
    sequence: list[int] = field(default_factory=list)
    _pending: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _work: list[_Interval] = field(default_factory=list)
    _terminal: list[_Interval] = field(default_factory=list)
    _emitted: list[PivotEvent] = field(default_factory=list)
    _skipped_reason: str | None = None
    n_nodes: int = 0

    # -- keys ------------------------------------------------------------

    def _prefix(self, upto: int) -> tuple[int, ...]:
        """Prompt plus the first ``upto`` generated tokens."""
        return tuple(self.prompt_token_ids) + tuple(self.sequence[:upto])

    def _key(self, prefix: Sequence[int]) -> str:
        return node_key(self.model_key, self.uid, prefix)

    # -- requests --------------------------------------------------------

    def _request(self, prefix: tuple[int, ...]) -> RolloutRequest:
        key = self._key(prefix)
        return RolloutRequest(
            request_id=key,
            prompt_token_ids=prefix,
            n=self.cfg.num_samples,
            seed=node_seed(self.cfg.run_seed, self.uid, key),
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
        )

    def ready_requests(self, cache: dict[str, ProbEstimate]) -> list[RolloutRequest]:
        """Every estimate this query needs right now and does not have.

        Returns an empty list when the query is blocked on something other
        than probabilities (a rollout to be generated) or is finished.
        """
        self._pending.clear()
        if self.phase is Phase.DONE:
            return []

        if self.phase is Phase.BASELINE:
            prefix = tuple(self.prompt_token_ids)
            if self._key(prefix) in cache:
                return []
            self._pending[self._key(prefix)] = prefix
            return [self._request(prefix)]

        if self.phase is Phase.BISECT:
            wanted: dict[str, tuple[int, ...]] = {}
            for iv in self._work:
                for upto in (iv.lo, iv.hi):
                    prefix = self._prefix(upto)
                    key = self._key(prefix)
                    if key not in cache:
                        wanted[key] = prefix
            # Single-token terminal segments are scored at emit time.
            for iv in self._terminal:
                if len(iv) == 1:
                    for upto in (iv.lo, iv.hi):
                        prefix = self._prefix(upto)
                        key = self._key(prefix)
                        if key not in cache:
                            wanted[key] = prefix
            self._pending = wanted
            return [self._request(p) for p in wanted.values()]

        return []

    # -- generation ------------------------------------------------------

    def needs_generation(self) -> bool:
        return self.phase is Phase.GENERATE

    def generation_request(self) -> RolloutRequest:
        """A single sampled rollout to be searched."""
        prefix = tuple(self.prompt_token_ids)
        key = f"gen:{self.uid}:{self.generation_index + 1}"
        return RolloutRequest(
            request_id=key,
            prompt_token_ids=prefix,
            n=1,
            seed=node_seed(self.cfg.run_seed, self.uid, key),
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
        )

    def supply_generation(self, token_ids: Sequence[int]) -> None:
        """Install a sampled rollout and begin bisecting it."""
        self.generation_index += 1
        self.sequence = list(token_ids)
        self._terminal.clear()
        self._work.clear()

        if len(self.sequence) < self.cfg.min_generation_len:
            self._advance_generation()
            return

        self._work = [_Interval(0, len(self.sequence))]
        self.phase = Phase.BISECT

    def _advance_generation(self) -> None:
        if self.generation_index + 1 >= self.cfg.max_generations:
            self.phase = Phase.DONE
        else:
            self.phase = Phase.GENERATE

    # -- stepping --------------------------------------------------------

    def advance(self, cache: dict[str, ProbEstimate]) -> list[PivotEvent]:
        """Consume whatever the cache now holds and move the query forward."""
        events: list[PivotEvent] = []

        if self.phase is Phase.BASELINE:
            est = cache.get(self._key(tuple(self.prompt_token_ids)))
            if est is None:
                return events
            self.baseline = est
            if not (self.cfg.min_prob <= est.p <= self.cfg.max_prob):
                self._skipped_reason = f"baseline {est.p:.2f} outside band"
                self.phase = Phase.DONE
                return events
            self.phase = Phase.GENERATE
            return events

        if self.phase is Phase.BISECT:
            progressed = True
            while progressed and self._work:
                progressed = False
                still: list[_Interval] = []
                for iv in self._work:
                    before = cache.get(self._key(self._prefix(iv.lo)))
                    after = cache.get(self._key(self._prefix(iv.hi)))
                    if before is None or after is None:
                        still.append(iv)
                        continue

                    self.n_nodes += 1
                    progressed = True
                    if abs(after.p - before.p) < self.cfg.prob_threshold:
                        self._terminal.append(iv)      # pruned, kept whole
                        continue

                    mid = iv.lo + len(iv) // 2
                    for child in (_Interval(iv.lo, mid), _Interval(mid, iv.hi)):
                        if len(child) <= 1:
                            self._terminal.append(child)   # never scored here
                        else:
                            still.append(child)
                self._work = still

            if not self._work:
                events = self._emit(cache)
                if events is None:  # still waiting on emit-time estimates
                    return []
                self._advance_generation()

        return events

    def _emit(self, cache: dict[str, ProbEstimate]) -> list[PivotEvent] | None:
        """Score single-token segments and emit those that clear threshold.

        Returns ``None`` if any needed estimate is still missing, so the
        scheduler issues them on the next wave.
        """
        singles = sorted((iv for iv in self._terminal if len(iv) == 1), key=lambda i: i.lo)
        for iv in singles:
            if (
                self._key(self._prefix(iv.lo)) not in cache
                or self._key(self._prefix(iv.hi)) not in cache
            ):
                return None

        events: list[PivotEvent] = []
        for iv in singles:
            before = cache[self._key(self._prefix(iv.lo))]
            after = cache[self._key(self._prefix(iv.hi))]
            if abs(after.p - before.p) < self.cfg.prob_threshold:
                continue
            events.append(
                PivotEvent(
                    query_uid=self.uid,
                    generation_index=self.generation_index,
                    position=len(self.prompt_token_ids) + iv.lo,
                    token_id=self.sequence[iv.lo],
                    prefix_token_ids=list(self._prefix(iv.lo)),
                    sequence_token_ids=list(self.prompt_token_ids) + list(self.sequence),
                    prompt_len=len(self.prompt_token_ids),
                    before=before,
                    after=after,
                    baseline_p=self.baseline.p if self.baseline else 0.0,
                )
            )
        self._emitted.extend(events)
        return events

    # -- status ----------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self.phase is Phase.DONE

    @property
    def skipped(self) -> bool:
        return self._skipped_reason is not None

    def record(self) -> dict:
        return {
            "query_uid": self.uid,
            "phase": self.phase.value,
            "baseline_p": self.baseline.p if self.baseline else None,
            "skipped_reason": self._skipped_reason,
            "n_events": len(self._emitted),
            "n_nodes": self.n_nodes,
            "generations": self.generation_index + 1,
        }

    def events(self) -> Iterator[PivotEvent]:
        return iter(self._emitted)
