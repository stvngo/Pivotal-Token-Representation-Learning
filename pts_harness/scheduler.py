"""Wave scheduler: many queries' bisection frontiers in one batch.

The bisection tree is sequential *within* a query -- both children of a node
depend on the node's own midpoint estimate. But queries are independent, so
running many concurrently keeps a deep batch available at every step. That,
not speculation, is where the throughput comes from: speculatively evaluating
grandchildren would discard 50-70% of its work whenever a branch prunes,
whereas cross-query concurrency wastes nothing.

The scheduler deliberately does *not* form batches. It hands the backend
every request that is ready and lets a continuous batcher pack them, which
is why a synchronous loop suffices and no asyncio is involved -- a
considerable simplification for checkpointing and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from .backends.base import RolloutBackend, RolloutRequest, RolloutResult
from .checkpoint import RunStore
from .probability import ProbEstimate
from .search import PivotEvent, QueryState, SearchConfig


@dataclass
class QuerySpec:
    """One question to search."""

    uid: str
    prompt_token_ids: list[int]
    query: str = ""
    answer: str = ""


@dataclass
class RunSummary:
    queries_attempted: int = 0
    queries_completed: int = 0
    queries_skipped: int = 0
    events: int = 0
    waves: int = 0
    rollouts: int = 0
    nodes: int = 0
    seconds: float = 0.0
    resumed_from_cache: int = 0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["rollouts_per_event"] = (
            self.rollouts / self.events if self.events else float("inf")
        )
        return d


class WaveScheduler:
    """Drive a pool of :class:`QueryState` machines against one backend."""

    def __init__(
        self,
        backend: RolloutBackend,
        oracle: Callable[[str, str], bool],
        cfg: SearchConfig,
        *,
        store: RunStore | None = None,
        max_active: int = 64,
        model_key: str = "model",
        on_wave: Callable[[dict], None] | None = None,
    ) -> None:
        self.backend = backend
        self.oracle = oracle
        self.cfg = cfg
        self.store = store
        self.max_active = max_active
        self.model_key = model_key
        self.on_wave = on_wave
        self.cache: dict[str, ProbEstimate] = {}

    # -- scoring ---------------------------------------------------------

    def _score(
        self, results: Sequence[RolloutResult], specs: dict[str, QuerySpec], owner: dict[str, str]
    ) -> list[ProbEstimate]:
        """Turn rollouts into success counts via the oracle."""
        out: list[ProbEstimate] = []
        for res in results:
            spec = specs.get(owner.get(res.request_id, ""))
            query_text = spec.query if spec else ""
            n_success = sum(1 for r in res.rollouts if self.oracle(query_text, r.text))
            est = ProbEstimate(
                key=res.request_id, n=len(res.rollouts), n_success=n_success
            )
            self.cache[est.key] = est
            out.append(est)
        return out

    # -- main loop -------------------------------------------------------

    def run(self, specs: Iterable[QuerySpec]) -> RunSummary:
        specs = list(specs)
        by_uid = {s.uid: s for s in specs}
        summary = RunSummary(queries_attempted=len(specs))
        started = time.time()

        if self.store is not None:
            self.cache.update(self.store.load_prob_cache())
            summary.resumed_from_cache = len(self.cache)
            done = self.store.completed_ids()
            specs = [s for s in specs if s.uid not in done]

        pending: list[QuerySpec] = list(specs)
        active: list[QueryState] = []

        while pending or active:
            while pending and len(active) < self.max_active:
                spec = pending.pop(0)
                active.append(
                    QueryState(
                        uid=spec.uid,
                        prompt_token_ids=list(spec.prompt_token_ids),
                        cfg=self.cfg,
                        model_key=self.model_key,
                    )
                )
            if not active:
                break

            # Collect the wave: probability estimates and fresh rollouts.
            requests: list[RolloutRequest] = []
            owner: dict[str, str] = {}
            gen_owner: dict[str, QueryState] = {}

            for qs in active:
                for req in qs.ready_requests(self.cache):
                    requests.append(req)
                    owner[req.request_id] = qs.uid
                if qs.needs_generation():
                    req = qs.generation_request()
                    requests.append(req)
                    owner[req.request_id] = qs.uid
                    gen_owner[req.request_id] = qs

            if not requests:
                # Nothing to ask for and nothing generating: everything that
                # can move has moved. Advance once more, then drop finished.
                for qs in active:
                    qs.advance(self.cache)
                active = [qs for qs in active if not qs.is_done]
                if not active:
                    continue
                break

            results = self.backend.generate(requests)
            summary.rollouts += sum(len(r.rollouts) for r in results)
            summary.waves += 1

            gen_results = [r for r in results if r.request_id in gen_owner]
            prob_results = [r for r in results if r.request_id not in gen_owner]

            new_estimates = self._score(prob_results, by_uid, owner)
            if self.store is not None and new_estimates:
                self.store.record_probs(new_estimates)

            for res in gen_results:
                qs = gen_owner[res.request_id]
                tokens = res.rollouts[0].token_ids if res.rollouts else ()
                qs.supply_generation(list(tokens))

            wave_events: list[PivotEvent] = []
            for qs in active:
                wave_events.extend(qs.advance(self.cache))

            if self.store is not None:
                if wave_events:
                    self.store.record_events(wave_events)
                for qs in active:
                    if qs.is_done:
                        self.store.complete_query(qs.record())
                self.store.flush()      # the checkpoint boundary
            summary.events += len(wave_events)

            finished = [qs for qs in active if qs.is_done]
            summary.queries_completed += len(finished)
            summary.queries_skipped += sum(1 for qs in finished if qs.skipped)
            summary.nodes += sum(qs.n_nodes for qs in finished)
            active = [qs for qs in active if not qs.is_done]

            if self.on_wave is not None:
                self.on_wave(
                    {
                        "wave": summary.waves,
                        "active": len(active),
                        "pending": len(pending),
                        "events": summary.events,
                        "rollouts": summary.rollouts,
                    }
                )
            if self.store is not None:
                self.store.heartbeat(
                    {
                        "wave": summary.waves,
                        "active": len(active),
                        "pending": len(pending),
                        "events": summary.events,
                        "elapsed_s": round(time.time() - started, 1),
                    }
                )

        summary.seconds = round(time.time() - started, 1)
        return summary
