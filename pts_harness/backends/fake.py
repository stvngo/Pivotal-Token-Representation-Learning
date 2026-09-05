"""A backend with no model, for tests.

This is deliberately the first backend built. It makes the bisection state
machine, the wave scheduler, the checkpoint format and resume all testable in
under a second on a laptop -- without it every test needs a model download,
and tests that need a model download stop being run.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .base import Rollout, RolloutRequest, RolloutResult


class ScriptedBackend:
    """Returns canned completions keyed by the prefix.

    Two ways to script it:

    * ``script``: prefix tuple -> list of completion strings.
    * ``success_fn``: prefix tuple -> success probability, with completions
      synthesized so that exactly that fraction are solved. This is the
      convenient one for testing search behaviour, because it lets a test
      declare "the probability jumps at position 7" directly.
    """

    name = "scripted"

    def __init__(
        self,
        script: Mapping[tuple[int, ...], Sequence[str]] | None = None,
        success_fn: Callable[[tuple[int, ...]], float] | None = None,
        *,
        generation: Callable[[tuple[int, ...]], Sequence[int]] | None = None,
        success_marker: str = "####42",
        strict: bool = False,
    ) -> None:
        if script is None and success_fn is None:
            raise ValueError("provide either script or success_fn")
        self.script = dict(script or {})
        self.success_fn = success_fn
        self.generation = generation
        self.success_marker = success_marker
        self.strict = strict
        self.calls: list[RolloutRequest] = []
        self.n_rollouts = 0

    def generate(self, requests: Sequence[RolloutRequest]) -> list[RolloutResult]:
        out: list[RolloutResult] = []
        for req in requests:
            self.calls.append(req)
            self.n_rollouts += req.n
            prefix = tuple(req.prompt_token_ids)

            # A generation request (n == 1) wants token ids back, not a
            # success verdict -- the scheduler feeds them to the searcher.
            if req.request_id.startswith("gen:") and self.generation is not None:
                ids = tuple(self.generation(prefix))
                out.append(
                    RolloutResult(
                        request_id=req.request_id,
                        rollouts=[Rollout(text=self.detokenize(ids), token_ids=ids)],
                    )
                )
                continue

            if prefix in self.script:
                texts = list(self.script[prefix])[: req.n]
                texts += [""] * (req.n - len(texts))
            elif self.success_fn is not None:
                p = float(self.success_fn(prefix))
                # Deterministic split, so a repeated node returns the same
                # count and cache hits are indistinguishable from re-runs.
                k = int(round(p * req.n))
                texts = [self.success_marker] * k + ["nope"] * (req.n - k)
            elif self.strict:
                raise KeyError(f"no script entry for prefix of length {len(prefix)}")
            else:
                texts = ["nope"] * req.n

            out.append(
                RolloutResult(
                    request_id=req.request_id,
                    rollouts=[Rollout(text=t) for t in texts],
                )
            )
        return out

    def detokenize(self, ids: Sequence[int]) -> str:
        return " ".join(f"t{i}" for i in ids)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass
