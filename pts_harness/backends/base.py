"""The rollout backend contract.

Token ids in, token ids out. Upstream passes decoded strings to the oracle
and then recovers positions by re-tokenizing, which is the round-trip that
shifts labels (``docs/issues.md`` Issue #9). Carrying ids end-to-end removes
that failure mode by construction, and lets emitted events record an exact
``position``.

Batching is the backend's job, not the scheduler's: the scheduler hands over
every request that is ready across all in-flight queries, and a continuous
batcher decides how to pack them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class RolloutRequest:
    """``n`` sampled continuations of one prefix."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    n: int
    seed: int
    max_new_tokens: int = 320
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")


@dataclass(frozen=True)
class Rollout:
    text: str
    token_ids: tuple[int, ...] = ()
    finish_reason: str = "length"


@dataclass
class RolloutResult:
    request_id: str
    rollouts: list[Rollout] = field(default_factory=list)


@runtime_checkable
class RolloutBackend(Protocol):
    """Anything that can turn prefixes into sampled continuations."""

    name: str

    def generate(self, requests: Sequence[RolloutRequest]) -> list[RolloutResult]:
        """Complete every request. Order of the returned list is not
        significant; results are matched by ``request_id``."""
        ...

    def detokenize(self, ids: Sequence[int]) -> str:
        ...

    def close(self) -> None:
        ...
