"""Success-probability estimates, stored as sufficient statistics.

The key decision: a node records ``(n, n_success)``, not ``p_hat`` and not
the accept/reject verdict. Three things fall out of that.

* ``prob_threshold`` becomes a **post-hoc** knob -- the event set can be
  re-derived at several thresholds with no GPU.
* Every event can carry a Wilson interval and a two-proportion test, which
  matters because PTS's own labels are noisy: at the reference setting
  (S=50, tau=0.2) the acceptance threshold sits at only 2 sigma, so roughly
  4.6% of tested positions pass by chance at p=0.5.
* Adaptive sampling can be added later without changing the on-disk format.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence


def node_key(model_key: str, query_uid: str, prefix_token_ids: Sequence[int]) -> str:
    """Content-addressed id for one probability estimate.

    Keyed by the *token ids* of the conditioning prefix, so two paths that
    reach the same prefix share the estimate -- which is what makes the
    bisection cost one new estimate per node rather than two.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(model_key.encode("utf-8"))
    h.update(b"\x00")
    h.update(query_uid.encode("utf-8"))
    h.update(b"\x00")
    h.update(",".join(map(str, prefix_token_ids)).encode("utf-8"))
    return h.hexdigest()


def node_seed(run_seed: int, query_uid: str, key: str) -> int:
    """Deterministic per-node sampling seed.

    vLLM seeds per request, unlike HF's global RNG, so a node's samples do
    not depend on what else happened to be in the batch. That is what makes
    a resumed run reproduce the original.
    """
    h = hashlib.blake2b(
        f"{run_seed}|{query_uid}|{key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


@dataclass
class ProbEstimate:
    """``n_success`` out of ``n`` rollouts solved the task."""

    key: str
    n: int
    n_success: int
    prefix_len: int = 0

    @property
    def p(self) -> float:
        return self.n_success / self.n if self.n else 0.0

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval -- well behaved at p near 0 or 1."""
        if self.n == 0:
            return 0.0, 1.0
        p, n = self.p, self.n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    def to_dict(self) -> dict:
        return {
            "k": self.key,
            "n": self.n,
            "s": self.n_success,
            "plen": self.prefix_len,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProbEstimate":
        return cls(key=d["k"], n=int(d["n"]), n_success=int(d["s"]), prefix_len=int(d.get("plen", 0)))


def two_proportion_p(before: ProbEstimate, after: ProbEstimate) -> float:
    """Two-sided p-value for ``p_after != p_before`` (pooled z-test)."""
    n1, n2 = before.n, after.n
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (before.n_success + after.n_success) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(after.p - before.p) / se
    return math.erfc(z / math.sqrt(2))


def delta_is_significant(
    before: ProbEstimate,
    after: ProbEstimate,
    threshold: float,
    alpha: float | None = None,
) -> bool:
    """Upstream's rule, optionally tightened by a significance test.

    With ``alpha=None`` this is exactly ``abs(p_after - p_before) >=
    threshold``, matching upstream. Supplying ``alpha`` additionally requires
    the difference to be statistically distinguishable, which is how the
    high-confidence subset in the paper is defined.
    """
    if abs(after.p - before.p) < threshold:
        return False
    if alpha is None:
        return True
    return two_proportion_p(before, after) < alpha


def false_accept_rate(num_samples: int, threshold: float, p: float = 0.5) -> float:
    """P(|delta_hat| >= threshold) under the null of no true change.

    Quantifies the label noise the paper reports: at S=50, tau=0.2, p=0.5
    this is ~4.6% per tested position.
    """
    if num_samples <= 0:
        return 1.0
    sd = math.sqrt(2 * p * (1 - p) / num_samples)
    if sd == 0:
        return 0.0
    return math.erfc((threshold / sd) / math.sqrt(2))
