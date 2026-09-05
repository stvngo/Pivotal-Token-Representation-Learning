"""Baselines the pivotality probe has to beat.

Two of these are load-bearing rather than decorative:

**Token identity.** On the v1 cache, layer 0 -- raw token embeddings, before
any transformer computation -- reached 65.4% against layer 14's 74.6%. So
roughly three quarters of the apparent signal may be "which token is this",
not a computed representation of pivotality. If the probe cannot beat a
lookup table over token ids, the mechanistic claim collapses.

**Next-token uncertainty.** High-entropy "forking" tokens are the incumbent
notion of a critical token (Wang et al., NeurIPS 2025), and trajectory-pivot
work uses entropy and top-2 margin directly as detectors. A pivotality probe
that merely rediscovers entropy is not a new finding.

Note the asymmetry that makes the *signed* probe the stronger claim: entropy
is sign-blind by construction, so it cannot in principle explain a probe that
separates helpful from harmful pivots.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .probes import binary_metrics


def token_identity_scores(
    token_ids_train: np.ndarray,
    y_train: np.ndarray,
    token_ids_eval: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Laplace-smoothed log-odds of pivotality given the token id alone.

    A frequency table rather than a fitted model, so it cannot exploit
    capacity the probe does not have. Unseen tokens fall back to the prior.
    """
    prior = float(y_train.mean())
    pos: dict[int, int] = {}
    tot: dict[int, int] = {}
    for t, label in zip(token_ids_train.tolist(), y_train.tolist()):
        tot[t] = tot.get(t, 0) + 1
        pos[t] = pos.get(t, 0) + int(label)

    scores = np.empty(len(token_ids_eval), dtype=np.float64)
    for i, t in enumerate(token_ids_eval.tolist()):
        n = tot.get(t, 0)
        k = pos.get(t, 0)
        p = (k + alpha * prior) / (n + alpha) if n else prior
        p = min(max(p, 1e-6), 1 - 1e-6)
        scores[i] = np.log(p / (1 - p))
    return scores


def token_onehot_probe_scores(
    token_ids_train: np.ndarray,
    y_train: np.ndarray,
    token_ids_eval: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Logistic regression on one-hot token identity.

    The stronger form of the token-identity control: it can weight tokens
    freely rather than trusting raw empirical rates.
    """
    vocab = {t: i for i, t in enumerate(sorted(set(token_ids_train.tolist())))}
    if not vocab:
        return np.zeros(len(token_ids_eval))

    def encode(ids: np.ndarray) -> np.ndarray:
        m = np.zeros((len(ids), len(vocab) + 1), dtype=np.float32)
        for i, t in enumerate(ids.tolist()):
            m[i, vocab.get(t, len(vocab))] = 1.0
        return m

    if len(np.unique(y_train)) < 2:
        return np.zeros(len(token_ids_eval))
    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    lr.fit(encode(token_ids_train), y_train)
    return lr.decision_function(encode(token_ids_eval))


def random_direction_scores(
    x_eval: np.ndarray, *, seed: int = 0, n_directions: int = 1
) -> np.ndarray:
    """Projection onto a random unit direction. The null for 'any direction'."""
    rng = np.random.default_rng(seed)
    d = x_eval.shape[1]
    acc = np.zeros(len(x_eval))
    for _ in range(n_directions):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        acc += x_eval @ v
    return acc / n_directions


def evaluate_baselines(
    *,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    token_ids_train: np.ndarray,
    token_ids_eval: np.ndarray,
    uncertainty_eval: dict[str, np.ndarray],
    x_eval: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Score every baseline on the eval split.

    Uncertainty signals are used directly as detectors, in the direction the
    literature uses them: higher entropy / lower margin / lower top-1
    probability all mean "more likely to be a critical token".
    """
    out: dict[str, dict[str, float]] = {}

    out["token_identity_freq"] = binary_metrics(
        y_eval, token_identity_scores(token_ids_train, y_train, token_ids_eval)
    )
    out["token_identity_lr"] = binary_metrics(
        y_eval, token_onehot_probe_scores(token_ids_train, y_train, token_ids_eval, seed=seed)
    )

    ent = uncertainty_eval.get("entropy")
    if ent is not None and np.any(ent != 0):
        out["entropy"] = binary_metrics(y_eval, ent)
        out["neg_margin"] = binary_metrics(y_eval, -uncertainty_eval["margin"])
        out["neg_top1_prob"] = binary_metrics(y_eval, -uncertainty_eval["top1_prob"])

    if x_eval is not None:
        out["random_direction"] = binary_metrics(
            y_eval, random_direction_scores(x_eval, seed=seed, n_directions=8)
        )

    return out
