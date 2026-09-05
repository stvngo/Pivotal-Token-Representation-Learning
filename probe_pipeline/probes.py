"""Probe fitting and honest evaluation.

Three things here that the original pipeline got wrong, each of which changed
a headline number:

**Weights are exported in raw activation space.** ``train_sklearn.py`` fits a
``StandardScaler -> LogisticRegression`` pipeline and then saves
``logreg.coef_`` -- coefficients that live in *standardized* space -- while
saving unscaled activations alongside. Steering, the reactive gate, and every
cosine-to-CAA number then applied those coefficients to raw residual streams.
:func:`to_raw_space` folds the scaler back in, which is the only form that can
be used as a direction.

**Layer choice is nested.** ``evaluate.py`` reported the best layer's accuracy
on the same rows used to pick it. Across 29 layers with ~130 validation rows,
the maximum of correlated noise is not a finding: under a null where every
layer is equally good, the *expected* max exceeded the observed one. Layers are
selected on an inner split here and reported on an outer one.

**Confidence intervals are clustered by question.** Positions drawn from the
same reasoning trace are not independent, so a bootstrap over rows understates
the interval. :func:`cluster_bootstrap_ci` resamples questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ARENA's Geometry-of-Truth setting. The repo default was C=1.0, which the
# existing NIE runs suggest yields a causally inert direction (NIE ~ +-0.008
# versus +-0.11 at C=0.001).
DEFAULT_C = 0.1


@dataclass
class FittedProbe:
    """A linear probe, with weights usable directly on raw activations."""

    w: np.ndarray            # (d,) raw-space
    b: float                 # raw-space intercept
    C: float
    layer: int | None = None
    n_train: int = 0
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None

    def decision(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(x)))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector, for use as a steering direction."""
        n = np.linalg.norm(self.w)
        return self.w / n if n > 0 else self.w


def to_raw_space(pipe: Pipeline) -> tuple[np.ndarray, float]:
    """Convert a StandardScaler+LogisticRegression fit into raw-space (w, b).

    ``logit = w_s . (x - mu)/sigma + b_s = (w_s/sigma) . x + (b_s - sum(w_s*mu/sigma))``
    """
    scaler: StandardScaler = pipe.named_steps["scaler"]
    lr: LogisticRegression = pipe.named_steps["logreg"]
    w_s = lr.coef_.ravel()
    b_s = float(lr.intercept_.ravel()[0])
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    w_raw = w_s / scale
    b_raw = b_s - float(np.sum(w_s * mean / scale))
    return w_raw.astype(np.float32), b_raw


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    C: float = DEFAULT_C,
    layer: int | None = None,
    max_iter: int = 3000,
    seed: int = 42,
) -> FittedProbe:
    """Fit a standardized logistic probe and return it in raw space."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=C, max_iter=max_iter, solver="saga", random_state=seed
                ),
            ),
        ]
    )
    pipe.fit(x, y)
    w, b = to_raw_space(pipe)
    scaler = pipe.named_steps["scaler"]
    return FittedProbe(
        w=w,
        b=b,
        C=C,
        layer=layer,
        n_train=len(y),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
    )


def mean_diff_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``mu_pos - mu_neg`` (CAA / mass-mean). Computed on TRAIN data."""
    return (x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)).astype(np.float32)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    """Metrics from a decision score (not a probability)."""
    pred = (score > threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, score))
        out["ap"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out


def cluster_bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile CI, resampling **questions** rather than rows.

    Positions from one reasoning trace are correlated, so a row-level
    bootstrap reports an interval that is too narrow.

    Returns ``(point, lo, hi)``.
    """
    if metric is None:
        metric = lambda yy, ss: float(roc_auc_score(yy, ss))  # noqa: E731

    point = metric(y, score)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    stats: list[float] = []

    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        yy, ss = y[idx], score[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            stats.append(metric(yy, ss))
        except ValueError:
            continue

    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# layer selection
# --------------------------------------------------------------------------


@dataclass
class LayerSweep:
    """Per-layer scores plus the nested-selection outcome."""

    inner: dict[int, float] = field(default_factory=dict)
    outer: dict[int, float] = field(default_factory=dict)
    selected_layer: int | None = None
    selected_outer: float = float("nan")
    naive_best_layer: int | None = None
    naive_best_outer: float = float("nan")

    @property
    def winners_curse(self) -> float:
        """How much the naive 'best layer on the report set' overstates."""
        return float(self.naive_best_outer - self.selected_outer)


def split_groups(
    groups: np.ndarray, *, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks splitting rows by whole question."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    n = max(1, int(round(len(shuffled) * frac)))
    held = set(shuffled[:n].tolist())
    mask_b = np.array([g in held for g in groups])
    return ~mask_b, mask_b


def nested_layer_selection(
    layer_xy: dict[int, tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    *,
    C: float = DEFAULT_C,
    inner_frac: float = 0.5,
    seed: int = 0,
    metric: str = "auroc",
) -> LayerSweep:
    """Choose the layer on an inner split, score it on a disjoint outer split.

    ``layer_xy`` maps layer -> (X, y) over the *same* rows in the same order,
    which is the invariant the v2 activation cache guarantees.
    """
    fit_mask, eval_mask = split_groups(groups, frac=0.5, seed=seed)
    inner_groups = groups[eval_mask]
    inner_a, inner_b = split_groups(inner_groups, frac=inner_frac, seed=seed + 1)

    sweep = LayerSweep()
    for layer, (x, y) in sorted(layer_xy.items()):
        probe = fit_probe(x[fit_mask], y[fit_mask], C=C, layer=layer, seed=seed)
        s_eval = probe.decision(x[eval_mask])
        y_eval = y[eval_mask]
        for mask, target in ((inner_a, sweep.inner), (inner_b, sweep.outer)):
            yy, ss = y_eval[mask], s_eval[mask]
            target[layer] = (
                float(roc_auc_score(yy, ss))
                if metric == "auroc" and len(np.unique(yy)) > 1
                else float(accuracy_score(yy, (ss > 0).astype(int)))
            )

    if sweep.inner:
        sweep.selected_layer = max(sweep.inner, key=lambda k: sweep.inner[k])
        sweep.selected_outer = sweep.outer[sweep.selected_layer]
        sweep.naive_best_layer = max(sweep.outer, key=lambda k: sweep.outer[k])
        sweep.naive_best_outer = sweep.outer[sweep.naive_best_layer]
    return sweep


def null_max_distribution(
    per_layer: Sequence[float], n_rows: int, *, n_sim: int = 20000, seed: int = 0
) -> dict[str, float]:
    """What a layer sweep looks like when no layer is actually special.

    Simulates ``len(per_layer)`` layers all sharing the observed mean accuracy,
    each measured on ``n_rows`` rows, and reports the distribution of the
    maximum. If the observed max sits inside this, "layer L is best" is a
    selection artifact.
    """
    vals = np.asarray(per_layer, dtype=float)
    p = float(vals.mean())
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n_rows, p, size=(n_sim, len(vals))) / n_rows
    maxima = draws.max(axis=1)
    return {
        "observed_max": float(vals.max()),
        "observed_mean": p,
        "observed_spread": float(vals.max() - vals.min()),
        "expected_max": float(maxima.mean()),
        "max_p95": float(np.percentile(maxima, 95)),
        "p_observed_or_higher": float(np.mean(maxima >= vals.max())),
    }
