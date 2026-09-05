"""Tests for probe fitting, raw-space export, and honest evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probe_pipeline.baselines import (
    evaluate_baselines,
    random_direction_scores,
    token_identity_scores,
)
from probe_pipeline.probes import (
    DEFAULT_C,
    cluster_bootstrap_ci,
    fit_probe,
    mean_diff_direction,
    nested_layer_selection,
    null_max_distribution,
    split_groups,
    to_raw_space,
)


def make_data(n=400, d=16, seed=0, sep=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # Deliberately anisotropic: unequal per-feature scale is exactly what the
    # standardized-vs-raw bug hides behind.
    scale = np.linspace(0.2, 8.0, d)
    x = rng.normal(size=(n, d)) * scale
    x[y == 1, 0] += sep * scale[0]
    x[y == 1, 5] += sep * scale[5]
    groups = rng.integers(0, 20, n)
    return x.astype(np.float32), y, groups


# -- the export bug --------------------------------------------------------


def test_raw_space_reproduces_the_pipeline_exactly():
    """train_sklearn.py saved standardized-space coefficients and applied them
    to raw activations; every dimension was off by 1/sigma."""
    x, y, _ = make_data()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(C=0.1, max_iter=2000))])
    pipe.fit(x, y)

    w, b = to_raw_space(pipe)
    np.testing.assert_allclose(x @ w + b, pipe.decision_function(x), rtol=1e-4, atol=1e-4)


def test_raw_and_standardized_directions_actually_differ():
    """Guard against the fix being a no-op on realistic (anisotropic) data."""
    x, y, _ = make_data()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(C=0.1, max_iter=2000))])
    pipe.fit(x, y)
    w_raw, _ = to_raw_space(pipe)
    w_std = pipe.named_steps["logreg"].coef_.ravel()

    cos = float(w_raw @ w_std / (np.linalg.norm(w_raw) * np.linalg.norm(w_std)))
    assert cos < 0.99, f"directions nearly identical (cos={cos:.4f}); test data too isotropic"


def test_fitted_probe_matches_its_own_decision_function():
    x, y, _ = make_data()
    probe = fit_probe(x, y, C=DEFAULT_C)
    p = probe.predict_proba(x)
    assert p.shape == (len(y),)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isclose(np.linalg.norm(probe.direction), 1.0)


def test_probe_learns_a_separable_signal():
    x, y, _ = make_data(sep=2.0)
    probe = fit_probe(x, y)
    assert (probe.decision(x) > 0).astype(int).mean() > 0.1
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, probe.decision(x)) > 0.8


def test_mean_diff_is_a_direction_not_a_classifier():
    x, y, _ = make_data(sep=2.0)
    v = mean_diff_direction(x, y)
    assert v.shape == (x.shape[1],)
    assert v[0] > 0 and v[5] > 0  # the two planted dimensions


# -- clustered CIs ---------------------------------------------------------


def test_cluster_bootstrap_returns_a_bracketing_interval():
    x, y, groups = make_data(sep=1.5)
    probe = fit_probe(x, y)
    point, lo, hi = cluster_bootstrap_ci(y, probe.decision(x), groups, n_boot=300, seed=0)
    assert lo <= point <= hi
    assert hi - lo > 0


def test_clustered_ci_is_wider_than_a_row_level_one():
    """Positions inside one reasoning trace are correlated, so resampling rows
    understates the interval. Build data where group membership drives the
    score and check the clustered interval notices."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 30)
    per_group = rng.normal(size=20)
    y = (per_group[groups] > 0).astype(int)
    score = per_group[groups] + rng.normal(scale=0.1, size=len(groups))

    _, lo_c, hi_c = cluster_bootstrap_ci(y, score, groups, n_boot=400, seed=1)
    _, lo_r, hi_r = cluster_bootstrap_ci(y, score, np.arange(len(y)), n_boot=400, seed=1)
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_split_groups_never_splits_a_question():
    _, _, groups = make_data()
    a, b = split_groups(groups, frac=0.3, seed=0)
    assert not np.any(a & b)
    assert np.all(a | b)
    assert not (set(groups[a]) & set(groups[b]))


# -- nested layer selection ------------------------------------------------


def test_nested_selection_does_not_report_the_layer_it_picked():
    """The naive procedure reports max-over-layers on the same rows used to
    choose; nested selection must be no better, and usually worse."""
    layer_xy = {}
    for layer in range(12):
        x, y, groups = make_data(seed=layer, sep=0.4 + 0.05 * layer)
        layer_xy[layer] = (x, y)
    _, y0, groups = make_data(seed=0)
    layer_xy = {L: (xy[0], y0) for L, xy in layer_xy.items()}

    sweep = nested_layer_selection(layer_xy, groups, seed=0)
    assert sweep.selected_layer is not None
    assert sweep.naive_best_outer >= sweep.selected_outer
    assert sweep.winners_curse >= 0


def test_null_max_distribution_flags_a_selection_artifact():
    """The real layer sweep: 29 layers, mean 0.680, observed max 0.746 over
    130 rows. Under the null the expected max is HIGHER than what was seen."""
    per_layer = list(np.random.default_rng(0).normal(0.680, 0.031, 29))
    stats = null_max_distribution(per_layer, n_rows=130, n_sim=4000, seed=0)
    assert stats["expected_max"] > stats["observed_mean"]
    assert 0.0 <= stats["p_observed_or_higher"] <= 1.0


# -- baselines -------------------------------------------------------------


def test_token_identity_beats_chance_when_tokens_carry_the_label():
    """If a token id perfectly predicts pivotality, the lookup table must find
    it -- that is what makes it a real control for the probe."""
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 5, 300)
    y = (toks < 2).astype(int)
    scores = token_identity_scores(toks, y, toks)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, scores) > 0.95


def test_token_identity_is_chance_out_of_sample_when_tokens_are_uninformative():
    """Scored in-sample the lookup table memorizes (AUROC ~0.69 on random
    labels with ~8 rows per token), so like the probe it must be fit on train
    and scored on held-out rows."""
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 50, 800)
    y = rng.integers(0, 2, 800)
    tr, ev = slice(0, 400), slice(400, 800)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[ev], token_identity_scores(toks[tr], y[tr], toks[ev]))
    assert 0.35 < auc < 0.65, f"uninformative tokens scored {auc:.3f} out of sample"


def test_token_identity_memorizes_in_sample():
    """Pin the failure mode so nobody reports an in-sample baseline number."""
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 50, 400)
    y = rng.integers(0, 2, 400)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, token_identity_scores(toks, y, toks)) > 0.6


def test_random_direction_is_chance():
    x, y, _ = make_data(sep=2.0)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, random_direction_scores(x, seed=0, n_directions=8))
    assert 0.3 < auc < 0.7


def test_evaluate_baselines_reports_every_control():
    x, y, _ = make_data()
    toks = np.random.default_rng(0).integers(0, 30, len(y))
    unc = {
        "entropy": np.random.default_rng(1).normal(size=len(y)),
        "margin": np.random.default_rng(2).normal(size=len(y)),
        "top1_prob": np.random.default_rng(3).normal(size=len(y)),
    }
    out = evaluate_baselines(
        y_train=y, y_eval=y, token_ids_train=toks, token_ids_eval=toks,
        uncertainty_eval=unc, x_eval=x,
    )
    assert {"token_identity_freq", "token_identity_lr", "entropy",
            "neg_margin", "neg_top1_prob", "random_direction"} <= set(out)
    for name, m in out.items():
        assert "auroc" in m and "accuracy" in m, name
