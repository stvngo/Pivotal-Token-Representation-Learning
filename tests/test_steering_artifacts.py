"""Locks in the artifact contract for ``artifacts/cached3/sklearn/steering_configs``.

Asserts that:

* Every layer used by ``notebooks/steering_layer_sweep.ipynb`` (8, 14, 16) ships
  a TRAIN-derived centroid-difference vector (Issue #5 in ``docs/issues.md``).
* The accompanying ``steering_layer{L}_vector_train.json`` declares
  ``split == "train"`` and points at ``train_all_layers_acts.pth``.
* The bundled sklearn LR probe weights file exists, has the expected
  ``(1, hidden_dim)`` shape, and is non-zero (i.e., a real LR fit, not zeros).
* The cosine similarity between the LR probe direction and the TRAIN CAA
  direction is positive (sanity check on the sign of the probe vector).

These assertions guard the Colab path in ``steering_layer_sweep.ipynb`` from
silently regressing back to val-derived CAA or to missing probe weight files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STEER_DIR = REPO_ROOT / "artifacts" / "cached3" / "sklearn" / "steering_configs"
LAYERS_USED_BY_LAYER_SWEEP_NB = (8, 14, 16)
EXPECTED_HIDDEN_DIM = 1024  # Qwen3-0.6B


def _skip_if_artifacts_missing() -> None:
    if not STEER_DIR.exists():
        pytest.skip(f"steering_configs dir missing: {STEER_DIR}")


@pytest.mark.parametrize("layer", LAYERS_USED_BY_LAYER_SWEEP_NB)
def test_train_caa_vector_is_shipped(layer: int) -> None:
    _skip_if_artifacts_missing()
    vec_path = STEER_DIR / f"steering_layer{layer}_vector_train.npy"
    cfg_path = STEER_DIR / f"steering_layer{layer}_vector_train.json"
    assert vec_path.exists(), (
        f"layer {layer}: missing TRAIN-derived CAA vector. "
        f"Re-run the artifact builder to regenerate {vec_path.name}."
    )
    assert cfg_path.exists(), (
        f"layer {layer}: missing companion JSON {cfg_path.name}"
    )

    vec = np.load(vec_path)
    assert vec.dtype == np.float32, f"{vec_path.name} dtype={vec.dtype}, expected float32"
    assert vec.shape == (EXPECTED_HIDDEN_DIM,), (
        f"{vec_path.name} shape={vec.shape}, expected ({EXPECTED_HIDDEN_DIM},)"
    )
    assert np.linalg.norm(vec) > 0.0, f"{vec_path.name} is the zero vector"

    meta = json.loads(cfg_path.read_text())
    assert meta.get("split") == "train", (
        f"{cfg_path.name} declares split={meta.get('split')!r}, "
        "expected 'train' (Issue #5: CAA must come from training activations)"
    )
    assert "train_all_layers_acts" in str(meta.get("source_cache", "")), (
        f"{cfg_path.name} source_cache={meta.get('source_cache')!r}; "
        "TRAIN cache path expected"
    )
    assert int(meta.get("hidden_dim", -1)) == EXPECTED_HIDDEN_DIM
    assert int(meta.get("layer", -1)) == layer


@pytest.mark.parametrize("layer", LAYERS_USED_BY_LAYER_SWEEP_NB)
def test_sklearn_probe_weights_are_shipped(layer: int) -> None:
    _skip_if_artifacts_missing()
    w_path = STEER_DIR / f"steering_layer{layer}_probe_weights.npy"
    b_path = STEER_DIR / f"steering_layer{layer}_probe_biases.npy"
    assert w_path.exists(), (
        f"layer {layer}: missing sklearn probe_weights. "
        f"steering_layer_sweep.ipynb depends on this file."
    )
    assert b_path.exists(), f"layer {layer}: missing {b_path.name}"

    w = np.load(w_path).astype(np.float32)
    b = np.load(b_path).astype(np.float32)
    flat = w.reshape(-1)
    assert flat.shape == (EXPECTED_HIDDEN_DIM,), (
        f"{w_path.name} reshape(-1) gave {flat.shape}, expected ({EXPECTED_HIDDEN_DIM},)"
    )
    norm = float(np.linalg.norm(flat))
    assert norm > 0.0, f"{w_path.name} probe weights have zero norm"
    assert b.shape == (1,), f"{b_path.name} shape={b.shape}, expected (1,)"


@pytest.mark.parametrize("layer", LAYERS_USED_BY_LAYER_SWEEP_NB)
def test_probe_aligns_with_train_caa(layer: int) -> None:
    """LR probe weight vector and TRAIN CAA should point in the same general
    direction (positive cosine). A negative cosine would indicate the probe
    was fit with the labels flipped or the weights got transposed."""
    _skip_if_artifacts_missing()
    w_path = STEER_DIR / f"steering_layer{layer}_probe_weights.npy"
    v_path = STEER_DIR / f"steering_layer{layer}_vector_train.npy"
    if not (w_path.exists() and v_path.exists()):
        pytest.skip(f"layer {layer}: probe or train CAA missing")

    w = np.load(w_path).astype(np.float32).reshape(-1)
    v = np.load(v_path).astype(np.float32).reshape(-1)
    cos = float(np.dot(w, v) / (np.linalg.norm(w) * np.linalg.norm(v) + 1e-12))
    assert cos > 0.0, (
        f"layer {layer}: cos(probe_weights, train_CAA)={cos:+.4f}; "
        "negative cosine suggests label-sign mismatch."
    )
