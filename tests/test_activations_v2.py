"""Tests for the v2 activation cache. No model, no GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from probe_pipeline.activations_v2 import (
    EXTRACTION_POSITION,
    HIDDEN_STATE_CONVENTION,
    ActivationManifest,
    ActivationStoreV2,
    ActivationWriter,
    _require_owned,
)

LAYERS = [0, 1, 2]
HIDDEN = 8


def manifest() -> ActivationManifest:
    return ActivationManifest(
        model_name="stub/tiny", n_layers=3, hidden_size=HIDDEN, split="train"
    )


def fake_hidden(seq_len: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(seq_len, HIDDEN, generator=g) for _ in LAYERS]


def write_small(path: Path, dtype: str = "bfloat16") -> Path:
    w = ActivationWriter(path, manifest(), LAYERS, dtype=dtype)
    w.add_query(fake_hidden(10, 1), [3, 7], [1, -1], [101, 102], [0.5, 0.0], "q1")
    w.add_query(fake_hidden(12, 2), [4], [1], [103], [-0.4], "q2")
    return w.close()


# -- the bug this format exists to fix -------------------------------------


def test_aliasing_guard_rejects_a_view():
    """`hidden[i].detach().cpu().float()` on CPU float32 returns a view that
    still backs the whole (T, d) storage, which is why the v1 cache is 3.35 GB
    for 594 rows."""
    block = torch.randn(516, HIDDEN)
    aliased = block[3].detach().cpu().float()
    with pytest.raises(ValueError, match="aliases a larger storage"):
        _require_owned(aliased)

    _require_owned(aliased.clone())  # the fix


def test_written_rows_do_not_drag_whole_sequences(tmp_path: Path):
    """The regression test proper: file size must scale with labelled rows,
    not with sequence length."""
    short = ActivationWriter(tmp_path / "short.safetensors", manifest(), LAYERS)
    short.add_query(fake_hidden(16, 1), [3], [1], [1], [0.5], "q")
    size_short = short.close().stat().st_size

    long = ActivationWriter(tmp_path / "long.safetensors", manifest(), LAYERS)
    long.add_query(fake_hidden(1024, 1), [3], [1], [1], [0.5], "q")
    size_long = long.close().stat().st_size

    assert abs(size_long - size_short) < 512, (
        f"a 64x longer sequence changed file size ({size_short} -> {size_long}); "
        "rows are aliasing their sequence again"
    )


# -- round trip ------------------------------------------------------------


def test_round_trip(tmp_path: Path):
    path = write_small(tmp_path / "acts.safetensors")
    store = ActivationStoreV2.open(path)

    assert store.manifest.n_rows == 3
    assert store.layers == LAYERS
    assert store.query_ids == ["q1", "q2"]
    np.testing.assert_array_equal(store.labels(), np.array([1, -1, 1], dtype=np.int8))
    np.testing.assert_array_equal(store.query_index(), np.array([0, 0, 1]))
    np.testing.assert_array_equal(store.token_ids(), np.array([101, 102, 103]))

    for layer in LAYERS:
        assert store.layer(layer).shape == (3, HIDDEN)


def test_values_match_the_source_positions(tmp_path: Path):
    hidden = fake_hidden(10, 1)
    w = ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS)
    w.add_query(hidden, [3, 7], [1, -1], [1, 2], [0.5, 0.0], "q1")
    path = w.close()

    store = ActivationStoreV2.open(path)
    got = store.layer(1)
    want = hidden[1][[3, 7]].numpy()
    # bfloat16 keeps ~3 decimal digits.
    np.testing.assert_allclose(got, want, atol=2e-2)


def test_all_layers_share_one_row_ordering(tmp_path: Path):
    """layer_L[i] must be the same (query, position) for every L."""
    path = write_small(tmp_path / "a.safetensors")
    store = ActivationStoreV2.open(path)
    n = store.manifest.n_rows
    for layer in store.layers:
        assert store.layer(layer).shape[0] == n
    assert len(store.labels()) == n
    assert len(store.query_index()) == n


def test_manifest_records_the_conventions(tmp_path: Path):
    """A steering run must be able to assert the convention, not trust a comment."""
    path = write_small(tmp_path / "a.safetensors")
    store = ActivationStoreV2.open(path)
    assert store.manifest.hidden_state_convention == HIDDEN_STATE_CONVENTION
    assert store.manifest.extraction_position == EXTRACTION_POSITION
    assert store.manifest.model_name == "stub/tiny"
    assert (tmp_path / "a.meta.json").exists()


# -- probe views -----------------------------------------------------------


def test_xy_maps_labels_to_binary(tmp_path: Path):
    path = write_small(tmp_path / "a.safetensors")
    x, y = ActivationStoreV2.open(path).xy(1)
    assert x.shape == (3, HIDDEN)
    np.testing.assert_array_equal(y, np.array([1, 0, 1]))


def test_signed_xy_uses_pivotal_rows_only(tmp_path: Path):
    """The signed probe asks: given something pivotal is coming, help or hurt?"""
    path = write_small(tmp_path / "a.safetensors")
    x, y = ActivationStoreV2.open(path).signed_xy(1)
    assert x.shape[0] == 2, "only the two pivotal rows"
    np.testing.assert_array_equal(y, np.array([1, 0]))  # +0.5 helps, -0.4 hurts


def test_signed_dead_zone_drops_borderline_rows(tmp_path: Path):
    w = ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS)
    w.add_query(fake_hidden(10, 1), [1, 2, 3], [1, 1, 1], [1, 2, 3], [0.6, 0.05, -0.7], "q")
    path = w.close()
    x, y = ActivationStoreV2.open(path).signed_xy(1, dead_zone=0.1)
    assert x.shape[0] == 2
    np.testing.assert_array_equal(y, np.array([1, 0]))


# -- guards ----------------------------------------------------------------


def test_float16_overflow_is_refused(tmp_path: Path):
    """Qwen residual streams have massive-activation dims; fp16 caps at 65504."""
    w = ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS, dtype="float16")
    hidden = fake_hidden(4, 1)
    hidden[0][1, 0] = 1e5
    w.add_query(hidden, [1], [1], [1], [0.5], "q")
    with pytest.raises(ValueError, match="overflows float16"):
        w.close()


def test_position_out_of_range_is_caught(tmp_path: Path):
    w = ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS)
    with pytest.raises(IndexError):
        w.add_query(fake_hidden(4, 1), [99], [1], [1], [0.5], "q")


def test_layer_count_mismatch_is_caught(tmp_path: Path):
    w = ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS)
    with pytest.raises(ValueError, match="expected 3 layer tensors"):
        w.add_query(fake_hidden(4, 1)[:2], [1], [1], [1], [0.5], "q")


def test_unknown_layer_is_rejected(tmp_path: Path):
    path = write_small(tmp_path / "a.safetensors")
    with pytest.raises(KeyError):
        ActivationStoreV2.open(path).layer(99)


def test_empty_write_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="nothing to write"):
        ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS).close()


def test_batched_shape_is_accepted(tmp_path: Path):
    """(1, T, d) straight from a forward pass, not just (T, d)."""
    w = ActivationWriter(tmp_path / "a.safetensors", manifest(), LAYERS)
    w.add_query([h.unsqueeze(0) for h in fake_hidden(6, 1)], [2], [1], [1], [0.3], "q")
    store = ActivationStoreV2.open(w.close())
    assert store.layer(0).shape == (1, HIDDEN)
