"""Tests for the artifact cache. No GPU, no network, no model downloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probe_pipeline.artifacts_io import (
    AppendWriter,
    ArtifactSpec,
    ArtifactStore,
    SCHEMA_VERSION,
    config_hash,
    iter_jsonl,
)


@pytest.fixture()
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        local_root=tmp_path / "cache",
        repo_root=tmp_path / "artifacts",
        repo_id=None,
        push=False,
        offline=True,
    )


# -- hashing ---------------------------------------------------------------


def test_hash_is_stable_and_order_independent():
    a = config_hash({"model": "Qwen3-0.6B", "layer": 14, "alpha": 1.5})
    b = config_hash({"alpha": 1.5, "layer": 14, "model": "Qwen3-0.6B"})
    assert a == b
    assert len(a) == 12


def test_hash_changes_with_a_real_parameter():
    assert config_hash({"layer": 14}) != config_hash({"layer": 15})
    # Float precision must matter: 1.5 and 1.50000001 are different runs.
    assert config_hash({"alpha": 1.5}) != config_hash({"alpha": 1.50000001})


def test_volatile_keys_do_not_invalidate():
    """Moving between MPS and A100 must not orphan every cached artifact."""
    on_mac = {"layer": 14, "device": "mps", "log_level": "DEBUG", "num_workers": 0}
    on_a100 = {"layer": 14, "device": "cuda", "log_level": "INFO", "num_workers": 8}
    assert config_hash(on_mac) == config_hash(on_a100)


def test_nested_volatile_keys_are_dropped():
    assert config_hash({"a": {"device": "mps", "k": 1}}) == config_hash(
        {"a": {"device": "cuda", "k": 1}}
    )


def test_arrays_are_hashed_by_content_not_identity():
    v1 = np.arange(8, dtype=np.float32)
    v2 = np.arange(8, dtype=np.float32)
    v3 = np.arange(8, dtype=np.float32)
    v3[0] = 99.0
    assert config_hash({"v": v1}) == config_hash({"v": v2})
    assert config_hash({"v": v1}) != config_hash({"v": v3})


def test_schema_version_participates_in_the_hash():
    import probe_pipeline.artifacts_io as mod

    before = config_hash({"layer": 14})
    mod.SCHEMA_VERSION = SCHEMA_VERSION + 1
    try:
        assert config_hash({"layer": 14}) != before
    finally:
        mod.SCHEMA_VERSION = SCHEMA_VERSION


# -- round trips -----------------------------------------------------------


@pytest.mark.parametrize(
    ("codec", "payload"),
    [
        ("json", {"acc": 0.746, "n": 130}),
        ("jsonl", [{"i": 0}, {"i": 1}, {"i": 2}]),
        ("text", "hello"),
        ("npy", np.arange(12, dtype=np.float32).reshape(3, 4)),
    ],
)
def test_round_trip(store: ArtifactStore, codec, payload):
    spec = ArtifactSpec("probe", {"layer": 14, "codec": codec}, codec=codec)
    store.save(spec, payload)
    loaded = ArtifactStore(
        local_root=store.local_root, repo_root=store.repo_root, push=False, offline=True
    ).load(spec)

    if isinstance(payload, np.ndarray):
        np.testing.assert_array_equal(loaded, payload)
    else:
        assert loaded == payload


def test_save_writes_an_invertible_sidecar(store: ArtifactStore):
    cfg = {"model": "Qwen/Qwen3-0.6B", "layer": 14, "alpha": 1.5}
    spec = ArtifactSpec("gsm8k_eval", cfg, tier="medium")
    store.save(spec, {"acc": 0.6})

    meta = json.loads(store.meta_path(spec).read_text())
    assert meta["config"]["model"] == "Qwen/Qwen3-0.6B"
    assert meta["config"]["layer"] == 14
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["key"] == spec.key


def test_tiers_route_to_different_roots(store: ArtifactStore):
    small = ArtifactSpec("probe", {"a": 1}, tier="small", codec="json")
    medium = ArtifactSpec("acts", {"a": 1}, tier="medium", codec="json")
    assert store.repo_root in store.path(small).parents
    assert store.local_root in store.path(medium).parents


# -- load_or_compute -------------------------------------------------------


def test_load_or_compute_runs_once(store: ArtifactStore):
    spec = ArtifactSpec("gsm8k_eval", {"factor": 1.2}, tier="medium")
    calls = []

    def compute():
        calls.append(1)
        return {"acc": 0.3}

    assert store.load_or_compute(spec, compute) == {"acc": 0.3}
    assert store.load_or_compute(spec, compute) == {"acc": 0.3}
    assert len(calls) == 1, "second call should have hit the cache"


def test_load_or_compute_survives_a_lost_process(store: ArtifactStore):
    """The Colab case: process dies, a fresh store must not recompute."""
    spec = ArtifactSpec("gsm8k_eval", {"factor": 1.2}, tier="medium")
    store.load_or_compute(spec, lambda: {"acc": 0.3})

    fresh = ArtifactStore(
        local_root=store.local_root, repo_root=store.repo_root, push=False, offline=True
    )
    calls = []
    fresh.load_or_compute(spec, lambda: calls.append(1) or {"acc": 0.0})
    assert not calls


def test_force_recomputes(store: ArtifactStore):
    spec = ArtifactSpec("gsm8k_eval", {"factor": 1.2}, tier="medium")
    store.load_or_compute(spec, lambda: {"acc": 0.3})
    out = store.load_or_compute(spec, lambda: {"acc": 0.9}, force=True)
    assert out == {"acc": 0.9}


def test_a_changed_config_is_a_different_artifact(store: ArtifactStore):
    a = ArtifactSpec("gsm8k_eval", {"factor": 1.2}, tier="medium")
    b = ArtifactSpec("gsm8k_eval", {"factor": 1.4}, tier="medium")
    store.load_or_compute(a, lambda: {"acc": 0.3})
    assert store.load_or_compute(b, lambda: {"acc": 0.5}) == {"acc": 0.5}
    assert store.load(a) == {"acc": 0.3}


def test_missing_artifact_raises(store: ArtifactStore):
    with pytest.raises(FileNotFoundError):
        store.load(ArtifactSpec("nope", {"x": 1}))


# -- append / torn writes --------------------------------------------------


def test_append_writer_round_trip(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    with AppendWriter(path) as w:
        w.extend([{"i": i} for i in range(3)])
        w.flush()
    assert [r["i"] for r in iter_jsonl(path)] == [0, 1, 2]


def test_torn_final_line_is_tolerated(tmp_path: Path):
    """A session killed mid-write leaves a partial last line; that is recoverable."""
    path = tmp_path / "probs.jsonl"
    path.write_text('{"i": 0}\n{"i": 1}\n{"i": 2, "part')
    assert [r["i"] for r in iter_jsonl(path)] == [0, 1]


def test_corruption_in_the_middle_raises(tmp_path: Path):
    """Skipping a bad line anywhere but the end would silently drop results."""
    path = tmp_path / "probs.jsonl"
    path.write_text('{"i": 0}\nNOT JSON\n{"i": 2}\n')
    with pytest.raises(json.JSONDecodeError):
        list(iter_jsonl(path))


def test_appends_accumulate_across_reopen(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    with AppendWriter(path) as w:
        w.write({"i": 0})
    with AppendWriter(path) as w:
        w.write({"i": 1})
    assert [r["i"] for r in iter_jsonl(path)] == [0, 1]
