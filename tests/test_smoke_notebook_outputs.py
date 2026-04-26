"""Validate the on-disk shape of artifacts/smoke/<variant>/*_state.json.

Asserts:

* Every state file carries ``metrics.parse_rate`` (Issue #10's "track parse
  rate consistently" requirement -- the answer-extractor was emitting None
  silently before; now we surface it as a metric and lock that in here).
* Every state file still carries the legacy top-level keys ``label`` and
  ``metrics``, so downstream notebooks (``steering_probe_weights.ipynb``,
  ``codelion_steering_vectors.ipynb`` overlays, etc.) keep deserializing as
  before.
* Every steered run (anything with ``mode != "none"``) carries the new
  provenance fields: ``hook_target_layer_idx``, ``mode``, ``vector_source``,
  ``convention_check_passed``.

The test is skipped per-variant when its directory hasn't been populated yet.
The expected workflow is:

    python scripts/smoke_colab_notebook.py --variant all --max-examples 2
    pytest tests/test_smoke_notebook_outputs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "artifacts" / "smoke"
VARIANTS = (
    "main", "random_control", "greedy", "layer_sweep", "additive",
    "codelion", "probe_weights", "reactive", "nie",
)
LEGACY_TOP_LEVEL_KEYS = ("label", "metrics")
PROVENANCE_KEYS = (
    "hook_target_layer_idx",
    "mode",
    "vector_source",
    "convention_check_passed",
)


def _state_files(variant: str) -> list[Path]:
    d = SMOKE_ROOT / variant
    if not d.exists():
        return []
    return sorted(d.glob("*_state.json"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_state_files_have_legacy_keys_and_parse_rate(variant: str) -> None:
    files = _state_files(variant)
    if not files:
        pytest.skip(
            f"no smoke output for {variant}; "
            f"run `python scripts/smoke_colab_notebook.py --variant {variant} --max-examples 2` first"
        )

    for path in files:
        d = json.loads(path.read_text())
        for key in LEGACY_TOP_LEVEL_KEYS:
            assert key in d, f"{path} missing legacy top-level key {key!r}"
        assert isinstance(d["metrics"], dict), f"{path} metrics not a dict"
        assert "parse_rate" in d["metrics"], f"{path} missing metrics.parse_rate"
        pr = d["metrics"]["parse_rate"]
        assert 0.0 <= float(pr) <= 1.0, f"{path} parse_rate out of range: {pr}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_steered_runs_have_provenance_fields(variant: str) -> None:
    files = _state_files(variant)
    if not files:
        pytest.skip(f"no smoke output for {variant}")

    for path in files:
        d = json.loads(path.read_text())
        if d.get("mode") in (None, "none"):
            # Base / no-op runs do not carry steering provenance fields.
            continue
        for key in PROVENANCE_KEYS:
            assert key in d, f"{path} missing provenance key {key!r}"
        # Sanity: hook target index is layer-1 for layer>=1, -1 for layer 0.
        layer = d.get("layer")
        if isinstance(layer, int) and layer >= 1:
            expected = layer - 1
            actual = d["hook_target_layer_idx"]
            assert actual == expected, (
                f"{path} hook_target_layer_idx={actual} but layer={layer} "
                f"=> expected layer-1 = {expected} (post-Issue-#2)"
            )


def test_at_least_one_variant_has_been_smoked() -> None:
    """Soft global check so a `pytest tests/` invocation surfaces the
    "you forgot to run the smoke script" case once instead of skipping all
    parametrized cases silently."""
    if not SMOKE_ROOT.exists():
        pytest.skip(
            "artifacts/smoke does not exist; run "
            "`python scripts/smoke_colab_notebook.py --variant all --max-examples 2` first"
        )
    found = sum(1 for v in VARIANTS if _state_files(v))
    assert found >= 1, (
        "no smoke variant has produced state files yet; "
        "run scripts/smoke_colab_notebook.py first"
    )
