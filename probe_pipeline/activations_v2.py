"""Activation cache: labelled positions only, bf16, memory-mappable.

The v1 cache (``data/cached_activations_*/``) stores the residual stream for
**every token of every sequence at every layer, in float32**, and selects the
labelled positions at load time. That is 3.35 GB for 594 labelled rows.

The cause is one line, ``activations.py:144``::

    activation = hidden_states[idx].detach().cpu().float()

On a CPU float32 tensor both ``.cpu()`` and ``.float()`` are no-ops that
return a *view*, so the row still aliases the whole ``(seq_len, d)`` storage
-- and ``torch.save`` serializes storages, not views. Measured: saving three
1024-dim rows out of a (516, 1024) block costs 2,115,241 bytes instead of
12,288. The fix is an explicit copy, and the writer below asserts it.

Storage, all 37 layers, at the scale we are targeting:

===================== ========== ========
 model / rows          v1 format  here
===================== ========== ========
 Qwen3-0.6B, 594          3.35 GB   35 MB
 Qwen3-4B, 12,000 rows    ~300 GB   2.3 GB
===================== ========== ========

Two conventions the manifest records explicitly, so that consumers can
assert rather than assume:

``hidden_state_convention``
    ``outputs.hidden_states[L]``. Index 0 is the embedding output; index L is
    the output of decoder layer ``L-1``. Getting this wrong is `issues.md`
    Issue #2, which cost a whole round of steering runs.
``extraction_position``
    ``t-1``. Rows are the residual stream at the position *before* the token
    being predicted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 2

HIDDEN_STATE_CONVENTION = "outputs.hidden_states[L]"
EXTRACTION_POSITION = "t-1"


@dataclass
class ActivationManifest:
    """Self-describing header for one activation file."""

    model_name: str
    n_layers: int
    hidden_size: int
    split: str
    dtype: str = "bfloat16"
    schema_version: int = SCHEMA_VERSION
    hidden_state_convention: str = HIDDEN_STATE_CONVENTION
    extraction_position: str = EXTRACTION_POSITION
    model_revision: str = ""
    tokenizer_sha: str = ""
    source_dataset: str = ""
    layers_kept: list[int] = field(default_factory=list)
    n_rows: int = 0
    label_map: dict[str, int] = field(
        default_factory=lambda: {"pivotal": 1, "non_pivotal": -1}
    )
    max_abs_activation: float = 0.0
    config_hash: str = ""
    git_sha: str = ""
    created_at: str = ""
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "ActivationManifest":
        data = json.loads(blob)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def _require_owned(arr: "Any") -> None:
    """Guard against the storage-aliasing bug this format exists to fix."""
    storage_elems = arr.untyped_storage().nbytes() // arr.element_size()
    if storage_elems != arr.numel():
        raise ValueError(
            f"tensor aliases a larger storage ({storage_elems} elements backing "
            f"{arr.numel()}); copy it before writing or the file will contain "
            "whole sequences instead of labelled positions"
        )


class ActivationWriter:
    """Collect labelled positions across queries, then write one file.

    All layers share a single row ordering: ``layer_L[i]`` is the same
    ``(query, position)`` for every ``L``. That is what lets labels and
    metadata be stored once rather than per layer, and lets a layer sweep
    memory-map one ``(N, d)`` slice at a time.
    """

    def __init__(
        self,
        path: str | Path,
        manifest: ActivationManifest,
        layers: Sequence[int],
        *,
        dtype: str = "bfloat16",
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.layers = list(layers)
        self.dtype = dtype
        self._rows: dict[int, list[Any]] = {L: [] for L in self.layers}
        self._y: list[int] = []
        self._query_idx: list[int] = []
        self._token_pos: list[int] = []
        self._token_id: list[int] = []
        self._prob_delta: list[float] = []
        self._entropy: list[float] = []
        self._margin: list[float] = []
        self._top1_prob: list[float] = []
        self._max_abs = 0.0
        self._query_ids: list[str] = []

    def add_query(
        self,
        hidden_states: Sequence[Any],
        positions: Sequence[int],
        labels: Sequence[int],
        token_ids: Sequence[int],
        prob_deltas: Sequence[float],
        query_id: str,
        uncertainty: Sequence[tuple[float, float, float]] | None = None,
    ) -> None:
        """Append the labelled positions of one sequence.

        Args:
            hidden_states: one ``(T, d)`` tensor per entry of ``layers``, in
                the same order. Typically ``outputs.hidden_states`` sliced to
                the layers being kept.
            positions: token indices to keep, all ``< T``.
            labels, token_ids, prob_deltas: aligned with ``positions``.
            uncertainty: optional ``(entropy, top1-top2 margin, top1 prob)``
                per position, from the next-token distribution. These are the
                baselines the probe has to beat -- high-entropy "forking"
                tokens are the incumbent notion of a critical token -- so they
                are captured here rather than needing a second pass.
        """
        import torch

        if len(hidden_states) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer tensors, got {len(hidden_states)}"
            )
        if not positions:
            return

        idx = torch.as_tensor(list(positions), dtype=torch.long)
        qi = len(self._query_ids)
        self._query_ids.append(query_id)

        for layer, block in zip(self.layers, hidden_states):
            if block.dim() == 3:  # (1, T, d) -> (T, d)
                block = block.squeeze(0)
            if int(idx.max()) >= block.shape[0]:
                raise IndexError(
                    f"position {int(idx.max())} out of range for sequence of "
                    f"length {block.shape[0]} (query {query_id!r})"
                )
            # index_select copies; .contiguous() + .clone() makes the "this
            # does not alias the full sequence" guarantee explicit.
            picked = block.index_select(0, idx.to(block.device)).detach().cpu()
            picked = picked.to(torch.float32).contiguous().clone()
            _require_owned(picked)
            self._max_abs = max(self._max_abs, float(picked.abs().max()))
            self._rows[layer].append(picked)

        self._y.extend(int(v) for v in labels)
        self._token_pos.extend(int(v) for v in positions)
        self._token_id.extend(int(v) for v in token_ids)
        self._prob_delta.extend(float(v) for v in prob_deltas)
        self._query_idx.extend([qi] * len(positions))

        unc = list(uncertainty) if uncertainty is not None else [(0.0, 0.0, 0.0)] * len(positions)
        if len(unc) != len(positions):
            raise ValueError("uncertainty must align with positions")
        for ent, margin, top1 in unc:
            self._entropy.append(float(ent))
            self._margin.append(float(margin))
            self._top1_prob.append(float(top1))

    @property
    def n_rows(self) -> int:
        return len(self._y)

    def close(self) -> Path:
        import torch
        from safetensors.torch import save_file

        if self.n_rows == 0:
            raise ValueError("nothing to write: no labelled positions were added")

        target = torch.bfloat16 if self.dtype == "bfloat16" else torch.float16
        if self.dtype == "float16" and self._max_abs > 6e4:
            raise ValueError(
                f"max |activation| = {self._max_abs:.1f} overflows float16; "
                "use bfloat16 (Qwen residual streams have massive-activation dims)"
            )

        tensors: dict[str, Any] = {}
        for layer in self.layers:
            tensors[f"layer_{layer}"] = torch.cat(self._rows[layer], dim=0).to(target)
        tensors["y"] = torch.tensor(self._y, dtype=torch.int8)
        tensors["query_idx"] = torch.tensor(self._query_idx, dtype=torch.int32)
        tensors["token_pos"] = torch.tensor(self._token_pos, dtype=torch.int32)
        tensors["token_id"] = torch.tensor(self._token_id, dtype=torch.int32)
        tensors["prob_delta"] = torch.tensor(self._prob_delta, dtype=torch.float32)
        tensors["entropy"] = torch.tensor(self._entropy, dtype=torch.float32)
        tensors["margin"] = torch.tensor(self._margin, dtype=torch.float32)
        tensors["top1_prob"] = torch.tensor(self._top1_prob, dtype=torch.float32)

        self.manifest.n_rows = self.n_rows
        self.manifest.layers_kept = list(self.layers)
        self.manifest.dtype = self.dtype
        self.manifest.max_abs_activation = self._max_abs

        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            str(self.path),
            metadata={
                "manifest": self.manifest.to_json(),
                "query_ids": json.dumps(self._query_ids),
            },
        )
        self.path.with_suffix(".meta.json").write_text(
            self.manifest.to_json(), encoding="utf-8"
        )
        return self.path


class ActivationStoreV2:
    """Read side. Memory-maps the file and materializes one layer at a time."""

    def __init__(self, path: str | Path) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        self._f = safe_open(str(self.path), framework="pt")
        meta = self._f.metadata() or {}
        self.manifest = ActivationManifest.from_json(meta.get("manifest", "{}"))
        self.query_ids: list[str] = json.loads(meta.get("query_ids", "[]"))

    @classmethod
    def open(cls, path: str | Path) -> "ActivationStoreV2":
        return cls(path)

    @property
    def layers(self) -> list[int]:
        return list(self.manifest.layers_kept)

    def _get(self, key: str) -> np.ndarray:
        return self._f.get_tensor(key).to("cpu").float().numpy()

    def layer(self, layer_num: int) -> np.ndarray:
        """``(n_rows, hidden)`` float32 for one layer."""
        if layer_num not in self.manifest.layers_kept:
            raise KeyError(
                f"layer {layer_num} not in this cache; have {self.manifest.layers_kept}"
            )
        return self._get(f"layer_{layer_num}")

    def labels(self) -> np.ndarray:
        return self._f.get_tensor("y").numpy()

    def query_index(self) -> np.ndarray:
        """Row -> index into :attr:`query_ids`. Use for grouped CV / bootstrap."""
        return self._f.get_tensor("query_idx").numpy()

    def prob_delta(self) -> np.ndarray:
        return self._f.get_tensor("prob_delta").numpy()

    def token_ids(self) -> np.ndarray:
        return self._f.get_tensor("token_id").numpy()

    def uncertainty(self) -> dict[str, np.ndarray]:
        """Next-token uncertainty at each labelled position.

        These are the baselines the probe must beat: high-entropy "forking"
        tokens are the incumbent notion of a critical token in the RLVR
        literature, and trajectory-pivot work uses entropy and top-2 margin
        directly as pivot detectors. Returns zeros for caches written before
        these were recorded.
        """
        out = {}
        for key in ("entropy", "margin", "top1_prob"):
            try:
                out[key] = self._f.get_tensor(key).numpy()
            except Exception:
                out[key] = np.zeros(self.manifest.n_rows, dtype=np.float32)
        return out

    def xy(self, layer_num: int) -> tuple[np.ndarray, np.ndarray]:
        """``(X, y)`` with y in {0, 1}, matching ``activations.layer_arrays``.

        The on-disk labels are +1 / -1; probes want 1 / 0.
        """
        x = self.layer(layer_num)
        y = (self.labels() > 0).astype(np.int64)
        return x, y

    def signed_xy(
        self, layer_num: int, *, dead_zone: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(X, y)`` over **pivotal rows only**, y=1 helpful, y=0 harmful.

        This is the signed probe's training set: among positions where
        something pivotal is about to happen, does it help or hurt? Rows with
        ``|prob_delta| <= dead_zone`` are excluded as borderline.
        """
        x = self.layer(layer_num)
        y_all = self.labels()
        delta = self.prob_delta()
        mask = (y_all > 0) & (np.abs(delta) > dead_zone)
        return x[mask], (delta[mask] > 0).astype(np.int64)


def load_activation_store_v2(path: str | Path) -> ActivationStoreV2:
    return ActivationStoreV2.open(path)


def iter_labelled(rows: Iterable[Any]) -> Iterable[tuple[Any, list[int], list[int]]]:
    """Yield ``(row, positions, labels)`` for rows with any labelled position."""
    for row in rows:
        positions = [i for i, v in enumerate(row.labels) if v != 0]
        if positions:
            yield row, positions, [row.labels[i] for i in positions]
