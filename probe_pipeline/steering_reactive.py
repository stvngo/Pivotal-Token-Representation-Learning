"""Token-conditional / reactive activation steering.

Closes Issue #4 from docs/issues.md ("Always-on additive steering at every
position"): instead of perturbing every position of every prompt, we run a
linear probe at each generation step on the *new* token's residual stream and
only apply the steering vector when the probe fires.

This module is **detection-only**: it gates on a binary ``is_pivotal`` probe.
A signed (positive/negative pivot) probe is in a separate plan and outside the
scope of this implementation; once available, ``ReactiveSteeringHook`` can be
extended trivially by adding a second probe and inverting the steering sign on
negative-pivotal predictions.

Convention notes (see docs/issues.md §1, §2):

* The probe was trained on ``outputs.hidden_states[L]``, which equals the
  *input* to ``model.model.layers[L-1]`` for ``L >= 1``.
* We therefore install a **pre**-hook on ``model.model.layers[L-1]`` so we see
  the same residual the probe was trained on, decide if the gate fires, and
  modify the input tensor before the layer runs (the steering perturbation
  appears in the residual that ``layers[L-1]`` actually consumes).
* ``hysteresis`` keeps the gate latched for ``N`` additional steps after a
  positive detection so the steering doesn't oscillate token-by-token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .steering import _get_decoder_layer, make_hook


@dataclass
class ReactiveSteeringStats:
    """Mutable counters tracked by ``ReactiveSteeringHook`` over a run."""

    n_calls: int = 0
    n_fired: int = 0
    n_hysteresis_holds: int = 0
    fire_log: list[dict[str, Any]] = field(default_factory=list)
    energy: float = 0.0  # sum of ||delta|| over fired positions

    def fire_rate(self) -> float:
        return float(self.n_fired) / max(1, self.n_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calls": int(self.n_calls),
            "n_fired": int(self.n_fired),
            "n_hysteresis_holds": int(self.n_hysteresis_holds),
            "fire_rate": float(self.fire_rate()),
            "energy": float(self.energy),
        }


class ReactiveSteeringHook:
    """Pre-hook that gates ``make_hook`` by a binary probe at the same layer.

    Args:
        model: HF causal LM.
        layer: probe layer (the residual we read AND the residual we modify;
            both end up being ``outputs.hidden_states[layer]`` thanks to the
            pre-hook on ``layers[layer-1]``).
        probe_w: shape ``(hidden_dim,)`` -- LR weight vector of the
            ``is_pivotal`` probe.
        probe_b: scalar -- LR bias.
        vector: ``(hidden_dim,)`` steering direction. Same conventions as
            ``make_hook``: ``additive_normalized`` and ``projection`` use it as
            ``v_hat = v / ||v||``; ``additive_raw`` uses it directly.
        coef: scalar steering coefficient (interpretation depends on ``mode``).
        mode: one of ``additive_raw``, ``additive_normalized``, ``projection``.
            Default ``additive_normalized`` so ``coef`` is layer-comparable.
        threshold: probability threshold above which the gate fires
            (default 0.5).
        hysteresis: number of *additional* tokens after a fire to keep the gate
            on (default 2). 0 means single-token-only firing.
        always_on_during_prefill: when True, fall back to always-on for the
            prefill pass (multi-token forward) so the prompt itself can be
            steered if desired. When False (default), prefill is left alone and
            only generated-token positions are eligible for steering.
        force_fire_pattern: optional ``np.ndarray`` of bool, indexed by the
            order in which gate decisions are made. When supplied, overrides
            the probe decision -- used to build matched-fire-rate random
            controls in ``steering_reactive.ipynb``. ``None`` (default) uses
            the probe.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int,
        probe_w: np.ndarray | torch.Tensor,
        probe_b: float | np.ndarray | torch.Tensor,
        vector: np.ndarray | torch.Tensor,
        coef: float,
        mode: str = "additive_normalized",
        threshold: float = 0.5,
        hysteresis: int = 2,
        always_on_during_prefill: bool = False,
        force_fire_pattern: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.coef = float(coef)
        self.mode = mode
        self.threshold = float(threshold)
        self.hysteresis = int(hysteresis)
        self.always_on_during_prefill = bool(always_on_during_prefill)
        self.force_fire_pattern = (
            np.asarray(force_fire_pattern, dtype=bool)
            if force_fire_pattern is not None
            else None
        )
        self._force_idx = 0

        self.probe_w = self._to_tensor(probe_w).reshape(-1)
        b_t = self._to_tensor(probe_b).reshape(-1)
        self.probe_b = float(b_t.item()) if b_t.numel() == 1 else float(b_t[0].item())
        self.vector = self._to_tensor(vector).reshape(-1)

        self._handle: Any | None = None
        self._latch_remaining: int = 0
        self.stats = ReactiveSteeringStats()

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(torch.float32)
        return torch.tensor(np.asarray(x, dtype=np.float32))

    def _decide(self, h_last: torch.Tensor) -> tuple[bool, float]:
        """Return (fire, prob_pivotal) for a single position residual.

        ``h_last`` shape: ``(hidden_dim,)``.
        """
        w = self.probe_w.to(device=h_last.device, dtype=h_last.dtype)
        logit = torch.dot(h_last.float(), w.float()).item() + self.probe_b
        prob = float(1.0 / (1.0 + np.exp(-logit)))

        if self.force_fire_pattern is not None:
            i = self._force_idx
            self._force_idx += 1
            if i < len(self.force_fire_pattern):
                return bool(self.force_fire_pattern[i]), prob
            return False, prob
        return prob > self.threshold, prob

    def _pre_hook(self, _module: nn.Module, inputs: tuple) -> tuple:
        if not inputs:
            return inputs
        hidden = inputs[0]
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return inputs

        seq_len = hidden.shape[1]
        is_prefill = seq_len > 1

        if is_prefill:
            if not self.always_on_during_prefill:
                self.stats.n_calls += 1
                return inputs
            # Always-on prefill path: apply the perturbation to every prompt
            # token. We don't run the probe per-prompt-token here; that would
            # duplicate the always-on baseline.
            mask = torch.ones(seq_len, dtype=torch.bool, device=hidden.device)
        else:
            # Decode step: single new token at index 0.
            self.stats.n_calls += 1
            h_last = hidden[0, -1, :]
            fired_now, prob = self._decide(h_last)

            if fired_now:
                self._latch_remaining = self.hysteresis
                self.stats.n_fired += 1
            elif self._latch_remaining > 0:
                self._latch_remaining -= 1
                self.stats.n_hysteresis_holds += 1
                fired_now = True

            self.stats.fire_log.append({
                "step": int(self.stats.n_calls),
                "prob": float(prob),
                "fired": bool(fired_now),
                "latch_remaining": int(self._latch_remaining),
            })

            if not fired_now:
                return inputs

            mask = torch.tensor([True], dtype=torch.bool, device=hidden.device)

        hook = make_hook(self.vector, self.coef, mode=self.mode, position_mask=mask)
        new_hidden = hook(_module, inputs, hidden)
        if isinstance(new_hidden, tuple):
            new_hidden = new_hidden[0]
        delta = new_hidden - hidden
        self.stats.energy += float(delta.float().norm().item())
        new_inputs = (new_hidden,) + tuple(inputs[1:])
        return new_inputs

    def __enter__(self) -> "ReactiveSteeringHook":
        target = _get_decoder_layer(self.model, self.layer)
        self._handle = target.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self) -> None:
        """Clear stats and latch between runs (e.g. between examples)."""
        self.stats = ReactiveSteeringStats()
        self._latch_remaining = 0
        self._force_idx = 0


def build_random_fire_pattern(
    n_steps: int,
    fire_rate: float,
    seed: int = 101,
) -> np.ndarray:
    """Bernoulli mask of length ``n_steps`` with the requested expected
    fire rate. Used by ``steering_reactive.ipynb`` for the matched-fire-rate
    random control arm.
    """
    rng = np.random.default_rng(seed)
    return rng.random(size=n_steps) < float(fire_rate)
