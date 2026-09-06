"""Probe-gated (reactive) activation steering, batched.

This replaces the earlier `ReactiveSteeringHook`, which was wrong in four
ways that between them made every number it produced meaningless:

* it installed a **pre**-hook on ``layers[L-1]``, which sees
  ``hidden_states[L-1]`` -- one layer below what the probe was trained on.
  The notebook that "validated the convention" used a *post*-hook, so it
  checked a code path that never ran;
* the gate computed ``h_raw . w_std + b_std`` with standardized-space
  weights, dropping the scaler's mean-centering entirely and using a bias
  off by two orders of magnitude (-0.0122 against a true -1.354);
* ``fire_rate`` counted only fresh detections and not hysteresis holds, so
  a reported duty cycle of 0.570 was really 0.878 -- the "reactive" arm was
  near-always-on, and its matched control got 39% less energy;
* ``reset()`` was never called, so the latch leaked across examples.

Three design decisions here, each of which removes a class of bug rather
than patching an instance of it.

**A post-hook at one layer, for both probes and the perturbation.**
``_get_decoder_layer(model, L)`` returns the module whose *output* is
``hidden_states[L]`` -- exactly the tensor the probes were fit on. Reading,
deciding and perturbing all happen on that one tensor, inside the forward
pass that then computes the logits for the token being gated. So the
intervention lands on the pivotal token itself rather than the one after
it, and there is no second convention to keep in sync. It also forces the
gate and the steering site to share a layer, which is not a limitation but
the causal requirement: a gate read *above* the steering site could not
influence the token it fired on.

**The gate is a cascade.** ``p_steer = P(pivotal) * (1 - P(helpful))`` --
fire when the next token is likely to be a pivot *and* likely to be a
harmful one. The unsigned probe alone cannot support an intervention: its
label is ``|prob_delta| > tau``, which is sign-symmetric, so the direction
it induces has no good end. The signed probe supplies the polarity.

**Weights are raw-space by contract.** ``FittedProbe.w`` / ``.b`` come out
of ``to_raw_space``, so ``h . w + b`` is the true logit on an unstandardized
residual stream. Passing standardized coefficients here is the bug listed
above; :func:`gate_logit_check` exists so a run can assert against the
probe's own training scores instead of trusting this comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .steering import _get_decoder_layer, make_hook

# How the gate decides, per decode position.
MODE_REACTIVE = "reactive"      # cascade probe above threshold
MODE_ALWAYS_ON = "always_on"    # every position
MODE_PATTERN = "pattern"        # a supplied boolean mask (matched control)
MODE_OBSERVE = "observe"        # never perturb; only record p_steer
_GATE_MODES = {MODE_REACTIVE, MODE_ALWAYS_ON, MODE_PATTERN, MODE_OBSERVE}


@dataclass
class SteeringStats:
    """Counters over a run. All rates are over *decode* positions only."""

    n_positions: int = 0          # gate decisions made (batch rows x steps)
    n_fired: int = 0              # fresh detections
    n_held: int = 0               # positions steered by hysteresis latch
    energy: float = 0.0           # sum of ||delta|| over perturbed positions
    # Per-step (B,) arrays, so a runner can trim each row to its true
    # generated length before computing rates -- HF keeps stepping finished
    # rows until the whole batch is done, and those pad positions would
    # otherwise inflate every denominator.
    p_steer: list[np.ndarray] = field(default_factory=list)
    perturbed: list[np.ndarray] = field(default_factory=list)
    # Recorded separately because the two halves of the cascade can fail in
    # different ways. The probes were fit on raw-conditioned PTS rollouts and
    # run here on chat-templated generation, so a degenerate gate -- every
    # P(pivotal) pinned at 0 or 1 -- is a live possibility and has to be
    # visible rather than hidden inside their product.
    p_pivotal: list[np.ndarray] = field(default_factory=list)
    p_helpful: list[np.ndarray] = field(default_factory=list)

    @property
    def duty_cycle(self) -> float:
        """Fraction of positions actually perturbed, holds included.

        This is the number that has to be matched across arms, not
        ``fire_rate``: a held position receives the full perturbation.
        """
        return float(self.n_fired + self.n_held) / max(1, self.n_positions)

    @property
    def fire_rate(self) -> float:
        """Fraction of positions where the gate *newly* fired."""
        return float(self.n_fired) / max(1, self.n_positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_positions": int(self.n_positions),
            "n_fired": int(self.n_fired),
            "n_held": int(self.n_held),
            "fire_rate": self.fire_rate,
            "duty_cycle": self.duty_cycle,
            "energy": float(self.energy),
        }

    def trimmed(self, n_new: list[int]) -> dict[str, float]:
        """Duty cycle over each row's real generated tokens only.

        HF keeps stepping finished rows until the whole batch is done. Those
        pad positions are gated like any other and would otherwise inflate
        the denominator of every rate, so they are dropped here. Duty cycle,
        not fire rate, is what has to match across arms: a hysteresis-held
        position receives the full perturbation.
        """
        if not self.perturbed:
            return {"duty_cycle_trimmed": 0.0, "n_positions_trimmed": 0}
        pert = np.stack(self.perturbed)                    # (steps, B)
        keep = np.arange(pert.shape[0])[:, None] < np.asarray(n_new)[None, :]
        return {
            "duty_cycle_trimmed": float(pert[keep].mean()) if keep.any() else 0.0,
            "n_positions_trimmed": int(keep.sum()),
        }


def _as_vec(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).reshape(-1)
    return torch.tensor(np.asarray(x, dtype=np.float32)).reshape(-1)


class CascadeSteeringHook:
    """Post-hook on ``hidden_states[L]`` implementing the gated intervention.

    Args:
        model: HF causal LM.
        layer: ``L``. Both probes were fit on ``hidden_states[L]`` and the
            perturbation is applied to it.
        gate_w, gate_b: raw-space weights of the **unsigned** probe
            (is the next token pivotal).
        sign_w, sign_b: raw-space weights of the **signed** probe
            (is the impending shift helpful). May be ``None``, in which case
            the cascade degenerates to ``P(pivotal)`` -- the unsigned control
            arm, whose predicted effect is nothing.
        vector: steering direction, ``mu_helpful - mu_harmful``.
        coef: steering coefficient (meaning set by ``mode``).
        mode: ``additive_raw`` | ``additive_normalized`` | ``projection``.
        gate_mode: ``reactive`` | ``always_on`` | ``pattern`` | ``observe``.
        threshold: fire when ``p_steer > threshold``. Set it from a
            calibration pass rather than guessing: 0.5 is not a fire rate.
        hysteresis: extra positions held on after a detection.
        pattern: for ``gate_mode="pattern"``, a ``(steps,)`` or ``(steps, B)``
            boolean array consumed one step at a time.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int,
        gate_w: Any,
        gate_b: float,
        vector: Any,
        coef: float,
        *,
        sign_w: Any | None = None,
        sign_b: float = 0.0,
        mode: str = "additive_normalized",
        gate_mode: str = MODE_REACTIVE,
        threshold: float = 0.5,
        hysteresis: int = 0,
        pattern: np.ndarray | None = None,
    ) -> None:
        if gate_mode not in _GATE_MODES:
            raise ValueError(f"unknown gate_mode {gate_mode!r}; expected {_GATE_MODES}")
        if gate_mode == MODE_PATTERN and pattern is None:
            raise ValueError("gate_mode='pattern' requires a pattern")

        self.model = model
        self.layer = int(layer)
        self.coef = float(coef)
        self.mode = mode
        self.gate_mode = gate_mode
        self.threshold = float(threshold)
        self.hysteresis = int(hysteresis)
        self.pattern = None if pattern is None else np.asarray(pattern, dtype=bool)

        self.gate_w = _as_vec(gate_w)
        self.gate_b = float(gate_b)
        self.sign_w = None if sign_w is None else _as_vec(sign_w)
        self.sign_b = float(sign_b)
        self.vector = _as_vec(vector)

        self._handle: Any | None = None
        self._latch: torch.Tensor | None = None
        self._step = 0
        self.stats = SteeringStats()

    # -- gating ------------------------------------------------------------

    def probs(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(P(pivotal), P(helpful))`` for a ``(B, d)`` residual batch."""
        hf = h.float()
        p_piv = torch.sigmoid(hf @ self.gate_w.to(hf.device) + self.gate_b)
        if self.sign_w is None:
            return p_piv, None
        return p_piv, torch.sigmoid(hf @ self.sign_w.to(hf.device) + self.sign_b)

    def p_steer(self, h: torch.Tensor) -> torch.Tensor:
        """``P(pivotal) * (1 - P(helpful))`` for a ``(B, d)`` residual batch."""
        p_piv, p_help = self.probs(h)
        return p_piv if p_help is None else p_piv * (1.0 - p_help)

    def _decide(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(perturb, p_steer)`` for a ``(B, d)`` batch."""
        p_piv, p_help = self.probs(h)
        p = p_piv if p_help is None else p_piv * (1.0 - p_help)
        self.stats.p_pivotal.append(p_piv.detach().float().cpu().numpy())
        if p_help is not None:
            self.stats.p_helpful.append(p_help.detach().float().cpu().numpy())
        b = h.shape[0]

        if self.gate_mode == MODE_OBSERVE:
            return torch.zeros(b, dtype=torch.bool, device=h.device), p
        if self.gate_mode == MODE_ALWAYS_ON:
            return torch.ones(b, dtype=torch.bool, device=h.device), p
        if self.gate_mode == MODE_PATTERN:
            row = self.pattern[self._step] if self._step < len(self.pattern) else False
            row = np.asarray(row, dtype=bool)
            # The final batch of a run is usually short, so a pattern built
            # for the full batch width has to be sliced, not broadcast.
            row = row[:b] if row.ndim and row.size >= b else np.broadcast_to(row, (b,))
            fresh = torch.as_tensor(row.copy(), device=h.device)
        else:
            fresh = p > self.threshold

        if self._latch is None or self._latch.shape[0] != b:
            self._latch = torch.zeros(b, dtype=torch.long, device=h.device)

        held = (~fresh) & (self._latch > 0)
        self._latch = torch.where(
            fresh,
            torch.full_like(self._latch, self.hysteresis),
            torch.clamp(self._latch - 1, min=0),
        )

        self.stats.n_fired += int(fresh.sum().item())
        self.stats.n_held += int(held.sum().item())
        return fresh | held, p

    # -- hook --------------------------------------------------------------

    def _post_hook(self, module: nn.Module, args: tuple, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        if hidden.shape[1] != 1:
            # Prefill. The prompt is not generated text and carries no PTS
            # labels, so it is never steered and never counted.
            return output

        h = hidden[:, -1, :]
        perturb, p = self._decide(h)
        self.stats.n_positions += int(h.shape[0])
        self.stats.p_steer.append(p.detach().float().cpu().numpy())
        self.stats.perturbed.append(perturb.detach().cpu().numpy())
        self._step += 1

        if not bool(perturb.any()):
            return output

        mask = perturb.view(-1, 1)
        inner = make_hook(self.vector, self.coef, mode=self.mode, position_mask=mask)
        new_hidden = inner(module, args, hidden)
        if isinstance(new_hidden, tuple):
            new_hidden = new_hidden[0]
        self.stats.energy += float((new_hidden - hidden).float().norm(dim=-1).sum().item())
        return (new_hidden,) + tuple(output[1:]) if isinstance(output, tuple) else new_hidden

    def __enter__(self) -> "CascadeSteeringHook":
        target = _get_decoder_layer(self.model, self.layer)
        self._handle = target.register_forward_hook(self._post_hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self) -> None:
        """Clear the latch, the step cursor and the stats.

        Call this **between batches**. The old implementation never did, so
        the hysteresis latch leaked across example boundaries and a fire on
        the last token of one example steered the first token of the next.
        """
        self.stats = SteeringStats()
        self._latch = None
        self._step = 0


def build_random_pattern(
    n_steps: int, batch: int, rate: float, seed: int = 101
) -> np.ndarray:
    """Bernoulli ``(n_steps, batch)`` mask at the requested expected rate.

    The matched control: same expected duty cycle as the reactive arm, same
    direction, same coefficient -- so the only difference is *where* it
    fires. If reactive beats this, the gate is carrying the effect rather
    than the perturbation energy.
    """
    rng = np.random.default_rng(seed)
    return rng.random(size=(n_steps, batch)) < float(rate)


def gate_logit_check(
    hook: CascadeSteeringHook,
    x: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-3,
) -> dict[str, float]:
    """Assert the hook's gate reproduces the probe's own training scores.

    ``expected`` is ``probe.decision(x)`` from the fitted probe. If the
    weights were exported in the wrong basis this fails loudly, which is
    the whole point -- the previous gate was off by two orders of magnitude
    in the bias alone and nothing caught it.
    """
    got = (torch.from_numpy(np.asarray(x, dtype=np.float32)).float()
           @ hook.gate_w + hook.gate_b).numpy()
    err = float(np.max(np.abs(got - np.asarray(expected, dtype=np.float32))))
    if err > atol:
        raise AssertionError(
            f"gate logits disagree with the probe by {err:.4g} (> {atol}); "
            "weights are probably in standardized space, not raw space"
        )
    return {"max_abs_err": err, "mean_logit": float(got.mean())}
