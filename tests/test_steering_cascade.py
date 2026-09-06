"""Pins the conventions the previous reactive hook got wrong.

Each test here corresponds to a specific defect that shipped and produced
published numbers before anyone noticed:

* the hook read one layer below the probe, and the notebook that "checked
  the convention" checked a different hook type than the code ran;
* coefficient 0 was never asserted to be a no-op, so the plumbing itself
  was never separated from the perturbation;
* duty cycle excluded hysteresis holds, so a "matched" control received
  39% less perturbation energy than the arm it was matched to;
* the latch was never reset, so a fire on one example's last token steered
  the next example's first token;
* the gate applied standardized-space weights to raw activations.

They run on a stub stack, so there is no model download and no skip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from probe_pipeline.steering_reactive import (
    MODE_ALWAYS_ON,
    MODE_OBSERVE,
    MODE_PATTERN,
    MODE_REACTIVE,
    CascadeSteeringHook,
    SteeringStats,
    build_random_pattern,
    gate_logit_check,
)

D = 8
N_LAYERS = 4


class _Block(nn.Module):
    """A layer that returns a tuple, like a real decoder block."""

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = k

    def forward(self, x, *args, **kwargs):  # noqa: ANN001
        return (x + self.k,)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList(_Block(float(i + 1)) for i in range(N_LAYERS))


class _Stub(nn.Module):
    """Mimics the ``model.model.layers`` shape ``_get_decoder_layer`` walks."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, x):  # noqa: ANN001
        states = [x]
        for blk in self.model.layers:
            x = blk(x)[0]
            states.append(x)
        return states


def _hook_kwargs(**over):
    base = dict(
        layer=2,
        gate_w=np.zeros(D, dtype=np.float32),
        gate_b=0.0,
        vector=np.ones(D, dtype=np.float32),
        coef=1.0,
        sign_w=np.zeros(D, dtype=np.float32),
        sign_b=0.0,
        mode="additive_raw",
        gate_mode=MODE_ALWAYS_ON,
    )
    base.update(over)
    return base


def _decode_step(b: int = 3) -> torch.Tensor:
    return torch.arange(b * D, dtype=torch.float32).reshape(b, 1, D)


def test_post_hook_sees_hidden_states_at_the_probe_layer() -> None:
    """The tensor the hook reads must be ``hidden_states[L]``, not [L-1].

    This is the defect that made every previous reactive number vacuous.
    """
    m = _Stub()
    seen = {}

    hook = CascadeSteeringHook(m, **_hook_kwargs(coef=0.0, gate_mode=MODE_OBSERVE))
    orig = hook.probs

    def spy(h):
        seen["h"] = h.detach().clone()
        return orig(h)

    hook.probs = spy
    x = _decode_step()
    with hook:
        states = m(x)

    assert torch.allclose(seen["h"], states[2][:, -1, :])
    assert not torch.allclose(seen["h"], states[1][:, -1, :])


def test_coef_zero_is_exactly_identity() -> None:
    m = _Stub()
    x = _decode_step()
    clean = m(x)[-1].clone()
    with CascadeSteeringHook(m, **_hook_kwargs(coef=0.0)):
        steered = m(x)[-1]
    assert torch.equal(clean, steered)


def test_perturbation_lands_at_the_requested_layer() -> None:
    m = _Stub()
    x = _decode_step()
    clean = m(x)
    with CascadeSteeringHook(m, **_hook_kwargs(coef=2.0)):
        steered = m(x)
    # additive_raw with vector=ones, coef=2 -> +2 everywhere from layer 2 on.
    assert torch.equal(clean[1], steered[1])
    assert torch.allclose(steered[2] - clean[2], torch.full_like(clean[2], 2.0))
    assert torch.allclose(steered[4] - clean[4], torch.full_like(clean[4], 2.0))


def test_prefill_is_never_steered_or_counted() -> None:
    m = _Stub()
    prefill = torch.zeros(2, 5, D)
    with CascadeSteeringHook(m, **_hook_kwargs(coef=3.0)) as h:
        out = m(prefill)
    assert torch.equal(out[-1], _Stub()(prefill)[-1])
    assert h.stats.n_positions == 0


def test_duty_cycle_counts_hysteresis_holds() -> None:
    """A held position gets the full perturbation, so it must count.

    The old ``fire_rate`` excluded holds and reported 0.570 for a duty cycle
    that was really 0.878.
    """
    stats = SteeringStats(n_positions=100, n_fired=30, n_held=48)
    assert stats.fire_rate == pytest.approx(0.30)
    assert stats.duty_cycle == pytest.approx(0.78)


def test_hysteresis_holds_then_releases() -> None:
    m = _Stub()
    # Gate fires only when the residual is large: bias makes step 1 fire.
    w = np.zeros(D, dtype=np.float32)
    w[0] = 1.0
    hook = CascadeSteeringHook(
        m, **_hook_kwargs(gate_w=w, gate_b=-50.0, sign_w=None,
                          gate_mode=MODE_REACTIVE, threshold=0.5, hysteresis=2)
    )
    hot = torch.zeros(1, 1, D)
    hot[0, 0, 0] = 100.0          # logit +50 -> fires
    cold = torch.zeros(1, 1, D)   # logit -50 -> does not
    with hook:
        m(hot)
        m(cold)
        m(cold)
        m(cold)
    assert hook.stats.n_fired == 1
    assert hook.stats.n_held == 2          # two holds, then release
    assert hook.stats.duty_cycle == pytest.approx(3 / 4)


def test_reset_clears_the_latch_between_batches() -> None:
    m = _Stub()
    w = np.zeros(D, dtype=np.float32)
    w[0] = 1.0
    hook = CascadeSteeringHook(
        m, **_hook_kwargs(gate_w=w, gate_b=-50.0, sign_w=None,
                          gate_mode=MODE_REACTIVE, threshold=0.5, hysteresis=5)
    )
    hot = torch.zeros(1, 1, D)
    hot[0, 0, 0] = 100.0
    cold = torch.zeros(1, 1, D)
    with hook:
        m(hot)
        hook.reset()
        m(cold)
    assert hook.stats.n_held == 0, "latch leaked across the reset"


def test_cascade_needs_both_probes_to_fire() -> None:
    """``p_steer = P(pivotal) * (1 - P(helpful))``.

    A confidently-helpful pivot must not fire: steering toward "helpful"
    when the model is already about to help is the intervention doing harm.
    """
    m = _Stub()
    w = np.zeros(D, dtype=np.float32)
    w[0] = 1.0
    h = torch.zeros(2, D)
    h[:, 0] = 10.0                      # both rows: pivotal with high prob

    pivotal_only = CascadeSteeringHook(m, **_hook_kwargs(gate_w=w, sign_w=None))
    assert float(pivotal_only.p_steer(h)[0]) > 0.99

    # sign probe says "helpful" with the same confidence -> cascade collapses
    helpful = CascadeSteeringHook(m, **_hook_kwargs(gate_w=w, sign_w=w, sign_b=0.0))
    assert float(helpful.p_steer(h)[0]) < 0.01

    # sign probe says "harmful" -> cascade fires
    harmful = CascadeSteeringHook(m, **_hook_kwargs(gate_w=w, sign_w=-w, sign_b=0.0))
    assert float(harmful.p_steer(h)[0]) > 0.99


def test_pattern_mode_handles_a_short_final_batch() -> None:
    """A pattern built for the full batch width must slice, not broadcast."""
    m = _Stub()
    pat = np.zeros((3, 8), dtype=bool)
    pat[0, :] = True
    hook = CascadeSteeringHook(
        m, **_hook_kwargs(gate_mode=MODE_PATTERN, pattern=pat, coef=2.0)
    )
    x = torch.zeros(3, 1, D)            # batch 3, pattern width 8
    with hook:
        out = m(x)
    assert hook.stats.n_fired == 3
    assert torch.allclose(out[-1] - _Stub()(x)[-1], torch.full_like(x, 2.0))


def test_trimmed_drops_padding_positions() -> None:
    """Finished rows keep getting stepped; those positions are not real."""
    st = SteeringStats()
    # 4 steps, 2 rows; row 0 really produced 2 tokens, row 1 produced 4.
    st.perturbed = [np.array([True, False]), np.array([True, False]),
                    np.array([True, True]), np.array([True, True])]
    out = st.trimmed([2, 4])
    assert out["n_positions_trimmed"] == 6          # 2 + 4, not 8
    assert out["duty_cycle_trimmed"] == pytest.approx(4 / 6)


def test_observe_mode_never_perturbs_but_records() -> None:
    m = _Stub()
    x = _decode_step()
    clean = m(x)[-1].clone()
    hook = CascadeSteeringHook(m, **_hook_kwargs(coef=5.0, gate_mode=MODE_OBSERVE))
    with hook:
        out = m(x)
    assert torch.equal(clean, out[-1])
    assert hook.stats.n_positions == 3
    assert hook.stats.n_fired == 0
    assert len(hook.stats.p_steer) == 1


def test_gate_logit_check_rejects_standardized_weights() -> None:
    """The exact failure that shipped: w/sigma applied to unstandardized h."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, D)).astype(np.float32) * 5.0 + 3.0
    w_raw = rng.normal(size=D).astype(np.float32)
    b_raw = -1.354
    expected = x @ w_raw + b_raw

    m = _Stub()
    good = CascadeSteeringHook(m, **_hook_kwargs(gate_w=w_raw, gate_b=b_raw))
    assert gate_logit_check(good, x, expected)["max_abs_err"] < 1e-3

    bad = CascadeSteeringHook(m, **_hook_kwargs(gate_w=w_raw * 5.0, gate_b=-0.0122))
    with pytest.raises(AssertionError, match="standardized"):
        gate_logit_check(bad, x, expected)


def test_random_pattern_hits_the_requested_rate() -> None:
    pat = build_random_pattern(2000, 16, 0.05, seed=3)
    assert pat.shape == (2000, 16)
    assert abs(pat.mean() - 0.05) < 0.005


def test_unknown_gate_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="gate_mode"):
        CascadeSteeringHook(_Stub(), **_hook_kwargs(gate_mode="nope"))
    with pytest.raises(ValueError, match="pattern"):
        CascadeSteeringHook(_Stub(), **_hook_kwargs(gate_mode=MODE_PATTERN))
