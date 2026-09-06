#!/usr/bin/env python
"""Phase 6: probe-gated steering on GSM8K, with the controls that matter.

The project has been here before: 114 steering arms were compared against
their baselines and 4 reached nominal p < 0.05 where chance predicts 5.7.
The difference this time is not a better direction, it is that the
comparison is designed before it is run.

**One primary arm.** ``reactive`` at the pre-registered fire rate and
coefficient, against base, by exact McNemar on greedy decoding. Everything
else is secondary: a dose-response curve reported as a curve, and controls
reported as controls. No arm other than the primary carries a p-value that
means anything on its own.

**Greedy, so pairing is exact.** Base and steered arms see identical
prompts and a deterministic decode, so every flipped answer is
attributable to the intervention. The v1 harness seeded once per arm and
consumed the RNG across variable-length generations, so base and steered
were genuinely paired only for example 0 -- which is why a perturbation of
0.1% of the residual norm appeared to flip 20 answers.

**Four controls, each ruling out a different thing.**

* ``always_on`` at matched energy -- if this matches reactive, the gate is
  doing nothing and only the perturbation budget matters. This is the
  comparison the paper's claim actually rests on.
* ``random_placement`` at matched duty cycle -- same direction, same
  coefficient, same number of fires, different positions. Isolates *where*
  from *how much*.
* ``unsigned_dir`` -- steers along the unsigned direction, whose label
  ``|prob_delta| > tau`` is sign-symmetric. Predicted to do nothing; that
  prediction is the explanation for the historical nulls.
* ``flip`` -- the primary arm with the coefficient negated. A real effect
  must reverse. In the old runs coef +0.96 and -1.04 had identical ||delta||
  and both scored above baseline, which is what a noise floor looks like.

Thresholds are not guessed. An observe pass runs the gate without
perturbing, and the firing threshold is read off the resulting quantiles,
so a "5% fire rate" arm fires on 5% of decode positions by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_pipeline.gsm8k_eval_v2 import (  # noqa: E402
    build_prompts, extract_gold, generate_batched, mcnemar, score,
)
from probe_pipeline.steering_reactive import (  # noqa: E402
    MODE_ALWAYS_ON, MODE_OBSERVE, MODE_PATTERN, MODE_REACTIVE,
    CascadeSteeringHook, build_random_pattern,
)


# The coefficient at which each mode leaves the residual stream untouched.
# Getting this wrong turns the identity check into an ablation check.
IDENTITY_COEF = {"additive_raw": 0.0, "additive_normalized": 0.0,
                 "projection": 1.0}


def gate_check(npz, atol: float = 2e-2) -> dict:
    """Assert the exported weights still reproduce the probe's own logits.

    The historical failure was silent: standardized coefficients applied to
    raw activations, bias off by two orders of magnitude, and a gate that
    fired on 58% of rows where the true probe fired on 45%. This makes that
    loud. The tolerance is loose because the check rows are stored in the
    bf16 activation cache's precision, not the fitting precision.
    """
    x = npz["check_x"].astype(np.float32)
    got_gate = x @ npz["gate_w"] + float(npz["gate_b"])
    got_sign = x @ npz["sign_w"] + float(npz["sign_b"])
    e_gate = float(np.max(np.abs(got_gate - npz["check_gate"])))
    e_sign = float(np.max(np.abs(got_sign - npz["check_sign"])))
    if max(e_gate, e_sign) > atol:
        raise AssertionError(
            f"exported weights do not reproduce the probe logits "
            f"(gate {e_gate:.4g}, sign {e_sign:.4g} > {atol}); "
            "they are probably in standardized space"
        )
    return {"gate_max_abs_err": e_gate, "sign_max_abs_err": e_sign,
            "mean_gate_logit": float(got_gate.mean()),
            "mean_sign_logit": float(got_sign.mean())}


def run_arm(model, tok, prompts, golds, *, name, hook_kwargs, max_new_tokens,
            batch_size, seed=0, greedy=True):
    """One generation pass, optionally hooked. Returns (ArmResult, stats)."""
    if hook_kwargs is None:
        resp, nnew = generate_batched(model, tok, prompts, greedy=greedy, seed=seed,
                                      max_new_tokens=max_new_tokens, batch_size=batch_size)
        return score(resp, golds, nnew, name=name, max_new_tokens=max_new_tokens), {}

    # The hook must be reset between batches: its latch and step cursor are
    # per-batch state, and generate() is called once per batch inside
    # generate_batched. Rather than reach inside, run batch by batch here.
    resp, nnew, stats = [], [], []
    p_all, piv_all, help_all, norm_all = [], [], [], []
    delta_all = []
    norm_kept = []   # ||delta|| at perturbed, non-padding positions
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i:i + batch_size])
        kw = dict(hook_kwargs)
        if "pattern_rate" in kw:
            # Rebuild per batch. The hook is constructed per batch and its
            # step cursor restarts, so a pattern built once would fire at the
            # identical step indices in every batch -- random within a
            # sequence but perfectly correlated across them, which quietly
            # reduces the control to far fewer independent draws than it
            # appears to have.
            kw["pattern"] = build_random_pattern(
                max_new_tokens, batch_size, kw.pop("pattern_rate"),
                seed=101 + i,
            )
        hook = CascadeSteeringHook(model, **kw)
        with hook:
            r, k = generate_batched(model, tok, chunk, greedy=greedy, seed=seed + i,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)
        resp.extend(r)
        nnew.extend(k)
        s = hook.stats.to_dict()
        s.update(hook.stats.trimmed(k))
        stats.append(s)
        if hook.stats.p_steer:
            keep = None
            for key, sink in (("p_steer", p_all), ("p_pivotal", piv_all),
                              ("p_helpful", help_all), ("h_norm", norm_all)):
                seq = getattr(hook.stats, key)
                if not seq:
                    continue
                arr = np.stack(seq)                               # (steps, B)
                if keep is None:
                    keep = np.arange(arr.shape[0])[:, None] < np.asarray(k)[None, :]
                sink.append(arr[keep])
            pert = np.stack(hook.stats.perturbed)
            norm_kept.append(np.stack(hook.stats.delta_norm)[keep & pert])
        hook.reset()

    agg = {
        "n_positions": sum(s["n_positions"] for s in stats),
        "n_fired": sum(s["n_fired"] for s in stats),
        "n_held": sum(s["n_held"] for s in stats),
        "energy": sum(s["energy"] for s in stats),
    }
    agg["n_positions_trimmed"] = sum(s["n_positions_trimmed"] for s in stats)
    # Energy over each row's *real* generated tokens. The raw sum includes the
    # padding positions that keep being stepped after a row emits EOS, and
    # always-on perturbs all of them while a 5% gate mostly does not -- which
    # made a correctly matched pair look mismatched by 1.9x. In
    # additive_normalized ||delta|| is exactly |coef|*||h||, so this is
    # recoverable from what is already recorded.
    agg["energy_trimmed"] = float(
        np.concatenate(norm_kept).sum() if norm_kept else 0.0
    )
    agg["duty_cycle"] = (agg["n_fired"] + agg["n_held"]) / max(1, agg["n_positions"])
    agg["fire_rate"] = agg["n_fired"] / max(1, agg["n_positions"])
    # Weighted by each batch's real generated positions, so a short final
    # batch does not count as much as a full one.
    agg["duty_cycle_trimmed"] = sum(
        s["duty_cycle_trimmed"] * s["n_positions_trimmed"] for s in stats
    ) / max(1, agg["n_positions_trimmed"])
    for key, sink in (("p_steer", p_all), ("p_pivotal", piv_all),
                      ("p_helpful", help_all), ("h_norm", norm_all)):
        agg[key] = np.concatenate(sink) if sink else np.zeros(0)
    return score(resp, golds, nnew, name=name, max_new_tokens=max_new_tokens), agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True, help="npz from fit_steering_probes.py")
    ap.add_argument("--band", default=None, help="band json from screen_band.py")
    ap.add_argument("--n", type=int, default=300, help="cap on evaluated questions")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bands", default=None,
                    help="comma-separated band files to pool, when one split "
                         "does not supply enough in-band questions")
    ap.add_argument("--mode", default="additive_normalized")
    ap.add_argument("--alpha", type=float, default=0.05, help="primary coefficient")
    ap.add_argument("--rate", type=float, default=0.05, help="primary fire rate")
    ap.add_argument("--alpha-grid", default="0.02,0.10")
    ap.add_argument("--rate-grid", default="0.02,0.10")
    ap.add_argument("--gate", default="cascade", choices=["cascade", "pivotal"],
                    help="'cascade' is P(pivotal)*(1-P(helpful)); 'pivotal' is "
                         "the plain unsigned probe, which is what the original "
                         "PTS label supports and what a reader would try first.")
    ap.add_argument("--direction", default="signed_caa",
                    choices=["signed_caa", "unsigned_caa",
                             "signed_probe", "unsigned_probe"],
                    help="'caa' is a mean difference, 'probe' is the logistic "
                         "regression weight vector. They are not the same "
                         "object and need not point the same way.")
    ap.add_argument("--n-random-dirs", type=int, default=3,
                    help="independent random directions to average the "
                         "random-direction control over")
    ap.add_argument("--hyst-grid", default="",
                    help="exploratory: hold the gate on for N extra positions "
                         "after a detection. A single-token nudge may be too "
                         "transient to change a trajectory.")
    ap.add_argument("--hysteresis", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    choices=["primary", "all", "ablation"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--device", default="cuda",
                    help="cuda on a VM; mps/cpu to dry-run the arms locally")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    npz = np.load(a.probes)
    layer = int(npz["layer"])
    checks = gate_check(npz)
    print(f"[gate] weights verified: {checks}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("openai/gsm8k", "main", split=a.split)
    band_files = [f for f in ((a.bands or a.band or "").split(",")) if f]
    if band_files:
        idx = []
        for f in band_files:
            b = json.loads(Path(f).read_text())
            if b.get("split", "test") != a.split:
                raise SystemExit(f"{f} screened split {b.get('split')!r}, "
                                 f"but --split is {a.split!r}")
            idx.extend(b["indices"])
        idx = sorted(set(idx))[: a.n]
        ds = ds.select(idx)
        print(f"[band] {len(idx)} in-band questions from "
              f"{len(band_files)} screen(s)", flush=True)
    else:
        ds = ds.select(range(min(a.n, len(ds))))
    questions = [r["question"] for r in ds]
    golds = [extract_gold(r["answer"]) for r in ds]
    prompts = build_prompts(questions, tok)

    print(f"[load] {a.model} layer {layer} | {len(prompts)} questions", flush=True)
    dtype = torch.float32 if a.device in ("cpu", "mps") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()

    # The direction actually steered along. Four are exported: mean-difference
    # and probe-weight, each signed and unsigned. The unsigned pair comes from
    # the sign-symmetric |prob_delta| > tau label, so it has no principled
    # polarity -- which is the point of offering it as a comparison rather
    # than assuming the signed one is the only candidate worth testing.
    v_primary = npz[{
        "signed_caa": "v_signed", "unsigned_caa": "v_unsigned",
        "signed_probe": "v_signed_probe", "unsigned_probe": "v_unsigned_probe",
    }[a.direction]]
    v_signed = v_primary
    contrast_name = "v_unsigned" if a.direction.startswith("signed") else "v_signed"
    v_unsigned = npz[contrast_name]
    # Key the contrast arm by the direction it actually uses. Naming it
    # "unsigned_direction" unconditionally was wrong the moment the primary
    # became unsigned, and it is the kind of error that survives into a
    # results table unnoticed.
    contrast_key = ("unsigned_direction" if contrast_name == "v_unsigned"
                    else "signed_direction")
    # Several independent draws, not one. A single random vector has real
    # variance, and this project has already been caught reporting one draw
    # as if it were a null distribution (the AUROC control averaged eight
    # vectors and then projected once, which is one sample, not eight).
    rng = np.random.default_rng(7)
    v_randoms = [rng.normal(size=v_signed.shape).astype(np.float32)
                 for _ in range(a.n_random_dirs)]
    v_random = v_randoms[0]

    base_kw = dict(layer=layer, gate_w=npz["gate_w"], gate_b=float(npz["gate_b"]),
                   sign_w=npz["sign_w"] if a.gate == "cascade" else None,
                   sign_b=float(npz["sign_b"]),
                   mode=a.mode, hysteresis=a.hysteresis)
    run = lambda **kw: run_arm(model, tok, prompts, golds, max_new_tokens=a.max_new_tokens,  # noqa: E731
                               batch_size=a.batch_size, **kw)
    results, t0 = {}, time.time()

    # -- base, and the observe pass that calibrates every threshold ---------
    print("[1] base + observe", flush=True)
    obs_arm, obs = run(name="base", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=IDENTITY_COEF[a.mode],
        gate_mode=MODE_OBSERVE))
    base_mask = obs_arm.correct_mask
    results["base"] = obs_arm.as_dict()
    p = obs["p_steer"]

    def dist(x):
        return {"mean": float(x.mean()), "sd": float(x.std()),
                "p05": float(np.quantile(x, 0.05)), "p50": float(np.quantile(x, 0.50)),
                "p95": float(np.quantile(x, 0.95)), "max": float(x.max())}

    # The probes were fit on raw-conditioned PTS rollouts; generation here is
    # chat-templated. If that shift has broken the gate it shows up as a
    # degenerate distribution -- everything pinned near 0 or 1 -- and the
    # thresholds below would then be quantiles of noise.
    results["observe"] = {
        "n_decode_positions": int(len(p)),
        "p_steer": dist(p),
        "p_pivotal": dist(obs["p_pivotal"]) if len(obs["p_pivotal"]) else None,
        "p_helpful": dist(obs["p_helpful"]) if len(obs["p_helpful"]) else None,
        "train_logit_means": {k: checks[k] for k in ("mean_gate_logit", "mean_sign_logit")},
    }
    print(f"    base {obs_arm.accuracy:.4f} | {len(p)} decode positions", flush=True)
    for k in ("p_steer", "p_pivotal", "p_helpful"):
        d = results["observe"][k]
        if d:
            print(f"    {k:<10} mean {d['mean']:.4f} sd {d['sd']:.4f} "
                  f"p50 {d['p50']:.4f} p95 {d['p95']:.4f} max {d['max']:.4f}", flush=True)

    def thresh(rate: float, scores: np.ndarray | None = None) -> float:
        """Threshold achieving ``rate`` on the gate's *own* score distribution.

        Each gate variant needs its own calibration: P(pivotal) alone is much
        larger than the cascade P(pivotal)*(1-P(helpful)), so reusing the
        cascade's quantile made the unsigned-gate control fire on 53% of
        positions where it was meant to fire on 5%.
        """
        return float(np.quantile(p if scores is None else scores, 1.0 - rate))

    # -- identity: the hook plumbing itself must be a no-op ----------------
    # Exactly one full batch, so the comparison is against the base run's
    # *first* batch and nothing else differs. Left padding pads each batch to
    # its own longest prompt, and in bf16 that is enough to flip a couple of
    # greedy answers -- so a 32-prompt check against a 48-prompt base batch
    # fails for reasons that have nothing to do with the hook. Arms are
    # exactly paired only because they all share these batch boundaries.
    n_id = min(a.batch_size, len(prompts))
    small = prompts[:n_id]
    ident, _ = run_arm(model, tok, small, golds[:n_id], name="identity",
                       max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
                       hook_kwargs=dict(base_kw, vector=v_signed,
                                        coef=IDENTITY_COEF[a.mode],
                                        gate_mode=MODE_ALWAYS_ON))
    diff = [i for i, (x, y) in enumerate(zip(ident.correct_mask, base_mask[:n_id]))
            if x != y]
    ok = not diff
    results["identity_check"] = {"passed": bool(ok), "n": n_id,
                                 "n_mismatched": len(diff)}
    print(f"[2] identity (coef=0, always on): {'PASS' if ok else 'FAIL'} "
          f"({len(diff)}/{n_id} mismatched)", flush=True)
    if not ok:
        print("    the hook changes generation at coef=0; stopping", flush=True)
        Path(a.out or "/content/steer.json").write_text(json.dumps(results, indent=2))
        return

    def record(nm, arm, st, primary=False):
        d = arm.as_dict()
        d.update({k: v for k, v in st.items()
                  if k not in ("p_steer", "p_pivotal", "p_helpful",
                               "h_norm", "delta_norm")})
        d["vs_base"] = mcnemar(base_mask, arm.correct_mask)
        d["delta_acc"] = arm.accuracy - obs_arm.accuracy
        d["primary"] = primary
        results[nm] = d
        print(f"    {nm:<24} acc {arm.accuracy:.4f} ({d['delta_acc']:+.4f})  "
              f"duty {st.get('duty_cycle_trimmed', 0):.3f}  "
              f"energy {st.get('energy_trimmed', 0):.0f}  "
              f"net {d['vs_base']['net']:+d}  p {d['vs_base']['p']:.3f}", flush=True)
        return d

    if a.stage == "ablation":
        # A different causal question from the additive arms. Adding a
        # direction asks whether it can be injected; removing the model's
        # existing component along it asks whether the model *uses* it. The
        # earlier NIE work in this project found its clearest signal in
        # ablation (-0.516 for the CAA direction against +-0.008 for the
        # additive probe direction), and running only additive arms left the
        # more informative test undone.
        #
        # Matching is by duty cycle, not energy: ablation has no injected
        # magnitude to match, so the control that isolates *where* is the
        # same number of ablations at random positions.
        print("[3] ablation (projection mode)", flush=True)
        for nm, coef, gm, vec, extra in [
            ("ablate_reactive", 0.0, MODE_REACTIVE, v_signed, {}),
            ("ablate_always_on", 0.0, MODE_ALWAYS_ON, v_signed, {}),
            ("ablate_random_placement", 0.0, MODE_PATTERN, v_signed, {}),
            *[(f"ablate_random_direction{'' if j == 0 else f'_{j}'}",
               0.0, MODE_REACTIVE, vr, {}) for j, vr in enumerate(v_randoms)],
            (f"ablate_{contrast_key}", 0.0, MODE_REACTIVE, v_unsigned, {}),
            ("amplify_reactive", 2.0, MODE_REACTIVE, v_signed, {}),
            ("amplify_always_on", 2.0, MODE_ALWAYS_ON, v_signed, {}),
            # The always-on arms are ~20x the perturbation of the gated ones,
            # and they order monotonically (ablate helps, amplify hurts). That
            # ordering is only evidence about *this* direction if a random one
            # does not reproduce it -- otherwise it just says scaling any
            # component up hurts more than scaling it down.
            ("ablate_always_on_random", 0.0, MODE_ALWAYS_ON, v_random, {}),
            ("amplify_always_on_random", 2.0, MODE_ALWAYS_ON, v_random, {}),
            (f"ablate_always_on_{contrast_key}", 0.0, MODE_ALWAYS_ON, v_unsigned, {}),
            (f"amplify_always_on_{contrast_key}", 2.0, MODE_ALWAYS_ON, v_unsigned, {}),
        ]:
            kw = dict(base_kw, vector=vec, coef=coef, gate_mode=gm, **extra)
            if gm == MODE_REACTIVE:
                kw["threshold"] = thresh(a.rate)
            elif gm == MODE_PATTERN:
                kw["pattern_rate"] = a.rate
            arm, st = run(name=nm, hook_kwargs=kw)
            record(nm, arm, st, primary=(nm == "ablate_reactive"))
        _finish(a, results, t0, results["ablate_reactive"])
        return

    # -- primary -----------------------------------------------------------
    print(f"[3] primary: reactive rate={a.rate} alpha={a.alpha}", flush=True)
    arm, st = run(name="reactive", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
    prim = record("reactive", arm, st, primary=True)
    duty = max(st["duty_cycle_trimmed"], 1e-6)

    # -- matched controls --------------------------------------------------
    print("[4] matched controls", flush=True)
    # Match injected energy, not duty cycle: sum_{would fire} ||h|| over
    # sum_{all} ||h||, measured in the observe pass. The gate concentrates on
    # high-norm positions, so duty-matching alone left always-on ~20% short.
    # obs["h_norm"] is already trimmed to real tokens, so this fraction and
    # the energy_trimmed it is matched against are on the same footing.
    hn, fired_mask = obs["h_norm"], p > thresh(a.rate)
    energy_frac = float(hn[fired_mask].sum() / max(hn.sum(), 1e-9))
    results["energy_match"] = {"duty_estimate": float(fired_mask.mean()),
                               "energy_fraction": energy_frac}
    arm, st = run(name="always_on", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=a.alpha * energy_frac,
        gate_mode=MODE_ALWAYS_ON))
    record("always_on_matched_energy", arm, st)

    arm, st = run(name="random_placement", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=a.alpha, gate_mode=MODE_PATTERN,
        pattern_rate=duty))
    record("random_placement_matched", arm, st)

    arm, st = run(name="flip", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=-a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
    record("sign_flipped", arm, st)

    if a.stage == "primary":
        _finish(a, results, t0, prim)
        return

    # -- direction and gate controls ---------------------------------------
    print("[5] direction / gate controls", flush=True)
    # Named for what it is relative to the primary: the other direction.
    arm, st = run(name="other_dir", hook_kwargs=dict(
        base_kw, vector=v_unsigned, coef=a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
    record(contrast_key, arm, st)

    for j, vr in enumerate(v_randoms):
        arm, st = run(name=f"random_dir{j}", hook_kwargs=dict(
            base_kw, vector=vr, coef=a.alpha,
            gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
        record("random_direction" if j == 0 else f"random_direction_{j}", arm, st)

    # Gate on P(pivotal) alone: the cascade's signed half removed, so the
    # same direction fires on pivots regardless of whether they help.
    if a.gate == "cascade":
        ung = dict(base_kw)
        ung["sign_w"] = None
        arm, st = run(name="unsigned_gate", hook_kwargs=dict(
            ung, vector=v_primary, coef=a.alpha, gate_mode=MODE_REACTIVE,
            threshold=thresh(a.rate, obs["p_pivotal"])))
        record("unsigned_gate", arm, st)

    # -- dose response -----------------------------------------------------
    print("[6] dose response", flush=True)
    for al in [float(x) for x in a.alpha_grid.split(",") if x]:
        arm, st = run(name=f"a{al}", hook_kwargs=dict(
            base_kw, vector=v_signed, coef=al,
            gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
        record(f"dose_alpha_{al}", arm, st)
    for r in [float(x) for x in a.rate_grid.split(",") if x]:
        arm, st = run(name=f"r{r}", hook_kwargs=dict(
            base_kw, vector=v_signed, coef=a.alpha,
            gate_mode=MODE_REACTIVE, threshold=thresh(r)))
        record(f"dose_rate_{r}", arm, st)
    for hy in [int(x) for x in a.hyst_grid.split(",") if x]:
        kw = dict(base_kw)
        kw["hysteresis"] = hy
        arm, st = run(name=f"h{hy}", hook_kwargs=dict(
            kw, vector=v_signed, coef=a.alpha,
            gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
        record(f"dose_hysteresis_{hy}", arm, st)

    _finish(a, results, t0, prim)


def _finish(a, results, t0, prim) -> None:
    results["_meta"] = {
        "model": a.model, "probes": a.probes, "band": a.band,
        "gate": a.gate, "direction": a.direction, "stage": a.stage,
        "contrast_direction": contrast_name,
        "mode": a.mode, "primary_alpha": a.alpha, "primary_rate": a.rate,
        "hysteresis": a.hysteresis, "seconds": round(time.time() - t0, 1),
        "primary_arm": "reactive",
        "primary_result": {"delta_acc": prim["delta_acc"], "p": prim["vs_base"]["p"]},
    }
    out = Path(a.out or f"/content/steer_{a.model.split('/')[-1]}.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out}", flush=True)
    if a.hf_repo:
        try:
            from huggingface_hub import HfApi
            from probe_pipeline.artifacts_io import resolve_hf_token
            HfApi(token=resolve_hf_token(required=True)).upload_file(
                path_or_fileobj=str(out), path_in_repo=f"steering/{out.name}",
                repo_id=a.hf_repo, repo_type="dataset",
            )
            print(f"pushed -> {a.hf_repo}/steering/{out.name}", flush=True)
        except Exception as exc:
            print(f"[hf] push failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
