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
    resp, nnew, stats, p_all = [], [], [], []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i:i + batch_size])
        hook = CascadeSteeringHook(model, **hook_kwargs)
        with hook:
            r, k = generate_batched(model, tok, chunk, greedy=greedy, seed=seed + i,
                                    max_new_tokens=max_new_tokens, batch_size=batch_size)
        resp.extend(r)
        nnew.extend(k)
        s = hook.stats.to_dict()
        s.update(hook.stats.trimmed(k))
        stats.append(s)
        if hook.stats.p_steer:
            ps = np.stack(hook.stats.p_steer)                     # (steps, B)
            keep = np.arange(ps.shape[0])[:, None] < np.asarray(k)[None, :]
            p_all.append(ps[keep])
        hook.reset()

    agg = {
        "n_positions": sum(s["n_positions"] for s in stats),
        "n_fired": sum(s["n_fired"] for s in stats),
        "n_held": sum(s["n_held"] for s in stats),
        "energy": sum(s["energy"] for s in stats),
    }
    agg["n_positions_trimmed"] = sum(s["n_positions_trimmed"] for s in stats)
    agg["duty_cycle"] = (agg["n_fired"] + agg["n_held"]) / max(1, agg["n_positions"])
    agg["fire_rate"] = agg["n_fired"] / max(1, agg["n_positions"])
    # Weighted by each batch's real generated positions, so a short final
    # batch does not count as much as a full one.
    agg["duty_cycle_trimmed"] = sum(
        s["duty_cycle_trimmed"] * s["n_positions_trimmed"] for s in stats
    ) / max(1, agg["n_positions_trimmed"])
    agg["p_steer"] = np.concatenate(p_all) if p_all else np.zeros(0)
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
    ap.add_argument("--mode", default="additive_normalized")
    ap.add_argument("--alpha", type=float, default=0.05, help="primary coefficient")
    ap.add_argument("--rate", type=float, default=0.05, help="primary fire rate")
    ap.add_argument("--alpha-grid", default="0.02,0.10")
    ap.add_argument("--rate-grid", default="0.02,0.10")
    ap.add_argument("--hysteresis", type=int, default=0)
    ap.add_argument("--stage", default="all", choices=["primary", "all"])
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
    if a.band:
        idx = json.loads(Path(a.band).read_text())["indices"][: a.n]
        ds = ds.select(idx)
        print(f"[band] {len(idx)} in-band questions", flush=True)
    else:
        ds = ds.select(range(min(a.n, len(ds))))
    questions = [r["question"] for r in ds]
    golds = [extract_gold(r["answer"]) for r in ds]
    prompts = build_prompts(questions, tok)

    print(f"[load] {a.model} layer {layer} | {len(prompts)} questions", flush=True)
    dtype = torch.float32 if a.device in ("cpu", "mps") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device).eval()

    v_signed = npz["v_signed"]
    v_unsigned = npz["v_unsigned"]
    rng = np.random.default_rng(7)
    v_random = rng.normal(size=v_signed.shape).astype(np.float32)

    base_kw = dict(layer=layer, gate_w=npz["gate_w"], gate_b=float(npz["gate_b"]),
                   sign_w=npz["sign_w"], sign_b=float(npz["sign_b"]),
                   mode=a.mode, hysteresis=a.hysteresis)
    run = lambda **kw: run_arm(model, tok, prompts, golds, max_new_tokens=a.max_new_tokens,  # noqa: E731
                               batch_size=a.batch_size, **kw)
    results, t0 = {}, time.time()

    # -- base, and the observe pass that calibrates every threshold ---------
    print("[1] base + observe", flush=True)
    obs_arm, obs = run(name="base", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=0.0, gate_mode=MODE_OBSERVE))
    base_mask = obs_arm.correct_mask
    results["base"] = obs_arm.as_dict()
    p = obs["p_steer"]
    print(f"    base {obs_arm.accuracy:.4f} | {len(p)} decode positions | "
          f"p_steer mean {p.mean():.4f} p95 {np.quantile(p, 0.95):.4f} "
          f"max {p.max():.4f}", flush=True)

    def thresh(rate: float) -> float:
        return float(np.quantile(p, 1.0 - rate))

    # -- identity: the hook plumbing itself must be a no-op ----------------
    small = prompts[: min(32, len(prompts))]
    ident, _ = run_arm(model, tok, small, golds[: len(small)], name="identity",
                       max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
                       hook_kwargs=dict(base_kw, vector=v_signed, coef=0.0,
                                        gate_mode=MODE_ALWAYS_ON))
    ok = ident.correct_mask == base_mask[: len(small)]
    results["identity_check"] = {"passed": bool(ok), "n": len(small)}
    print(f"[2] identity (coef=0, always on): {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("    the hook changes generation at coef=0; stopping", flush=True)
        Path(a.out or "/content/steer.json").write_text(json.dumps(results, indent=2))
        return

    def record(nm, arm, st, primary=False):
        d = arm.as_dict()
        d.update({k: v for k, v in st.items() if k != "p_steer"})
        d["vs_base"] = mcnemar(base_mask, arm.correct_mask)
        d["delta_acc"] = arm.accuracy - obs_arm.accuracy
        d["primary"] = primary
        results[nm] = d
        print(f"    {nm:<24} acc {arm.accuracy:.4f} ({d['delta_acc']:+.4f})  "
              f"duty {st.get('duty_cycle_trimmed', 0):.3f}  energy {st.get('energy', 0):.0f}  "
              f"net {d['vs_base']['net']:+d}  p {d['vs_base']['p']:.3f}", flush=True)
        return d

    # -- primary -----------------------------------------------------------
    print(f"[3] primary: reactive rate={a.rate} alpha={a.alpha}", flush=True)
    arm, st = run(name="reactive", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
    prim = record("reactive", arm, st, primary=True)
    duty = max(st["duty_cycle_trimmed"], 1e-6)

    # -- matched controls --------------------------------------------------
    print("[4] matched controls", flush=True)
    arm, st = run(name="always_on", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=a.alpha * duty, gate_mode=MODE_ALWAYS_ON))
    record("always_on_matched_energy", arm, st)

    arm, st = run(name="random_placement", hook_kwargs=dict(
        base_kw, vector=v_signed, coef=a.alpha, gate_mode=MODE_PATTERN,
        pattern=build_random_pattern(a.max_new_tokens, a.batch_size, duty)))
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
    arm, st = run(name="unsigned_dir", hook_kwargs=dict(
        base_kw, vector=v_unsigned, coef=a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
    record("unsigned_direction", arm, st)

    arm, st = run(name="random_dir", hook_kwargs=dict(
        base_kw, vector=v_random, coef=a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
    record("random_direction", arm, st)

    # Gate on P(pivotal) alone: the cascade's signed half removed, so the
    # same direction fires on pivots regardless of whether they help.
    ung = dict(base_kw)
    ung["sign_w"] = None
    arm, st = run(name="unsigned_gate", hook_kwargs=dict(
        ung, vector=v_signed, coef=a.alpha,
        gate_mode=MODE_REACTIVE, threshold=thresh(a.rate)))
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

    _finish(a, results, t0, prim)


def _finish(a, results, t0, prim) -> None:
    results["_meta"] = {
        "model": a.model, "probes": a.probes, "band": a.band,
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
