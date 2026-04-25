"""Short smoke test for the Colab notebook pipelines.

Exercises the distinct code paths behind each notebook in ``notebooks/`` on a
trivially small workload so we can verify correctness in ~1-2 minutes per
variant instead of the full Colab run. Pick a variant with ``--variant``:

    main                 notebooks/steering_gsm8k_colab.ipynb
    random_control       notebooks/steering_random_control.ipynb
    greedy               notebooks/steering_greedy_decode.ipynb
    layer_sweep          notebooks/steering_layer_sweep.ipynb
    additive             notebooks/steering_additive_multilayer.ipynb
    codelion             notebooks/codelion_steering_vectors.ipynb
    probe_weights        notebooks/steering_probe_weights.ipynb
    all                  run every variant sequentially

Usage:
    python scripts/smoke_colab_notebook.py --variant main --max-examples 2
    python scripts/smoke_colab_notebook.py --variant all --max-examples 2

Successful runs write to ``artifacts/smoke/<variant>/``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("main", "random_control", "greedy", "layer_sweep", "additive", "codelion",
            "probe_weights")


# --------------------------------------------------------------------------- shared helpers
def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_gsm8k_answer(text: str):
    m = re.search(r"####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def is_correct(pred, gt):
    if pred is None or gt is None:
        return False
    p = pred.strip().replace(",", "")
    g = gt.strip().replace(",", "")
    try:
        return float(p) == float(g)
    except ValueError:
        return p == g


def compute_metrics(results):
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "f1": 0.0, "parse_rate": 0.0, "correct": 0, "n": 0}
    correct = sum(1 for r in results if r["correct"])
    parsed = sum(1 for r in results if r["predicted"] is not None)
    tp = correct
    fp = parsed - correct
    fn = n - correct
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": correct / n, "f1": f1, "parse_rate": parsed / n,
            "correct": correct, "n": n}


def _get_decoder_layer(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError(f"Cannot find decoder layers in {type(model)}")


def _make_hook(vector: torch.Tensor, strength: float):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        v = vector.to(hidden.device).to(hidden.dtype)
        if v.dim() == 1:
            v = v.view(1, 1, -1)
        delta = strength * v
        if isinstance(output, tuple):
            return (hidden + delta,) + output[1:]
        return hidden + delta
    return hook


def _register_steering(model, layer, vec, strength):
    return [_get_decoder_layer(model, layer).register_forward_hook(_make_hook(vec, strength))]


def _register_additive(model, injections):
    handles = []
    for layer, vec, strength in injections:
        handles.append(
            _get_decoder_layer(model, layer).register_forward_hook(_make_hook(vec, strength))
        )
    return handles


def _make_projection_hook(unit_vec: torch.Tensor, alpha: float):
    """h_new = h + (alpha - 1) * (h . v_hat) * v_hat. Scales only the component
    along the probe direction; alpha=1 is identity, alpha=0 ablates."""
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        vh = unit_vec.to(hidden.device).to(hidden.dtype)
        if vh.dim() == 1:
            vh_b = vh.view(1, 1, -1)
        else:
            vh_b = vh
        proj_scalar = (hidden * vh_b).sum(dim=-1, keepdim=True)
        proj_vec = proj_scalar * vh_b
        delta = (alpha - 1.0) * proj_vec
        if isinstance(output, tuple):
            return (hidden + delta,) + output[1:]
        return hidden + delta
    return hook


def _register_projection(model, layer: int, unit_vec: torch.Tensor, alpha: float):
    return [_get_decoder_layer(model, layer).register_forward_hook(
        _make_projection_hook(unit_vec, alpha)
    )]


def _remove(handles):
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- device / model / probe


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_probe(probe_dir: Path, layer: int):
    cfg_path = probe_dir / f"steering_layer{layer}.json"
    vec_path = probe_dir / f"steering_layer{layer}_vector.npy"
    if not (cfg_path.exists() and vec_path.exists()):
        raise FileNotFoundError(f"probe files missing at {cfg_path}, {vec_path}")
    cfg = json.loads(cfg_path.read_text())
    vec = np.load(vec_path).astype(np.float32)
    return cfg, vec, cfg_path, vec_path


def load_model(model_name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def load_examples(max_examples, seed):
    from datasets import load_dataset
    ds_full = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(ds_full), size=min(max_examples, len(ds_full)), replace=False).tolist())
    subset = ds_full.select(idxs)
    return [
        {"question": r["question"], "ground_truth": extract_gsm8k_answer(r["answer"])}
        for r in subset
    ]


# --------------------------------------------------------------------------- generator closure


def make_generator(model, tokenizer, device, max_new_tokens, do_sample=True,
                   temperature=0.6, top_p=0.9):
    def generate_once(prompt: str) -> str:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=max_new_tokens,
                              pad_token_id=tokenizer.pad_token_id)
            if do_sample:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(do_sample=False)
            out = model.generate(**enc, **gen_kwargs)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    return generate_once


def run_once(generate_once, examples, register_fn, label, seed):
    _seed(seed)
    handles = register_fn() if register_fn is not None else []
    results = []
    try:
        for i, ex in enumerate(examples):
            prompt = f"Question: {ex['question']}\n\nLet's think step by step.\n\n"
            text = generate_once(prompt)
            pred = extract_gsm8k_answer(text)
            ok = is_correct(pred, ex["ground_truth"])
            results.append({"idx": i, "question": ex["question"],
                            "ground_truth": ex["ground_truth"],
                            "predicted": pred, "correct": ok,
                            "full_output": text[-800:]})
            print(f"[smoke]   {label} #{i}: gt={ex['ground_truth']} pred={pred} correct={ok}")
    finally:
        _remove(handles)
    return results, compute_metrics(results)


def save_run(out_dir: Path, label, results, metrics, extra=None):
    (out_dir / f"{label}_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / f"{label}_generations.txt").write_text(
        "\n\n====\n\n".join(
            f"[{r['idx']}] gt={r['ground_truth']} pred={r['predicted']} correct={r['correct']}\n{r['full_output']}"
            for r in results
        )
    )
    state = {"label": label, "metrics": metrics}
    if extra:
        state.update(extra)
    (out_dir / f"{label}_state.json").write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- variant runners


def run_variant_main(args, out_dir, device, dtype, probe_dir):
    cfg, vec, _, vec_path = load_probe(probe_dir, args.layer)
    vec_tensor = torch.from_numpy(vec).to(torch.float32)
    tokenizer, model = load_model(args.model, device, dtype)
    assert vec.shape[0] == model.config.hidden_size
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens, do_sample=True)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met)

    label = f"steered_{args.factor}"
    strength = args.factor - 1.0
    res, met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, vec_tensor, strength),
        label, args.seed,
    )
    save_run(out_dir, label, res, met, extra={"factor": args.factor, "strength": strength})
    print(f"[smoke/main] base={base_met['accuracy']:.3f} {label}={met['accuracy']:.3f}")


def run_variant_random_control(args, out_dir, device, dtype, probe_dir):
    cfg, vec, _, _ = load_probe(probe_dir, args.layer)
    piv_tensor = torch.from_numpy(vec).to(torch.float32)
    piv_norm = float(piv_tensor.norm().item())

    g = torch.Generator().manual_seed(101)
    r = torch.randn(vec.shape[0], generator=g, dtype=torch.float32)
    r = r * (piv_norm / r.norm().item())

    tokenizer, model = load_model(args.model, device, dtype)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met)

    strength = args.factor - 1.0
    piv_res, piv_met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, piv_tensor, strength),
        "pivotal", args.seed,
    )
    save_run(out_dir, "pivotal", piv_res, piv_met)
    rand_res, rand_met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, r, strength),
        "random101", args.seed,
    )
    save_run(out_dir, "random101", rand_res, rand_met)
    print(f"[smoke/random_control] base={base_met['accuracy']:.3f} "
          f"pivotal={piv_met['accuracy']:.3f} random={rand_met['accuracy']:.3f}")


def run_variant_greedy(args, out_dir, device, dtype, probe_dir):
    cfg, vec, _, _ = load_probe(probe_dir, args.layer)
    vec_tensor = torch.from_numpy(vec).to(torch.float32)
    tokenizer, model = load_model(args.model, device, dtype)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens, do_sample=False)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met, extra={"decoding": "greedy"})

    strength = args.factor - 1.0
    res, met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, vec_tensor, strength),
        f"steered_{args.factor}", args.seed,
    )
    save_run(out_dir, f"steered_{args.factor}", res, met, extra={"decoding": "greedy"})
    print(f"[smoke/greedy] base={base_met['accuracy']:.3f} steered={met['accuracy']:.3f}")


def run_variant_layer_sweep(args, out_dir, device, dtype, probe_dir):
    # Only use layers that are actually present locally (skips layer 8 if missing).
    layers = []
    for L in (8, 14, 16):
        if (probe_dir / f"steering_layer{L}.json").exists():
            layers.append(L)
    if not layers:
        raise RuntimeError("no probe layers available for smoke test")
    probes = {L: load_probe(probe_dir, L) for L in layers}

    tokenizer, model = load_model(args.model, device, dtype)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met)

    for L in layers:
        cfg, vec, _, _ = probes[L]
        vec_tensor = torch.from_numpy(vec).to(torch.float32)
        label = f"L{L}_f{args.factor}"
        strength = args.factor - 1.0
        res, met = run_once(
            gen, examples,
            lambda l=L, v=vec_tensor, s=strength: _register_steering(model, l, v, s),
            label, args.seed,
        )
        save_run(out_dir, label, res, met, extra={"layer": L, "factor": args.factor})
        print(f"[smoke/layer_sweep] L{L}: {met['accuracy']:.3f}")


def run_variant_additive(args, out_dir, device, dtype, probe_dir):
    present = [L for L in (14, 16, 20) if (probe_dir / f"steering_layer{L}.json").exists()]
    assert len(present) >= 2, "need at least 2 probe layers for additive smoke"
    vecs = {}
    for L in present:
        cfg, v, _, _ = load_probe(probe_dir, L)
        vecs[L] = torch.from_numpy(v).to(torch.float32)

    tokenizer, model = load_model(args.model, device, dtype)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met)

    spec = [(present[0], vecs[present[0]], 0.2), (present[1], vecs[present[1]], 0.2)]
    res, met = run_once(
        gen, examples,
        lambda: _register_additive(model, spec),
        "multi_layer_additive", args.seed,
    )
    save_run(out_dir, "multi_layer_additive", res, met,
             extra={"injections": [(L, float(s)) for L, _, s in spec]})
    print(f"[smoke/additive] base={base_met['accuracy']:.3f} additive={met['accuracy']:.3f}")


def run_variant_codelion(args, out_dir, device, dtype, probe_dir):
    from datasets import load_dataset
    ds = load_dataset("codelion/Qwen3-0.6B-pts-steering-vectors", split="train[:32]")
    df = ds.to_pandas()
    print(f"[smoke/codelion] loaded {len(df)} rows, cols={list(df.columns)[:8]}...")

    df["steering_vector"] = df["steering_vector"].map(lambda x: np.asarray(x, dtype=np.float32))
    df["abs_prob_delta"] = df["prob_delta"].abs()

    V = np.stack(df["steering_vector"].values)
    mean_vec = V.mean(axis=0).astype(np.float32)
    top1_vec = V[int(df["abs_prob_delta"].idxmax())].astype(np.float32)
    top1_vec = top1_vec * (np.linalg.norm(mean_vec) / max(1e-8, float(np.linalg.norm(top1_vec))))

    mean_tensor = torch.from_numpy(mean_vec).to(torch.float32)
    top1_tensor = torch.from_numpy(top1_vec).to(torch.float32)

    tokenizer, model = load_model(args.model, device, dtype)
    assert V.shape[1] == model.config.hidden_size, (
        f"codelion hidden_dim {V.shape[1]} != model hidden_size {model.config.hidden_size}"
    )

    # Codelion PTS extracts at layers 19/23/27 on Qwen3-0.6B; optillm autothink's default is 19.
    codelion_layer = 19 if args.layer == 14 else args.layer

    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met)

    strength = args.factor - 1.0
    for label, tensor in (("mean", mean_tensor), ("top1", top1_tensor)):
        res, met = run_once(
            gen, examples,
            lambda t=tensor, s=strength, l=codelion_layer: _register_steering(model, l, t, s),
            label, args.seed,
        )
        save_run(out_dir, label, res, met, extra={"arm": label, "factor": args.factor, "layer": codelion_layer})
        print(f"[smoke/codelion] {label} @ layer {codelion_layer}: {met['accuracy']:.3f}")


def _load_probe_weights(repo_root: Path, layer: int) -> np.ndarray:
    """Load probe-weight vector saved by sklearn training (one row per class)."""
    p = repo_root / f"artifacts/cached3/sklearn/analysis_data/layer_{layer}/probe_weights.npy"
    if not p.exists():
        raise FileNotFoundError(f"probe_weights file missing at {p}")
    return np.load(p).astype(np.float32).reshape(-1)


def run_variant_probe_weights(args, out_dir, device, dtype, probe_dir):
    """Smoke test for notebooks/steering_probe_weights.ipynb. Exercises BOTH
    additive (h + alpha*w) and projection-scaling (h + (alpha-1)*proj_w(h)) on
    n examples each."""
    w = _load_probe_weights(REPO_ROOT, args.layer)
    w_norm = float(np.linalg.norm(w))
    w_tensor = torch.from_numpy(w).to(torch.float32)
    v_hat = w / max(1e-12, w_norm)
    v_hat_tensor = torch.from_numpy(v_hat).to(torch.float32)

    # Sanity: cosine vs centroid-diff direction (if available locally).
    cd_path = probe_dir / f"steering_layer{args.layer}_vector.npy"
    if cd_path.exists():
        cd = np.load(cd_path).astype(np.float32).reshape(-1)
        cos = float(np.dot(w, cd) / max(1e-12, np.linalg.norm(w) * np.linalg.norm(cd)))
        print(f"[smoke/probe_weights] cos(probe_weights, centroid_diff) = {cos:+.4f}")

    tokenizer, model = load_model(args.model, device, dtype)
    assert w.shape[0] == model.config.hidden_size, (
        f"probe-weights dim {w.shape[0]} != hidden_size {model.config.hidden_size}"
    )

    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met)

    add_alpha = args.factor  # default 1.4
    add_label = f"addw_a{add_alpha}"
    add_res, add_met = run_once(
        gen, examples,
        lambda a=add_alpha: _register_steering(model, args.layer, w_tensor, a),
        add_label, args.seed,
    )
    save_run(out_dir, add_label, add_res, add_met, extra={
        "arm": "additive", "hook_type": "additive",
        "alpha": add_alpha, "factor": add_alpha,
        "layer": args.layer, "vector_type": "probe_weights",
        "vector_norm": w_norm,
    })

    for proj_alpha in (0.0, 2.0):
        proj_label = f"projw_a{proj_alpha}"
        proj_res, proj_met = run_once(
            gen, examples,
            lambda a=proj_alpha: _register_projection(model, args.layer, v_hat_tensor, a),
            proj_label, args.seed,
        )
        save_run(out_dir, proj_label, proj_res, proj_met, extra={
            "arm": "projection", "hook_type": "projection",
            "alpha": proj_alpha,
            "layer": args.layer, "vector_type": "probe_weights",
            "vector_norm": w_norm,
        })

    # Verify state files contain hook_type and vector_type.
    for label in ("base", add_label, "projw_a0.0", "projw_a2.0"):
        state = json.loads((out_dir / f"{label}_state.json").read_text())
        assert "metrics" in state, f"{label} state missing metrics"
        if label != "base":
            assert state.get("hook_type") in {"additive", "projection"}, label
            assert state.get("vector_type") == "probe_weights", label
    print(f"[smoke/probe_weights] base={base_met['accuracy']:.3f} "
          f"{add_label}={add_met['accuracy']:.3f}")


RUNNERS = {
    "main": run_variant_main,
    "random_control": run_variant_random_control,
    "greedy": run_variant_greedy,
    "layer_sweep": run_variant_layer_sweep,
    "additive": run_variant_additive,
    "codelion": run_variant_codelion,
    "probe_weights": run_variant_probe_weights,
}


# --------------------------------------------------------------------------- entrypoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="main", choices=(*VARIANTS, "all"))
    ap.add_argument("--max-examples", type=int, default=2)
    ap.add_argument("--factor", type=float, default=1.4)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--probe-dir",
        default=str(REPO_ROOT / "artifacts/cached3/sklearn/steering_configs"),
    )
    ap.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "artifacts/smoke"),
    )
    args = ap.parse_args()

    device, dtype = pick_device()
    print(f"[smoke] device={device} dtype={dtype}")

    # Parser inline sanity check (runs regardless of variant).
    assert extract_gsm8k_answer("#### 42") == "42"
    assert is_correct("3.0", "3") is True
    assert compute_metrics([])["accuracy"] == 0.0
    print("[smoke] parser sanity OK")

    variants = VARIANTS if args.variant == "all" else (args.variant,)
    probe_dir = Path(args.probe_dir)
    out_root = Path(args.out_root)

    for v in variants:
        t0 = time.time()
        out_dir = out_root / v
        out_dir.mkdir(parents=True, exist_ok=True)
        _seed(args.seed)
        print(f"\n[smoke] ==== variant: {v} ====")
        try:
            RUNNERS[v](args, out_dir, device, dtype, probe_dir)
        except Exception as exc:
            print(f"[smoke] variant {v} FAILED: {exc}")
            return 2
        print(f"[smoke] variant {v} OK in {time.time() - t0:.1f}s -> {out_dir}")

    print("\n[smoke] all done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
