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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VARIANTS = ("main", "random_control", "greedy", "layer_sweep", "additive", "codelion",
            "probe_weights", "reactive", "nie")


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
    """Off-by-one-fixed accessor (Issue #2 in docs/issues.md).

    Returns the module whose forward-hook output IS
    ``outputs.hidden_states[layer_idx]``. For Qwen3 / LLaMA-style models that
    means ``layers[layer_idx - 1]`` for ``layer_idx >= 1`` and ``embed_tokens``
    for ``layer_idx == 0``.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if layer_idx == 0:
            return model.model.embed_tokens
        return model.model.layers[layer_idx - 1]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        if layer_idx == 0:
            return model.transformer.wte
        return model.transformer.h[layer_idx - 1]
    raise AttributeError(f"Cannot find decoder layers in {type(model)}")


def _make_hook(vector: torch.Tensor, coef: float, mode: str = "additive_raw"):
    """Three-mode steering hook (Issue #7 in docs/issues.md).

    Modes:
        additive_raw         h <- h + coef * v
        additive_normalized  h <- h + coef * ||h_pos|| * v_hat
        projection           h <- h + (coef - 1) * (h . v_hat) * v_hat
    """
    if mode not in {"additive_raw", "additive_normalized", "projection"}:
        raise ValueError(f"unknown mode {mode!r}")

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        v = vector.to(hidden.device).to(hidden.dtype)
        if v.dim() == 1:
            v_full = v.view(1, 1, -1)
        else:
            v_full = v
        if mode == "additive_raw":
            delta = coef * v_full
        elif mode == "additive_normalized":
            v_hat = v / (v.norm() + 1e-8)
            v_hat_b = v_hat.view(1, 1, -1)
            h_norm = hidden.norm(dim=-1, keepdim=True)
            delta = coef * h_norm * v_hat_b
        else:  # projection
            v_hat = v / (v.norm() + 1e-8)
            v_hat_b = v_hat.view(1, 1, -1)
            proj = (hidden * v_hat_b).sum(dim=-1, keepdim=True)
            delta = (coef - 1.0) * proj * v_hat_b
        if isinstance(output, tuple):
            return (hidden + delta,) + output[1:]
        return hidden + delta
    return hook


def _register_steering(model, layer, vec, coef, mode: str = "additive_raw"):
    return [_get_decoder_layer(model, layer).register_forward_hook(
        _make_hook(vec, coef, mode=mode)
    )]


def _register_additive(model, injections, mode: str = "additive_raw"):
    handles = []
    for layer, vec, coef in injections:
        handles.append(
            _get_decoder_layer(model, layer).register_forward_hook(
                _make_hook(vec, coef, mode=mode)
            )
        )
    return handles


def _make_projection_hook(unit_vec: torch.Tensor, alpha: float):
    """Legacy alias kept for callers that still pass an already-unit-normed
    vector. Internally delegates to ``_make_hook(mode='projection')``."""
    return _make_hook(unit_vec, alpha, mode="projection")


def _register_projection(model, layer: int, unit_vec: torch.Tensor, alpha: float):
    return [_get_decoder_layer(model, layer).register_forward_hook(
        _make_projection_hook(unit_vec, alpha)
    )]


def _remove(handles):
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- provenance + convention check


def _hook_target_layer_idx(layer: int) -> int:
    """The actual HF index of the module the hook attaches to (post-Issue-#2)."""
    return -1 if layer == 0 else layer - 1  # -1 sentinel = embed_tokens


def prov(
    layer: int,
    mode: str,
    vector_source: str,
    convention_check_passed: bool | None = None,
) -> dict:
    """Build a steering-provenance dict to merge into every run's state.

    Required fields (per docs/issues.md alignment):
        hook_target_layer_idx, mode, vector_source, convention_check_passed.
    """
    return {
        "layer": layer,
        "hook_target_layer_idx": _hook_target_layer_idx(layer),
        "mode": mode,
        "vector_source": vector_source,
        "convention_check_passed": (
            None if convention_check_passed is None else bool(convention_check_passed)
        ),
    }


def run_convention_check(model, layer: int) -> bool:
    """Verify hook on ``_get_decoder_layer(model, layer)`` captures the same
    tensor as ``outputs.hidden_states[layer]`` for a tiny dummy forward.

    Returns True on success. Returns False (rather than raising) so smoke can
    surface the result via state file rather than crashing -- the
    ``test_layer_alignment.py`` pytest is the gating regression test.
    """
    captured = {}

    def cap(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["pre"] = h.detach()
        return output

    target = _get_decoder_layer(model, layer)
    handle = target.register_forward_hook(cap)
    try:
        device = next(model.parameters()).device
        ids = torch.tensor([[1, 2, 3]], device=device)
        with torch.no_grad():
            outs = model(input_ids=ids, output_hidden_states=True)
    finally:
        handle.remove()

    expected = outs.hidden_states[layer]
    if "pre" not in captured:
        return False
    same_shape = captured["pre"].shape == expected.shape
    if not same_shape:
        return False
    return bool(torch.allclose(captured["pre"], expected, atol=1e-4, rtol=1e-3))


# --------------------------------------------------------------------------- device / model / probe


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_probe(probe_dir: Path, layer: int, *, prefer_train_caa: bool = True):
    """Load steering bundle from ``probe_dir``.

    Closes Issue #5: when a TRAIN-derived CAA is available
    (``steering_layer{L}_vector_train.npy``) it is preferred over the legacy
    val-derived ``_vector.npy``. The corresponding ``_train.json`` (if
    present) is used to refresh ``vector_norm`` so downstream consumers see
    the actual norm of the returned tensor.
    """
    cfg_path = probe_dir / f"steering_layer{layer}.json"
    train_vec_path = probe_dir / f"steering_layer{layer}_vector_train.npy"
    val_vec_path = probe_dir / f"steering_layer{layer}_vector.npy"
    if not cfg_path.exists():
        raise FileNotFoundError(f"probe config missing at {cfg_path}")
    if prefer_train_caa and train_vec_path.exists():
        vec_path = train_vec_path
        vec_source = "centroid_diff_train"
    elif val_vec_path.exists():
        vec_path = val_vec_path
        vec_source = "centroid_diff_val"
    else:
        raise FileNotFoundError(
            f"no steering vector found for layer {layer}: tried "
            f"{train_vec_path}, {val_vec_path}"
        )
    cfg = json.loads(cfg_path.read_text())
    vec = np.load(vec_path).astype(np.float32).reshape(-1)
    cfg["vector_path_resolved"] = str(vec_path)
    cfg["vector_source"] = vec_source
    cfg["vector_norm"] = float(np.linalg.norm(vec))
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
    conv_ok = run_convention_check(model, args.layer)
    assert vec.shape[0] == model.config.hidden_size
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens, do_sample=True)

    mode = "additive_normalized"  # canonical post-Issue-#7 convention.
    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(args.layer, "none", "n/a", conv_ok))

    label = f"steered_{args.factor}"
    coef = args.factor - 1.0
    res, met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, vec_tensor, coef, mode=mode),
        label, args.seed,
    )
    save_run(out_dir, label, res, met,
             extra={**prov(args.layer, mode, cfg.get("vector_source", "caa_val"), conv_ok),
                    "factor": args.factor, "strength": coef})
    print(f"[smoke/main] base={base_met['accuracy']:.3f} {label}={met['accuracy']:.3f} "
          f"(conv_check={conv_ok})")


def run_variant_random_control(args, out_dir, device, dtype, probe_dir):
    cfg, vec, _, _ = load_probe(probe_dir, args.layer)
    piv_tensor = torch.from_numpy(vec).to(torch.float32)
    piv_norm = float(piv_tensor.norm().item())

    g = torch.Generator().manual_seed(101)
    r = torch.randn(vec.shape[0], generator=g, dtype=torch.float32)
    r = r * (piv_norm / r.norm().item())

    tokenizer, model = load_model(args.model, device, dtype)
    conv_ok = run_convention_check(model, args.layer)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    mode = "additive_normalized"
    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(args.layer, "none", "n/a", conv_ok))

    coef = args.factor - 1.0
    piv_res, piv_met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, piv_tensor, coef, mode=mode),
        "pivotal", args.seed,
    )
    save_run(out_dir, "pivotal", piv_res, piv_met,
             extra=prov(args.layer, mode, cfg.get("vector_source", "caa_val"), conv_ok))
    rand_res, rand_met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, r, coef, mode=mode),
        "random101", args.seed,
    )
    save_run(out_dir, "random101", rand_res, rand_met,
             extra=prov(args.layer, mode, "random101", conv_ok))
    print(f"[smoke/random_control] base={base_met['accuracy']:.3f} "
          f"pivotal={piv_met['accuracy']:.3f} random={rand_met['accuracy']:.3f}")


def run_variant_greedy(args, out_dir, device, dtype, probe_dir):
    cfg, vec, _, _ = load_probe(probe_dir, args.layer)
    vec_tensor = torch.from_numpy(vec).to(torch.float32)
    tokenizer, model = load_model(args.model, device, dtype)
    conv_ok = run_convention_check(model, args.layer)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens, do_sample=False)

    mode = "additive_normalized"
    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra={**prov(args.layer, "none", "n/a", conv_ok), "decoding": "greedy"})

    coef = args.factor - 1.0
    res, met = run_once(
        gen, examples,
        lambda: _register_steering(model, args.layer, vec_tensor, coef, mode=mode),
        f"steered_{args.factor}", args.seed,
    )
    save_run(out_dir, f"steered_{args.factor}", res, met,
             extra={**prov(args.layer, mode, cfg.get("vector_source", "caa_val"), conv_ok), "decoding": "greedy"})
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
    conv_oks = {L: run_convention_check(model, L) for L in layers}
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    mode = "additive_normalized"
    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(layers[0], "none", "n/a", all(conv_oks.values())))

    for L in layers:
        cfg, vec, _, _ = probes[L]
        vec_tensor = torch.from_numpy(vec).to(torch.float32)
        label = f"L{L}_f{args.factor}"
        coef = args.factor - 1.0
        res, met = run_once(
            gen, examples,
            lambda l=L, v=vec_tensor, s=coef: _register_steering(model, l, v, s, mode=mode),
            label, args.seed,
        )
        save_run(out_dir, label, res, met,
                 extra={**prov(L, mode, cfg.get("vector_source", "caa_val"), conv_oks[L]),
                        "factor": args.factor})
        print(f"[smoke/layer_sweep] L{L}: {met['accuracy']:.3f}")


def run_variant_additive(args, out_dir, device, dtype, probe_dir):
    present = [L for L in (14, 16, 20) if (probe_dir / f"steering_layer{L}.json").exists()]
    assert len(present) >= 2, "need at least 2 probe layers for additive smoke"
    vecs = {}
    for L in present:
        cfg, v, _, _ = load_probe(probe_dir, L)
        vecs[L] = torch.from_numpy(v).to(torch.float32)

    tokenizer, model = load_model(args.model, device, dtype)
    conv_oks = {L: run_convention_check(model, L) for L in present}
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    mode = "additive_normalized"
    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(present[0], "none", "n/a", all(conv_oks.values())))

    spec = [(present[0], vecs[present[0]], 0.2), (present[1], vecs[present[1]], 0.2)]
    res, met = run_once(
        gen, examples,
        lambda: _register_additive(model, spec, mode=mode),
        "multi_layer_additive", args.seed,
    )
    save_run(out_dir, "multi_layer_additive", res, met,
             extra={**prov(present[0], mode,
                           cfg.get("vector_source", "caa_val") + "_multilayer",
                           all(conv_oks.values())),
                    "injections": [(L, float(s)) for L, _, s in spec]})
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
    conv_ok = run_convention_check(model, codelion_layer)

    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    mode = "additive_normalized"
    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(codelion_layer, "none", "n/a", conv_ok))

    coef = args.factor - 1.0
    for label, tensor, vsource in (("mean", mean_tensor, "codelion_mean"),
                                   ("top1", top1_tensor, "codelion_top1")):
        res, met = run_once(
            gen, examples,
            lambda t=tensor, s=coef, l=codelion_layer: _register_steering(model, l, t, s, mode=mode),
            label, args.seed,
        )
        save_run(out_dir, label, res, met,
                 extra={**prov(codelion_layer, mode, vsource, conv_ok),
                        "arm": label, "factor": args.factor})
        print(f"[smoke/codelion] {label} @ layer {codelion_layer}: {met['accuracy']:.3f}")


def _load_probe_weights(repo_root: Path, layer: int) -> np.ndarray:
    """Load probe-weight vector. Accepts either the tracked steering_configs/
    layout (preferred, also what GitHub serves) or the raw analysis_data/
    output of probe_pipeline."""
    candidates = [
        repo_root / "artifacts/cached3/sklearn/steering_configs"
        / f"steering_layer{layer}_probe_weights.npy",
        repo_root / "artifacts/cached3/sklearn/analysis_data"
        / f"layer_{layer}" / "probe_weights.npy",
    ]
    for p in candidates:
        if p.exists():
            return np.load(p).astype(np.float32).reshape(-1)
    raise FileNotFoundError(
        f"probe_weights file missing; tried {[str(c) for c in candidates]}"
    )


def run_variant_probe_weights(args, out_dir, device, dtype, probe_dir):
    """Smoke test for notebooks/steering_probe_weights.ipynb. Exercises all
    three modes (additive_normalized, projection, projection alpha=0 patching)
    on the probe-weight direction."""
    w = _load_probe_weights(REPO_ROOT, args.layer)
    w_norm = float(np.linalg.norm(w))
    w_tensor = torch.from_numpy(w).to(torch.float32)

    # Sanity: cosine vs centroid-diff direction (if available locally).
    cd_path = probe_dir / f"steering_layer{args.layer}_vector_train.npy"
    if not cd_path.exists():
        cd_path = probe_dir / f"steering_layer{args.layer}_vector.npy"
    if cd_path.exists():
        cd = np.load(cd_path).astype(np.float32).reshape(-1)
        cos = float(np.dot(w, cd) / max(1e-12, np.linalg.norm(w) * np.linalg.norm(cd)))
        print(f"[smoke/probe_weights] cos(probe_weights, centroid_diff) = {cos:+.4f}")

    tokenizer, model = load_model(args.model, device, dtype)
    conv_ok = run_convention_check(model, args.layer)
    assert w.shape[0] == model.config.hidden_size, (
        f"probe-weights dim {w.shape[0]} != hidden_size {model.config.hidden_size}"
    )

    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(args.layer, "none", "n/a", conv_ok))

    # Additive (normalized) arm.
    add_coef = args.factor - 1.0
    add_label = f"addw_a{args.factor}"
    add_res, add_met = run_once(
        gen, examples,
        lambda c=add_coef: _register_steering(model, args.layer, w_tensor, c,
                                              mode="additive_normalized"),
        add_label, args.seed,
    )
    save_run(out_dir, add_label, add_res, add_met, extra={
        **prov(args.layer, "additive_normalized", "probe_weights_C1.0", conv_ok),
        "arm": "additive", "alpha": args.factor, "factor": args.factor,
        "vector_type": "probe_weights", "vector_norm": w_norm,
    })

    # Projection arm at alpha in {0, 2}: 0 ablates, 2 doubles. Mode 'projection'
    # auto-normalizes the vector so we can pass w_tensor directly.
    for proj_alpha in (0.0, 2.0):
        proj_label = f"projw_a{proj_alpha}"
        proj_res, proj_met = run_once(
            gen, examples,
            lambda a=proj_alpha: _register_steering(model, args.layer, w_tensor, a,
                                                    mode="projection"),
            proj_label, args.seed,
        )
        save_run(out_dir, proj_label, proj_res, proj_met, extra={
            **prov(args.layer, "projection", "probe_weights_C1.0", conv_ok),
            "arm": "projection" if proj_alpha != 0.0 else "patching",
            "alpha": proj_alpha,
            "vector_type": "probe_weights", "vector_norm": w_norm,
        })

    for label in ("base", add_label, "projw_a0.0", "projw_a2.0"):
        state = json.loads((out_dir / f"{label}_state.json").read_text())
        assert "metrics" in state, f"{label} state missing metrics"
        if label != "base":
            assert state.get("mode") in {"additive_normalized", "projection"}, label
            assert state.get("vector_source", "").startswith("probe_weights"), label
    print(f"[smoke/probe_weights] base={base_met['accuracy']:.3f} "
          f"{add_label}={add_met['accuracy']:.3f}")


def _load_probe_bias(repo_root: Path, layer: int) -> float:
    """Load LR probe bias scalar (matched to ``_load_probe_weights``)."""
    candidates = [
        repo_root / "artifacts/cached3/sklearn/steering_configs"
        / f"steering_layer{layer}_probe_biases.npy",
        repo_root / "artifacts/cached3/sklearn/analysis_data"
        / f"layer_{layer}" / "probe_biases.npy",
    ]
    for p in candidates:
        if p.exists():
            arr = np.load(p).astype(np.float32).reshape(-1)
            return float(arr.item() if arr.size == 1 else arr[0])
    raise FileNotFoundError(
        f"probe_biases file missing; tried {[str(c) for c in candidates]}"
    )


def run_variant_reactive(args, out_dir, device, dtype, probe_dir):
    """Smoke test for notebooks/steering_reactive.ipynb. Exercises always_on,
    reactive_detect, and reactive_random arms in 1 example each.

    The reactive smoke uses a tiny ``max_new_tokens`` budget so we hit the
    hook on the order of 50-100 decode steps -- enough to verify the gate
    fires, log file is non-empty, and energy/fire-rate are tracked.
    """
    from probe_pipeline.steering_reactive import (
        ReactiveSteeringHook,
        build_random_fire_pattern,
    )

    w = _load_probe_weights(REPO_ROOT, args.layer)
    b = _load_probe_bias(REPO_ROOT, args.layer)
    cfg, vec, _, _ = load_probe(probe_dir, args.layer)
    vec_tensor = torch.from_numpy(vec).to(torch.float32)

    tokenizer, model = load_model(args.model, device, dtype)
    conv_ok = run_convention_check(model, args.layer)
    examples = load_examples(args.max_examples, args.seed)
    gen = make_generator(model, tokenizer, device, args.max_new_tokens)

    base_res, base_met = run_once(gen, examples, None, "base", args.seed)
    save_run(out_dir, "base", base_res, base_met,
             extra=prov(args.layer, "none", "n/a", conv_ok))

    coef = args.factor - 1.0
    arms = {}

    # Always-on baseline (re-register through the reactive hook with prefill on
    # so it behaves like make_hook applied at every decode step).
    always_hook = ReactiveSteeringHook(
        model, args.layer, w, b, vec, coef,
        mode="additive_normalized", threshold=-1e9,  # always fires
        hysteresis=0, always_on_during_prefill=True,
    )
    with always_hook:
        always_res, always_met = run_once(gen, examples, None, "always_on", args.seed)
    arms["always_on"] = always_hook.stats.to_dict()
    save_run(out_dir, "always_on", always_res, always_met, extra={
        **prov(args.layer, "additive_normalized",
               cfg.get("vector_source", "caa_val"), conv_ok),
        "arm": "always_on", "factor": args.factor,
        "fire_rate": arms["always_on"]["fire_rate"],
        "energy": arms["always_on"]["energy"],
    })

    # Reactive on the binary probe.
    detect_hook = ReactiveSteeringHook(
        model, args.layer, w, b, vec, coef,
        mode="additive_normalized", threshold=0.5, hysteresis=2,
    )
    with detect_hook:
        det_res, det_met = run_once(gen, examples, None, "reactive_detect", args.seed)
    arms["reactive_detect"] = detect_hook.stats.to_dict()
    detect_fire_rate = arms["reactive_detect"]["fire_rate"]
    save_run(out_dir, "reactive_detect", det_res, det_met, extra={
        **prov(args.layer, "additive_normalized", f"probe_layer{args.layer}",
               conv_ok),
        "arm": "reactive_detect", "factor": args.factor,
        "fire_rate": detect_fire_rate,
        "energy": arms["reactive_detect"]["energy"],
        "n_calls": arms["reactive_detect"]["n_calls"],
        "n_fired": arms["reactive_detect"]["n_fired"],
    })

    # Random-fire control matched to the detect arm's empirical fire rate.
    rate = max(0.05, min(0.95, detect_fire_rate or 0.2))
    n_steps = max(64, args.max_new_tokens * args.max_examples + 16)
    pattern = build_random_fire_pattern(n_steps, rate, seed=101)
    rand_hook = ReactiveSteeringHook(
        model, args.layer, w, b, vec, coef,
        mode="additive_normalized", threshold=0.5, hysteresis=0,
        force_fire_pattern=pattern,
    )
    with rand_hook:
        rand_res, rand_met = run_once(gen, examples, None, "reactive_random", args.seed)
    arms["reactive_random"] = rand_hook.stats.to_dict()
    save_run(out_dir, "reactive_random", rand_res, rand_met, extra={
        **prov(args.layer, "additive_normalized", "random_pattern_seed101",
               conv_ok),
        "arm": "reactive_random", "factor": args.factor,
        "fire_rate": arms["reactive_random"]["fire_rate"],
        "energy": arms["reactive_random"]["energy"],
        "expected_fire_rate": rate,
    })

    (out_dir / "reactive_summary.json").write_text(json.dumps(arms, indent=2))
    print(f"[smoke/reactive] base={base_met['accuracy']:.3f} "
          f"always={always_met['accuracy']:.3f} "
          f"detect={det_met['accuracy']:.3f} (fire={detect_fire_rate:.2f}) "
          f"random={rand_met['accuracy']:.3f}")


def run_variant_nie(args, out_dir, device, dtype, probe_dir):
    """Smoke test for notebooks/nie_eval.ipynb. Runs ``compute_token_nie`` on
    a couple of synthetic probe rows with hand-picked labels so the harness
    works without depending on the still-being-built signed pivotal cache."""
    from probe_pipeline.nie import compute_token_nie

    cfg, vec, _, _ = load_probe(probe_dir, args.layer)
    tokenizer, model = load_model(args.model, device, dtype)
    conv_ok = run_convention_check(model, args.layer)

    # Synthetic probe rows: label the position(s) just before the canonical
    # answer marker in a couple of GSM8K continuations. This is not the real
    # probe dataset (that's built in data_preprocessing.ipynb, out of scope
    # here) -- it's just enough to drive compute_token_nie end-to-end.
    rows = []
    for ex_id in range(min(args.max_examples, 2)):
        text = (
            f"Question: A bag has {ex_id+3} apples and {ex_id+2} oranges. "
            f"How many fruits in total? Answer: ####"
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            continue
        labels = [0] * len(ids)
        labels[-2] = 1  # the position immediately before the last token.
        rows.append({
            "text": text,
            "labels": labels,
            "original_dataset_item_id": f"smoke-{ex_id}",
        })

    nie_results = {}
    for mode, coef in (("additive_raw", 0.4), ("additive_normalized", 0.2),
                       ("projection", 0.0)):
        result = compute_token_nie(
            model, tokenizer, rows, args.layer,
            np.asarray(vec, dtype=np.float32),
            coef=coef, mode=mode, device=device,
            max_rows=len(rows), max_positions_per_row=1,
        )
        nie_results[f"{mode}__c{coef}"] = result
        label = f"nie_{mode}_c{coef}"
        save_run(out_dir, label,
                 [{"idx": i, "question": r["text"], "ground_truth": "n/a",
                   "predicted": "n/a", "correct": False, "full_output": ""}
                  for i, r in enumerate(rows)],
                 {"accuracy": 0.0, "f1": 0.0, "parse_rate": 1.0,
                  "correct": 0, "n": len(rows),
                  "nie_mean": result["nie_mean"],
                  "nie_median": result["nie_median"],
                  "n_positions": result["n_positions"]},
                 extra={**prov(args.layer, mode,
                               cfg.get("vector_source", "caa_val"), conv_ok),
                        "coef": coef,
                        "bootstrap_ci": result["bootstrap_ci"],
                        "log_p_base_mean": result["log_p_base_mean"],
                        "log_p_steered_mean": result["log_p_steered_mean"]})

    (out_dir / "nie_summary.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
         for k, v in nie_results.items()},
        indent=2,
    ))
    print(f"[smoke/nie] {len(rows)} rows, "
          f"NIE means: " + ", ".join(
              f"{k}={v['nie_mean']:+.3f}" for k, v in nie_results.items()
          ))


RUNNERS = {
    "main": run_variant_main,
    "random_control": run_variant_random_control,
    "greedy": run_variant_greedy,
    "layer_sweep": run_variant_layer_sweep,
    "additive": run_variant_additive,
    "codelion": run_variant_codelion,
    "probe_weights": run_variant_probe_weights,
    "reactive": run_variant_reactive,
    "nie": run_variant_nie,
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
