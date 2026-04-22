"""Short smoke test for the Colab notebook pipeline.

Exercises the same code paths as `notebooks/steering_gsm8k_colab.ipynb`
(probe loading, steering hook, GSM8K generate + parse, metrics, plotting,
state-save) on a trivially small workload so we can verify correctness in
~1-2 minutes instead of ~1 hour.

Usage:
    python scripts/smoke_colab_notebook.py --max-examples 2 --factor 1.4

Successful run writes:
    artifacts/smoke/nb_results/base_*.json
    artifacts/smoke/nb_results/steered_1.4_*.json
    artifacts/smoke/nb_results/summaries.json
    artifacts/smoke/nb_results/accuracy_f1_vs_factor.png
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
    return {
        "accuracy": correct / n,
        "f1": f1,
        "parse_rate": parsed / n,
        "correct": correct,
        "n": n,
    }


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


def main() -> int:
    ap = argparse.ArgumentParser()
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
        "--out-dir",
        default=str(REPO_ROOT / "artifacts/smoke/nb_results"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed(args.seed)

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    print(f"[smoke] device={device} dtype={dtype}")

    # ------------------------------------------------------------ probe
    probe_dir = Path(args.probe_dir)
    cfg_path = probe_dir / f"steering_layer{args.layer}.json"
    vec_path = probe_dir / f"steering_layer{args.layer}_vector.npy"
    if not cfg_path.exists() or not vec_path.exists():
        print(f"[smoke] FAIL: probe files missing in {probe_dir}")
        return 2
    steering_cfg = json.loads(cfg_path.read_text())
    steering_vector = np.load(vec_path).astype(np.float32)
    vec_tensor = torch.from_numpy(steering_vector).to(torch.float32)
    print(
        f"[smoke] probe: layer={steering_cfg['layer']} "
        f"hidden_dim={steering_cfg['hidden_dim']} "
        f"norm={steering_cfg['vector_norm']:.3f}"
    )

    # ------------------------------------------------------------ parser
    assert extract_gsm8k_answer("#### 42") == "42"
    assert is_correct("3.0", "3") is True
    print("[smoke] parser inline-asserts OK")

    # ------------------------------------------------------------ model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[smoke] loaded {args.model} in {time.time() - t0:.1f}s")
    assert steering_vector.shape[0] == model.config.hidden_size

    # ------------------------------------------------------------ data
    from datasets import load_dataset
    ds_full = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(args.seed)
    idxs = sorted(
        rng.choice(len(ds_full), size=min(args.max_examples, len(ds_full)), replace=False).tolist()
    )
    subset = ds_full.select(idxs)
    examples = [
        {
            "question": r["question"],
            "ground_truth": extract_gsm8k_answer(r["answer"]),
        }
        for r in subset
    ]
    print(f"[smoke] sampled {len(examples)} GSM8K examples")

    # ------------------------------------------------------------ generate helper
    def generate_once(prompt: str) -> str:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def evaluate(label, strength):
        _seed(args.seed)
        handles = []
        if strength is not None:
            handles = [
                _get_decoder_layer(model, args.layer).register_forward_hook(
                    _make_hook(vec_tensor, strength)
                )
            ]
        results = []
        try:
            for i, ex in enumerate(examples):
                prompt = f"Question: {ex['question']}\n\nLet's think step by step.\n\n"
                text = generate_once(prompt)
                pred = extract_gsm8k_answer(text)
                ok = is_correct(pred, ex["ground_truth"])
                results.append({
                    "idx": i,
                    "question": ex["question"],
                    "ground_truth": ex["ground_truth"],
                    "predicted": pred,
                    "correct": ok,
                    "full_output": text[-800:],
                })
                print(
                    f"[smoke]   {label} #{i}: gt={ex['ground_truth']} "
                    f"pred={pred} correct={ok}"
                )
        finally:
            for h in handles:
                h.remove()
        return results, compute_metrics(results)

    def save_run(label, results, metrics, extra=None):
        (out_dir / f"{label}_results.json").write_text(json.dumps(results, indent=2))
        (out_dir / f"{label}_generations.txt").write_text(
            "\n\n====\n\n".join(
                f"[{r['idx']}] gt={r['ground_truth']} pred={r['predicted']} correct={r['correct']}\n{r['full_output']}"
                for r in results
            )
        )
        state = {
            "label": label,
            "model": args.model,
            "layer": args.layer,
            "seed": args.seed,
            "max_examples": args.max_examples,
            "max_new_tokens": args.max_new_tokens,
            "device": device,
            "metrics": metrics,
        }
        if extra:
            state.update(extra)
        (out_dir / f"{label}_state.json").write_text(json.dumps(state, indent=2))

    # ------------------------------------------------------------ base
    base_res, base_met = evaluate("base", strength=None)
    save_run("base", base_res, base_met)
    print(
        f"[smoke] base: acc={base_met['accuracy']:.3f} "
        f"f1={base_met['f1']:.3f} parse={base_met['parse_rate']:.3f}"
    )

    # ------------------------------------------------------------ steered
    label = f"steered_{args.factor}"
    strength = args.factor - 1.0
    steered_res, steered_met = evaluate(label, strength=strength)
    save_run(label, steered_res, steered_met, extra={
        "factor": args.factor,
        "strength": strength,
        "vector_type": steering_cfg["vector_type"],
        "vector_norm": steering_cfg["vector_norm"],
        "vector_path": vec_path.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(
        f"[smoke] {label}: acc={steered_met['accuracy']:.3f} "
        f"f1={steered_met['f1']:.3f} parse={steered_met['parse_rate']:.3f}"
    )

    # ------------------------------------------------------------ aggregate + tiny plot
    all_metrics = {"base": base_met, label: steered_met}
    (out_dir / "summaries.json").write_text(json.dumps(all_metrics, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = list(all_metrics.keys())
    ax.bar(names, [all_metrics[n]["accuracy"] for n in names], color=["#888", "#4c72b0"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Smoke: base vs steered")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_f1_vs_factor.png")
    plt.close(fig)

    print(f"[smoke] wrote artifacts to {out_dir}")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
