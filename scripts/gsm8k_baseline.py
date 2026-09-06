#!/usr/bin/env python
"""Phase 6 gate: does the rebuilt evaluator recover a sane GSM8K baseline?

The v1 harness measured 0.16-0.20 for Qwen3-0.6B against a published ~0.60.
Nothing downstream is worth running until that gap is closed, because a
floored baseline compresses the dynamic range and lets any perturbation look
helpful by chance.

Reports greedy (the paired primary metric) plus sampled seeds (the noise
band), with diagnostics that would have caught the v1 failure: the fraction
of responses actually using the #### marker, and the fraction truncated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_pipeline.gsm8k_eval_v2 import (  # noqa: E402
    build_prompts, extract_gold, generate_batched, score,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("openai/gsm8k", "main", split=a.split).select(range(a.n))
    questions = [r["question"] for r in ds]
    golds = [extract_gold(r["answer"]) for r in ds]
    prompts = build_prompts(questions, tok)

    print(f"[1/3] {a.model} | {len(prompts)} questions from {a.split}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()

    arms = []
    t0 = time.time()
    print("[2/3] greedy", flush=True)
    resp, nnew = generate_batched(model, tok, prompts, greedy=True,
                                  max_new_tokens=a.max_new_tokens, batch_size=a.batch_size)
    arms.append(score(resp, golds, nnew, name="greedy", max_new_tokens=a.max_new_tokens))
    print("      " + json.dumps(arms[-1].as_dict()), flush=True)

    print("[3/3] sampled seeds", flush=True)
    for s in [int(x) for x in a.seeds.split(",")]:
        resp, nnew = generate_batched(model, tok, prompts, greedy=False, seed=s,
                                      max_new_tokens=a.max_new_tokens, batch_size=a.batch_size)
        arms.append(score(resp, golds, nnew, name=f"sampled_s{s}",
                          max_new_tokens=a.max_new_tokens, seed=s))
        print("      " + json.dumps(arms[-1].as_dict()), flush=True)

    sampled = [x.accuracy for x in arms if x.name.startswith("sampled")]
    report = {
        "model": a.model, "n": len(prompts), "seconds": round(time.time() - t0, 1),
        "arms": [x.as_dict() for x in arms],
        "greedy_accuracy": arms[0].accuracy,
        "sampled_mean": sum(sampled) / len(sampled) if sampled else None,
        "sampled_min": min(sampled) if sampled else None,
        "sampled_max": max(sampled) if sampled else None,
        "noise_band": (max(sampled) - min(sampled)) if sampled else None,
    }
    print("\n" + json.dumps(report, indent=2))
    out = a.out or f"/content/baseline_{a.model.split('/')[-1]}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
