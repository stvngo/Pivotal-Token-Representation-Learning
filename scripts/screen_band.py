#!/usr/bin/env python
"""Find the GSM8K questions where the model is genuinely uncertain.

Qwen3-4B answers 94% of GSM8K correctly under greedy decoding with a chat
template. On that population an intervention has 18 questions to move and
a ceiling effect swamps any real effect -- so a null would be
uninterpretable and a positive would be noise.

The fix is not to pick a different metric but to evaluate on the population
the detector is *defined* on. PTS only searches queries whose baseline
success probability lies in ``[min_prob, max_prob]``, because a token can
only shift ``P(success)`` if the outcome was not already settled: a question
the model always or never solves has no pivotal token. Restricting the
causal evaluation to that same band is therefore not cherry-picking, it is
matching the evaluation population to the data-generating process.

``--backend vllm`` is the fast path and the default choice for the upper
scales: screening is pure generation with no hooks, so there is no reason
to pay HuggingFace's decode loop for it. vLLM and HF sample slightly
differently, which is acceptable here precisely because the screen is an
independent estimate of question difficulty rather than the metric being
compared -- it decides *which* questions are evaluated, not how any arm
scores on them.

The screen must be independent of the arms it will be used to compare.
Selecting on whether the *base arm* got a question right would condition on
a random outcome, and regression to the mean would then make any
intervention look good on the base-wrong subset. So the band is estimated
from its own independent sampled rollouts, before any arm runs.

    python scripts/screen_band.py --model Qwen/Qwen3-4B --n 800 --rollouts 8
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
    build_prompts, extract_answer, extract_gold, generate_batched, is_correct,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=800, help="test questions to screen")
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--min-prob", type=float, default=0.2)
    ap.add_argument("--max-prob", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many questions first. GSM8K test is only "
                         "1319 questions and the band is a small slice of it, "
                         "so the upper scales need train[3000:] as well -- PTS "
                         "consumed train[0:3000] and nothing else, so the "
                         "remainder is unseen by both the search and the probes.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("openai/gsm8k", "main", split=a.split)
    lo = min(a.offset, len(ds))
    ds = ds.select(range(lo, min(lo + a.n, len(ds))))
    questions = [r["question"] for r in ds]
    golds = [extract_gold(r["answer"]) for r in ds]
    prompts = build_prompts(questions, tok)

    print(f"[1/2] {a.model} | screening {len(prompts)} questions "
          f"x {a.rollouts} rollouts | backend={a.backend}", flush=True)

    t0 = time.time()
    n_success = [0] * len(prompts)

    if a.backend == "vllm":
        from pts_harness.backends.base import RolloutRequest
        from pts_harness.backends.vllm import VLLMRolloutBackend

        backend = VLLMRolloutBackend(
            a.model, dtype="bfloat16", max_model_len=a.max_model_len,
            gpu_memory_utilization=a.gpu_memory_utilization, seed=1000,
        )
        reqs = [
            RolloutRequest(
                request_id=str(i),
                prompt_token_ids=tuple(tok.encode(pr, add_special_tokens=False)),
                n=a.rollouts, seed=1000 + i,
                max_new_tokens=a.max_new_tokens, temperature=a.temperature,
            )
            for i, pr in enumerate(prompts)
        ]
        for res in backend.generate(reqs):
            i = int(res.request_id)
            n_success[i] = sum(int(is_correct(extract_answer(r.text), golds[i]))
                               for r in res.rollouts)
        print(f"  all {a.rollouts} rollouts done ({time.time()-t0:.0f}s)", flush=True)
        backend.close()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16).to("cuda").eval()
        for k in range(a.rollouts):
            resp, _ = generate_batched(
                model, tok, prompts, greedy=False, seed=1000 + k,
                temperature=a.temperature, max_new_tokens=a.max_new_tokens,
                batch_size=a.batch_size,
            )
            for i, (r, g) in enumerate(zip(resp, golds)):
                n_success[i] += int(is_correct(extract_answer(r), g))
            done = sum(1 for s in n_success
                       if a.min_prob <= s / (k + 1) <= a.max_prob)
            print(f"  rollout {k+1}/{a.rollouts}  in-band so far {done}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    p_hat = [s / a.rollouts for s in n_success]
    band = [lo + i for i, p in enumerate(p_hat) if a.min_prob <= p <= a.max_prob]

    report = {
        "model": a.model, "split": a.split, "offset": a.offset,
        "n_screened": len(prompts),
        "rollouts": a.rollouts, "min_prob": a.min_prob, "max_prob": a.max_prob,
        "backend": a.backend,
        "seconds": round(time.time() - t0, 1),
        "mean_success": sum(p_hat) / len(p_hat),
        "n_in_band": len(band),
        "band_fraction": len(band) / len(prompts),
        "n_always_right": sum(1 for p in p_hat if p > a.max_prob),
        "n_always_wrong": sum(1 for p in p_hat if p < a.min_prob),
        "indices": band,
        "p_hat": p_hat,
    }
    print(f"\n[2/2] {len(band)} / {len(prompts)} in band "
          f"[{a.min_prob}, {a.max_prob}]  (mean success {report['mean_success']:.3f})",
          flush=True)

    tag = a.model.split("/")[-1] + (f"_{a.split}{a.offset}" if a.offset or a.split != "test" else "")
    out = Path(a.out or f"/content/band_{tag}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    if a.hf_repo:
        try:
            from huggingface_hub import HfApi
            from probe_pipeline.artifacts_io import resolve_hf_token
            HfApi(token=resolve_hf_token(required=True)).upload_file(
                path_or_fileobj=str(out), path_in_repo=f"bands/{out.name}",
                repo_id=a.hf_repo, repo_type="dataset",
            )
            print(f"pushed -> {a.hf_repo}/bands/{out.name}")
        except Exception as exc:
            print(f"[hf] push failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
