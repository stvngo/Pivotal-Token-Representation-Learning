#!/usr/bin/env python
"""Estimate P(success) at *every* position of one rollout, not just midpoints.

PTS bisects, so it measures a handful of prefixes per rollout and the paper
can only draw sparse points. Nothing stops us from measuring every prefix
for a single example -- it is one branch, so the cost is
positions x S rollouts, a few million tokens, minutes on one GPU.

Worth doing once because it answers a question the sparse plot raises and
cannot settle: is the trajectory between measured points smooth, or does
bisection skip over structure? The dense curve is the ground truth that the
search is approximating, and drawing both together shows how much of it a
handful of midpoints recovers.

    scripts/dense_probability_sweep.py --model Qwen/Qwen3-4B --query 1714
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--events", default="runs/qwen3-4b-full/events.jsonl")
    ap.add_argument("--query", default="1714")
    ap.add_argument("--generation", type=int, default=0)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1, help="evaluate every k-th position")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--out", default="artifacts/dense_sweep.json")
    ap.add_argument("--hf-repo", default=None)
    a = ap.parse_args()

    from datasets import load_dataset

    from pts_harness.backends.base import RolloutRequest
    from pts_harness.backends.vllm import VLLMRolloutBackend
    from pts_harness.oracle import GSM8KOracle, gsm8k_answers_from_dataset

    ev = [json.loads(l) for l in open(ROOT / a.events)]
    by = collections.defaultdict(list)
    for e in ev:
        by[(e["query_uid"], e["generation_index"])].append(e)
    es = sorted(by[(a.query, a.generation)], key=lambda e: e["position"])
    if not es:
        raise SystemExit(f"no events for query {a.query}")
    seq, plen = list(es[0]["sequence_token_ids"]), es[0]["prompt_len"]

    ds = load_dataset(a.dataset, "main", split=a.split)
    answers = gsm8k_answers_from_dataset(ds)
    row = ds[int(a.query)]
    oracle = GSM8KOracle(answers)
    question = row["question"]

    positions = list(range(plen, len(seq), a.stride))
    print(f"[1/3] query {a.query}: prompt {plen}, sequence {len(seq)}, "
          f"{len(positions)} prefixes x {a.samples} rollouts", flush=True)

    backend = VLLMRolloutBackend(a.model, dtype="bfloat16",
                                 max_model_len=a.max_model_len,
                                 gpu_memory_utilization=0.90, seed=1234)
    reqs = [
        RolloutRequest(request_id=str(p), prompt_token_ids=tuple(seq[:p]),
                       n=a.samples, seed=9000 + p,
                       max_new_tokens=a.max_new_tokens, temperature=0.6)
        for p in positions
    ]
    t0 = time.time()
    res = backend.generate(reqs)
    print(f"[2/3] generated in {time.time()-t0:.0f}s", flush=True)

    curve = {}
    for r in res:
        p = int(r.request_id)
        s = sum(1 for ro in r.rollouts if oracle.check_success(question, ro.text))
        curve[p] = {"n": len(r.rollouts), "s": s, "p": s / max(1, len(r.rollouts))}

    out = {
        "model": a.model, "query": a.query, "generation": a.generation,
        "question": question, "prompt_len": plen, "seq_len": len(seq),
        "samples": a.samples, "stride": a.stride,
        "seconds": round(time.time() - t0, 1),
        "sequence_token_ids": seq,
        "curve": {str(k): v for k, v in sorted(curve.items())},
        "events": [{k: e[k] for k in
                    ("position", "token_id", "prob_before", "prob_after",
                     "prob_delta", "n_before", "s_before", "n_after", "s_after")}
                   for e in es],
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"[3/3] wrote {path} ({len(curve)} positions)", flush=True)

    if a.hf_repo:
        try:
            from huggingface_hub import HfApi

            from probe_pipeline.artifacts_io import resolve_hf_token
            HfApi(token=resolve_hf_token(required=True)).upload_file(
                path_or_fileobj=str(path), path_in_repo=f"sweeps/{path.name}",
                repo_id=a.hf_repo, repo_type="dataset")
            print(f"pushed -> {a.hf_repo}/sweeps/{path.name}", flush=True)
        except Exception as exc:
            print(f"[hf] push failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
