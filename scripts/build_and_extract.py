#!/usr/bin/env python
"""Build probe rows from a PTS dataset and extract activations into a v2 cache.

Runs identically on a laptop (MPS/CPU), on Colab, and over SSH -- the only
difference is --device and --dtype.

    python scripts/build_and_extract.py \
        --model Qwen/Qwen3-0.6B \
        --pts-dataset codelion/Qwen3-0.6B-pts \
        --out data/acts_v2 --device mps

Timing note: extraction is cheap. Qwen3-0.6B over 425 branches is ~10 min on
an M3 laptop. The multi-hour figure associated with this project belongs to
`pts run` (the rollout search), not to this step -- see docs/pts_semantics.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Running as `python scripts/build_and_extract.py` puts scripts/ on sys.path,
# not the repo root, so probe_pipeline would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--pts-dataset", default="codelion/Qwen3-0.6B-pts",
                   help="released HF events; positions are recovered by re-tokenizing")
    p.add_argument("--pts-events", default=None,
                   help="events.jsonl from our own pts_harness run. Preferred: "
                        "these carry exact positions, so nothing is inferred.")
    p.add_argument("--revision", default=None, help="pin the PTS dataset revision")
    p.add_argument("--out", default="data/acts_v2")
    p.add_argument("--tag", default=None, help="filename prefix (default: model basename)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="model compute dtype. PTS itself used bf16 on CUDA, fp32 on MPS.",
    )
    p.add_argument("--store-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--ratio", type=float, default=1.0, help="negatives per positive")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default=None, help="comma-separated hidden_states indices")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap branches, for smoke runs")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_pipeline.extract_v2 import run_extraction
    from probe_pipeline.probe_dataset import (
        build_probe_dataset,
        build_rows_from_harness_events,
        normalize_pts_rows,
        split_by_query,
    )

    tag = args.tag or args.model.split("/")[-1].lower()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = args.pts_events or args.pts_dataset
    print(f"[1/4] tokenizer + PTS events  ({source})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.pts_events:
        # Our own run: exact positions, no re-tokenization, no merge hazard.
        from probe_pipeline.artifacts_io import iter_jsonl

        events = list(iter_jsonl(Path(args.pts_events)))
        print(f"      {len(events)} harness events")
        print("[2/4] building probe rows (exact positions)")
        built, stats = build_rows_from_harness_events(
            events, negative_to_positive_ratio=args.ratio, seed=args.seed
        )
    else:
        ds = load_dataset(args.pts_dataset, split="train", revision=args.revision)
        rows = normalize_pts_rows(dict(r) for r in ds)
        print(f"      {len(ds)} events -> {len(rows)} token events")
        print("[2/4] building probe rows (positions recovered + verified)")
        built, stats = build_probe_dataset(
            rows, tokenizer, negative_to_positive_ratio=args.ratio, seed=args.seed
        )
    print("      " + json.dumps(stats.as_dict()))
    if args.limit:
        built = built[: args.limit]

    train, test = split_by_query(built, test_size=args.test_size, seed=args.seed)
    print(
        f"      train {len({r.query_id for r in train})}q/{len(train)}b  "
        f"test {len({r.query_id for r in test})}q/{len(test)}b"
    )

    print(f"[3/4] loading {args.model} ({args.dtype})")
    device = pick_device(args.device)
    t0 = time.time()
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        .to(device)
        .eval()
    )
    print(f"      on {device} in {time.time() - t0:.1f}s")

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    print("[4/4] extracting")
    summaries = {}
    for split, subset in (("train", train), ("test", test)):
        summaries[split] = run_extraction(
            subset,
            model,
            tokenizer,
            out_dir / f"{tag}_{split}.safetensors",
            split=split,
            model_name=args.model,
            layers=layers,
            dtype=args.store_dtype,
            device=device,
            max_seq_len=args.max_seq_len,
            source_dataset=source,
        )
        print("      " + json.dumps(summaries[split]))

    report = {"args": vars(args), "build_stats": stats.as_dict(), "extraction": summaries}
    (out_dir / f"{tag}_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\ndone -> {out_dir}/{tag}_report.json")


if __name__ == "__main__":
    main()
