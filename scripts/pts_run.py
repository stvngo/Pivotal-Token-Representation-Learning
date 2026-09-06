#!/usr/bin/env python
"""Run PTS token search over a dataset, resumably.

Identical on Colab and over SSH; only --backend and --device differ. State
lives in a run directory that can be mirrored to HuggingFace, so a killed
session resumes without recomputing anything.

    # validation: reproduce a slice of codelion's released events
    python scripts/pts_run.py --model Qwen/Qwen3-0.6B --backend vllm \\
        --max-examples 20 --out runs/qwen3-0.6b-validate

    # generation, sharded across two Colab sessions
    python scripts/pts_run.py --model Qwen/Qwen3-1.7B --backend vllm \\
        --max-examples 1500 --shard 0 --num-shards 2 --out runs/qwen3-1.7b

Cost model (see docs/pts_semantics.md): total generated tokens is roughly
    Q*S*L  +  Q*f*G*B*S*L
with f the fraction of queries inside [min-prob, max-prob] and B the unique
bisection midpoints per rollout. Both are measured by --calibrate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pts_harness.checkpoint import RunStore  # noqa: E402
from pts_harness.oracle import (  # noqa: E402
    GSM8KOracle, MathOracle, gsm8k_answers_from_dataset, math_answers_from_dataset,
)
from pts_harness.scheduler import QuerySpec, WaveScheduler  # noqa: E402
from pts_harness.search import SearchConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--revision", default=None)
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--config", default="main")
    p.add_argument("--split", default="train")
    p.add_argument("--task", default="gsm8k", choices=["gsm8k", "math"],
                   help="which oracle and which field names. 'math' expects a "
                        "'problem' field and a boxed answer, and scores a "
                        "response with no \\boxed as a failure rather than "
                        "falling back to the last number -- MATH solutions are "
                        "full of intermediate quantities, so that fallback "
                        "would reward models that never committed.")
    p.add_argument("--question-key", default=None,
                   help="override the field holding the question")
    p.add_argument("--out", default="runs/pts")
    p.add_argument("--session", default=None)
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")

    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--prob-threshold", type=float, default=0.2)
    p.add_argument("--min-prob", type=float, default=0.2)
    p.add_argument("--max-prob", type=float, default=0.8)
    p.add_argument("--max-generations", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-active", type=int, default=64,
                   help="queries in flight; this is what fills the GPU")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--chat-template", action="store_true",
                   help="condition through the chat template. Upstream does "
                        "NOT, so leave this off to match released datasets.")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--calibrate", action="store_true",
                   help="report measured f, B and throughput, then exit")
    p.add_argument("--hf-repo", default=None,
                   help="mirror the run directory to this HF dataset repo. "
                        "Strongly recommended: a Colab VM can be reclaimed "
                        "mid-run, and without this the checkpoint dies with it.")
    p.add_argument("--hf-every", type=int, default=2,
                   help="minutes between mirror commits. Deliberately short: "
                        "this is the maximum amount of search that a "
                        "reclaimed VM can cost you.")
    p.add_argument("--hf-private", action="store_true", default=True)
    p.add_argument("--no-hf-resume", action="store_true",
                   help="do not restore prior state from --hf-repo first")
    return p.parse_args()


def pull_hf_run(run_dir: str, repo_id: str) -> int:
    """Restore a mirrored run before starting, so a fresh VM resumes.

    The other half of mirroring, and useless without it: pushing state off
    a VM only helps if the next VM reads it back. Resume is keyed on
    content-addressed node ids, so every estimate recovered here is a
    rollout that does not have to be paid for again.
    """
    try:
        from huggingface_hub import snapshot_download

        from probe_pipeline.artifacts_io import resolve_hf_token

        name = Path(run_dir).name
        dest = Path(run_dir).parent
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=[f"{name}/**"],
            token=resolve_hf_token(),
        )
        n = sum(1 for _ in Path(run_dir).rglob("probs.jsonl"))
        print(f"[hf] restored {run_dir} from {repo_id} ({n} session file(s))", flush=True)
        return n
    except Exception as exc:
        print(f"[hf] nothing to restore: {type(exc).__name__}: {exc}", flush=True)
        return 0


def start_hf_mirror(run_dir: str, repo_id: str, every: int):
    """Mirror the run directory to HuggingFace while the search runs.

    The VM is not durable. Colab reclaims runtimes, and the CLI can lose
    track of one mid-run; either way /content goes with it. Because the
    store is append-only JSONL and resume is keyed on content, a mirrored
    run costs at most `every` minutes when that happens instead of the
    whole job.
    """
    try:
        from huggingface_hub import CommitScheduler

        from probe_pipeline.artifacts_io import resolve_hf_token

        token = resolve_hf_token(required=True)
        sched = CommitScheduler(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=run_dir,
            path_in_repo=Path(run_dir).name,
            every=every,
            private=True,
            token=token,
            allow_patterns=["*.jsonl", "*.json"],
        )
        print(f"[hf] mirroring {run_dir} -> {repo_id} every {every}m", flush=True)
        return sched
    except Exception as exc:  # never let mirroring failure kill the run
        print(f"[hf] mirror disabled: {type(exc).__name__}: {exc}", flush=True)
        return None


# Without this the model has no reason to emit \boxed at all, and the
# oracle -- which deliberately has no fallback -- would score almost every
# rollout a failure, collapsing the acceptance band to nothing.
MATH_SUFFIX = "\n\nPut your final answer in \\boxed{}."


def shard_of(uid: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    h = hashlib.blake2b(uid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % num_shards


def build_specs(args, tokenizer) -> tuple[list[QuerySpec], dict[str, str]]:
    from datasets import load_dataset

    ds = (load_dataset(args.dataset, args.config, split=args.split)
          if args.config else load_dataset(args.dataset, split=args.split))
    ds = ds.select(range(min(args.max_examples, len(ds))))
    answers = (math_answers_from_dataset(ds) if args.task == "math"
               else gsm8k_answers_from_dataset(ds))
    qkey = args.question_key or ("problem" if args.task == "math" else "question")

    specs: list[QuerySpec] = []
    for i, row in enumerate(ds):
        uid = str(i)
        if shard_of(uid, args.num_shards) != args.shard:
            continue
        question = row[qkey]
        conditioned = question + (MATH_SUFFIX if args.task == "math" else "")
        if args.chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": conditioned}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            text = conditioned
        specs.append(
            QuerySpec(
                uid=uid,
                prompt_token_ids=tokenizer.encode(text, add_special_tokens=False),
                query=question,
                answer=answers.get(question, ""),
            )
        )
    return specs, answers


def make_backend(args, tokenizer):
    if args.backend == "vllm":
        from pts_harness.backends.vllm import VLLMRolloutBackend

        return VLLMRolloutBackend(
            args.model,
            revision=args.revision,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )

    import torch
    from transformers import AutoModelForCausalLM

    device = args.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=getattr(torch, args.dtype)
        )
        .to(device)
        .eval()
    )
    from pts_harness.backends.hf import HFRolloutBackend

    return HFRolloutBackend(model, tokenizer, device=device)


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    specs, answers = build_specs(args, tokenizer)
    print(f"[1/3] {len(specs)} queries (shard {args.shard}/{args.num_shards})")

    cfg = SearchConfig(
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        min_prob=args.min_prob,
        max_prob=args.max_prob,
        max_generations=args.max_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        run_seed=args.seed,
    )

    # Bisection conditions on ever-longer prefixes of a rollout, so the
    # context at a deep node is prompt + (up to) a whole rollout, and we then
    # ask for max_new_tokens more. A window sized for one rollout is
    # therefore too small by roughly max_new_tokens, and the failure is
    # quiet: vLLM rejects the individual request, that probability estimate
    # is lost, and the search carries on around the hole.
    if args.backend == "vllm":
        longest_prompt = max((len(s.prompt_token_ids) for s in specs), default=0)
        needed = longest_prompt + 2 * args.max_new_tokens
        if args.max_model_len < needed:
            print(f"[warn] --max-model-len {args.max_model_len} is below the "
                  f"{needed} a deep bisection node can require "
                  f"(longest prompt {longest_prompt} + 2 x {args.max_new_tokens}). "
                  f"Raising it to {needed}.", flush=True)
            args.max_model_len = needed

    print(f"[2/3] backend={args.backend} model={args.model}")
    backend = make_backend(args, tokenizer)
    oracle = MathOracle(answers) if args.task == "math" else GSM8KOracle(answers)

    run_cfg = vars(args) | {"search": cfg.__dict__}
    run_cfg.pop("hf_repo", None)          # not part of the run's identity
    started = time.time()
    if args.hf_repo and not args.no_hf_resume:
        pull_hf_run(args.out, args.hf_repo)
    mirror = start_hf_mirror(args.out, args.hf_repo, args.hf_every) if args.hf_repo else None

    with RunStore(args.out, session=args.session, config=run_cfg) as store:
        sched = WaveScheduler(
            backend,
            oracle,
            cfg,
            store=store,
            max_active=args.max_active,
            model_key=f"{args.model}@{args.revision or 'main'}",
            on_wave=lambda w: print(
                f"  wave {w['wave']:>3}  active {w['active']:>3}  "
                f"pending {w['pending']:>4}  events {w['events']:>4}  "
                f"rollouts {w['rollouts']:>7}",
                flush=True,
            ),
        )
        print("[3/3] searching")
        summary = sched.run(specs)
        events_path = store.export_events(Path(args.out) / "events.jsonl")

    if mirror is not None:
        try:
            mirror.trigger().result()      # final push before we exit
            print(f"[hf] final push to {args.hf_repo} done", flush=True)
        except Exception as exc:
            print(f"[hf] final push failed: {exc}", flush=True)

    elapsed = time.time() - started
    out = summary.as_dict()
    searched = summary.queries_completed - summary.queries_skipped
    out["measured_filter_fraction_f"] = (
        searched / summary.queries_completed if summary.queries_completed else 0.0
    )
    out["measured_nodes_per_searched_query"] = (
        summary.nodes / searched if searched else 0.0
    )
    out["rollout_tokens_per_second"] = (
        summary.rollouts * args.max_new_tokens / elapsed if elapsed else 0.0
    )
    out["events_path"] = str(events_path)
    if hasattr(oracle, "parse_rate"):
        # The single number that says whether a MATH run is meaningful.
        out["oracle_parse_rate"] = oracle.parse_rate
        out["oracle_rollouts_scored"] = oracle.n_scored
        if oracle.n_scored and oracle.parse_rate < 0.5:
            print(f"\n[warn] only {oracle.parse_rate:.1%} of rollouts produced a "
                  f"parseable answer; the acceptance band is probably starved "
                  f"and these events should not be trusted", flush=True)

    (Path(args.out) / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n" + json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
