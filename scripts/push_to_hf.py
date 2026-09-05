#!/usr/bin/env python
"""Publish PTS events and probe activations to HuggingFace.

Two dataset repos per model, kept separate because they have very different
shapes and audiences:

* ``<user>/<model>-pts-tokens`` -- the pivotal-token events. Comparable to
  ``codelion/<model>-pts`` but carrying an exact ``position``, the full
  searched rollout, and the sufficient statistics behind each judgement.
* ``<user>/<model>-pivotal-activations`` -- the labelled residual-stream
  activations at *t-1*, ready to train probes on without a GPU.

    python scripts/push_to_hf.py --events runs/qwen3-0.6b-full/events.jsonl \\
        --acts data/acts_v2 --tag qwen3-0.6b --model Qwen/Qwen3-0.6B

Private by default. Pass --public when you are ready to release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_pipeline.artifacts_io import iter_jsonl, resolve_hf_token  # noqa: E402

EVENTS_CARD = """---
license: apache-2.0
task_categories:
- text-generation
tags:
- interpretability
- pivotal-tokens
- mechanistic-interpretability
- reasoning
---

# {model} pivotal token events

Token-granularity Pivotal Token Search (PTS) events for `{model}` on
{dataset}. A token is pivotal when appending it to a prefix moves the
model's estimated probability of eventually solving the task by more than
`{threshold}`:

```
|P(success | prefix + token) - P(success | prefix)| > {threshold}
```

Generated with a reimplementation of the search from
[codelion/pts](https://github.com/codelion/pts) (pinned at `8334808`) that
batches rollouts across queries, so the wall clock scales with the
bisection depth rather than the number of questions.

## What is different from the released `codelion/*-pts` datasets

**Exact positions.** Every event carries `position`, the absolute
prompt-inclusive index of the pivotal token, plus `sequence_token_ids` for
the whole searched rollout. The released datasets predate that field, so
consumers must recover the index by re-tokenizing the prefix -- which is
unsafe, because BPE does not guarantee
`enc(a + b) == enc(a) + enc(b)` and a pivot token like `"."` frequently
merges with the preceding word.

**Sufficient statistics, not verdicts.** Each event stores `n_before`,
`s_before`, `n_after`, `s_after` -- the raw rollout counts -- rather than
only `prob_delta`. So the acceptance threshold is a *post-hoc* knob: the
event set can be re-derived at a different `tau` with no GPU, and each
event carries a Wilson interval and can be filtered by a two-proportion
test.

That last point matters more than it sounds. `delta_hat` is a difference of
two binomials, so at the common setting (S=50, tau=0.2) the acceptance
threshold sits at only 2 sigma -- a false-acceptance rate around 4.6% per
tested position under the null. A meaningful share of any PTS positive
class is sampling noise, and these fields are what let you quantify it.

## Schema

| field | meaning |
| --- | --- |
| `query_uid` | source question id |
| `generation_index` | which sampled rollout of that question |
| `position` | **absolute, prompt-inclusive index of the pivotal token** |
| `token_id` | the pivotal token |
| `sequence_token_ids` | prompt + the full searched rollout |
| `prompt_len` | where the generated span begins |
| `prob_before` / `prob_after` / `prob_delta` | success probabilities |
| `is_positive` | `prob_delta > 0` |
| `n_before` / `s_before` / `n_after` / `s_after` | rollouts and successes |
| `ci_before` / `ci_after` | Wilson 95% intervals |
| `baseline_prob` | the query's unconditioned success rate |

## Generation settings

```json
{settings}
```

Note the search only visits questions whose baseline success probability
lies inside `[{min_prob}, {max_prob}]`: a question the model always or
never solves has no pivotal token, because a saturated probability cannot
move. Pivot density is therefore confounded with task difficulty relative
to model capability, and this set is **not** a uniform sample of the source
dataset.

## Companion

Labelled residual-stream activations built from these events:
`{acts_repo}`
"""

ACTS_CARD = """---
license: apache-2.0
tags:
- interpretability
- pivotal-tokens
- probing
- activations
---

# {model} pivotal-token activations

Residual-stream activations for `{model}`, labelled for training probes
that predict **whether the next token will be pivotal**, from the position
*before* it is generated.

Built from `{events_repo}`.

## The labelling convention

The probe reads position *t-1* and predicts pivotality at *t*. That
one-step offset is the point: at *t-1* the prediction is available before
the forward pass that commits to the token, so it can gate an intervention.
A probe read at the pivotal token itself is a post-hoc classifier.

| label | meaning |
| --- | --- |
| `1` | immediately precedes a pivotal token |
| `-1` | a sampled non-pivotal answer position |

Sampled 1:1. Rows are grouped by question, and the train/test split is at
the **question** level -- positions from one reasoning trace are not
independent, so splitting at the row level would leak.

**One row per rollout branch.** PTS samples several rollouts per question
and they diverge after a few tokens. Collapsing a question to a single
"longest" sequence discards every pivot on a diverging branch; on the
released Qwen3-0.6B events that loses 74% of them. Each maximal branch is
kept as its own sequence, and because branches share prefixes -- and the
residual stream at position `p` depends only on tokens `0..p` -- each
distinct context is labelled exactly once.

## Files

`{tag}_train.safetensors`, `{tag}_test.safetensors`. Each holds every
layer of `outputs.hidden_states` at the labelled positions only, in
bfloat16, with all layers sharing one row ordering.

| tensor | shape | notes |
| --- | --- | --- |
| `layer_{{L}}` | `(N, d)` | `hidden_states[L]`; index 0 is the embedding output |
| `y` | `(N,)` | `+1` pivotal, `-1` non-pivotal |
| `query_idx` | `(N,)` | index into the `query_ids` metadata; **cluster by this** |
| `token_pos` | `(N,)` | position within the sequence |
| `token_id` | `(N,)` | token at that position |
| `prob_delta` | `(N,)` | signed shift of the *following* token |
| `entropy`, `margin`, `top1_prob` | `(N,)` | next-token uncertainty, for baselines |

The file metadata carries a manifest recording
`hidden_state_convention` and `extraction_position`, so a consumer can
assert the convention rather than trust a comment.

## Usage

```python
from safetensors.torch import load_file
d = load_file("{tag}_train.safetensors")
X = d["layer_18"].float().numpy()
y = (d["y"] > 0).numpy().astype(int)
groups = d["query_idx"].numpy()      # cluster bootstrap / grouped CV by question
```

`prob_delta` is what makes the *signed* probe possible: restrict to `y == 1`
and relabel by `sign(prob_delta)` to ask whether the impending shift is
helpful or harmful, rather than merely large.

## Baselines worth beating

`entropy`, `margin` and `top1_prob` are included because they are the
cheaper explanations any claim here has to rule out: high-entropy "forking"
tokens are the incumbent notion of a critical position. `token_id` supports
the other one -- a token-identity lookup table.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", help="events.jsonl from a pts_harness run")
    p.add_argument("--acts", help="directory holding <tag>_{train,test}.safetensors")
    p.add_argument("--tag", default="qwen3-0.6b")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--user", default=None, help="defaults to the token's owner")
    p.add_argument("--events-repo", default=None)
    p.add_argument("--acts-repo", default=None)
    p.add_argument("--summary", default=None, help="run summary.json, for the card")
    p.add_argument("--public", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    from huggingface_hub import HfApi

    token = resolve_hf_token(required=True)
    api = HfApi(token=token)
    user = a.user or api.whoami().get("name")

    model_base = a.model.split("/")[-1]
    events_repo = a.events_repo or f"{user}/{model_base}-pts-tokens"
    acts_repo = a.acts_repo or f"{user}/{model_base}-pivotal-activations"
    private = not a.public

    settings, threshold, min_prob, max_prob, dataset = {}, "0.2", "0.2", "0.8", "openai/gsm8k"
    if a.summary and Path(a.summary).exists():
        s = json.loads(Path(a.summary).read_text())
        args = s.get("args", {})
        settings = {
            k: args.get(k)
            for k in ("model", "dataset", "split", "num_samples", "prob_threshold",
                      "min_prob", "max_prob", "max_generations", "max_new_tokens",
                      "temperature", "top_p", "top_k", "max_examples", "seed")
            if args.get(k) is not None
        }
        threshold = str(args.get("prob_threshold", threshold))
        min_prob, max_prob = str(args.get("min_prob", min_prob)), str(args.get("max_prob", max_prob))
        dataset = str(args.get("dataset", dataset))

    plan = []
    if a.events:
        plan.append((events_repo, "events"))
    if a.acts:
        plan.append((acts_repo, "activations"))
    print(f"user={user} private={private}")
    for repo, what in plan:
        print(f"  {what:12s} -> https://huggingface.co/datasets/{repo}")
    if a.dry_run:
        return

    if a.events:
        path = Path(a.events)
        n = sum(1 for _ in iter_jsonl(path))
        api.create_repo(events_repo, repo_type="dataset", private=private, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo="events.jsonl",
            repo_id=events_repo, repo_type="dataset",
        )
        card = EVENTS_CARD.format(
            model=a.model, dataset=dataset, threshold=threshold,
            min_prob=min_prob, max_prob=max_prob, acts_repo=acts_repo,
            settings=json.dumps(settings, indent=2),
        )
        api.upload_file(
            path_or_fileobj=card.encode(), path_in_repo="README.md",
            repo_id=events_repo, repo_type="dataset",
        )
        if a.summary and Path(a.summary).exists():
            api.upload_file(
                path_or_fileobj=a.summary, path_in_repo="summary.json",
                repo_id=events_repo, repo_type="dataset",
            )
        print(f"pushed {n} events -> {events_repo}")

    if a.acts:
        acts_dir = Path(a.acts)
        files = sorted(acts_dir.glob(f"{a.tag}_*"))
        if not files:
            print(f"no files matching {a.tag}_* in {acts_dir}")
        else:
            api.create_repo(acts_repo, repo_type="dataset", private=private, exist_ok=True)
            for f in files:
                api.upload_file(
                    path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=acts_repo, repo_type="dataset",
                )
                print(f"  uploaded {f.name} ({f.stat().st_size/1e6:.1f} MB)")
            card = ACTS_CARD.format(model=a.model, events_repo=events_repo, tag=a.tag)
            api.upload_file(
                path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id=acts_repo, repo_type="dataset",
            )
            print(f"pushed activations -> {acts_repo}")


if __name__ == "__main__":
    main()
