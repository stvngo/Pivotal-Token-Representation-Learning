#!/usr/bin/env python
"""Generate the per-model Colab notebooks.

One notebook per model, so they can run concurrently in separate Colab
sessions (Colab allows several GPU instances but no true parallelism inside
one). Generated rather than hand-written so they cannot drift apart.

Each notebook is deliberately thin: a config cell, pull-from-HF, run,
push-to-HF. All logic lives in the repo so it is testable off-Colab and
behaves identically over SSH.

    python scripts/make_colab_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/stvngo/Pivotal-Token-Representation-Learning.git"
BRANCH = "rebuild/pipeline-and-scale"

MODELS = [
    # (tag, model id, max_examples, note)
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", 1500, "validation rung: also has released PTS events to check against"),
    ("qwen3-1.7b", "Qwen/Qwen3-1.7B", 1500, "middle rung of the ladder"),
    ("qwen3-4b", "Qwen/Qwen3-4B", 2000, "top rung; over-sample, more queries screen out on GSM8K"),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(tag: str, model: str, max_examples: int, note: str) -> dict:
    cells = [
        md(
            f"""# PTS generation - `{model}`

{note}

Runs token-granularity Pivotal Token Search and then extracts probe
activations, both checkpointed to a HuggingFace dataset repo. **Safe to
interrupt**: re-running this notebook resumes from whatever is already on
the Hub and recomputes nothing.

To split one model across two Colab sessions, set `SHARD` to 0 in one and 1
in the other, with `NUM_SHARDS = 2`. They need no coordination.
"""
        ),
        code(
            """# GPU check first -- an L4 works but is ~3.5x slower than an A100.
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"""
        ),
        code(
            f"""%pip install -q vllm
!git clone -q --branch {BRANCH} {REPO} /content/ptrl || (cd /content/ptrl && git pull -q)
%cd /content/ptrl
%pip install -q -r requirements.txt"""
        ),
        code(
            f'''# ---- config -------------------------------------------------------
MODEL         = "{model}"
TAG           = "{tag}"
MAX_EXAMPLES  = {max_examples}   # candidate questions; ~40% survive screening
NUM_SAMPLES   = 40               # rollouts per probability estimate
MAX_NEW_TOK   = 320
MAX_GEN       = 1                # rollouts searched per question
MAX_ACTIVE    = 64               # queries in flight; this is what fills the GPU
SHARD, NUM_SHARDS = 0, 1

HF_REPO       = "USERNAME/ptrl-runs"   # <-- set me: a private dataset repo
RUN_DIR       = f"runs/{{TAG}}"
# -------------------------------------------------------------------
import os; os.environ["HF_HOME"] = "/content/hf_cache"'''
        ),
        code(
            """# Token from Colab Secrets (key: HF_TOKEN). Never paste one into a cell --
# notebook outputs are committed and a pasted token leaks.
from probe_pipeline.artifacts_io import resolve_hf_token
from huggingface_hub import HfApi
token = resolve_hf_token(required=True)
api = HfApi(token=token)
api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
print("hub ok")"""
        ),
        md("## Resume: pull whatever this run already produced"),
        code(
            """from huggingface_hub import snapshot_download
from pathlib import Path
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=".",
                      allow_patterns=[f"{RUN_DIR}/**"], token=token)
    print("pulled existing state")
except Exception as e:
    print("nothing to resume:", type(e).__name__)

from pts_harness.checkpoint import RunStore
store = RunStore(RUN_DIR)
print(f"{len(store.completed_ids())} queries already done, "
      f"{len(store.load_prob_cache())} cached estimates")"""
        ),
        md(
            """## Search

Cost is dominated by generated tokens: roughly
`Q*S*L + Q*f*G*B*S*L`, with `f` the fraction of questions inside the
`[min-prob, max-prob]` band and `B` the unique bisection midpoints per
rollout. The run reports both measured values in its summary, which
replace the planning estimates."""
        ),
        code(
            """!python scripts/pts_run.py \\
    --model $MODEL --backend vllm \\
    --max-examples $MAX_EXAMPLES --num-samples $NUM_SAMPLES \\
    --max-new-tokens $MAX_NEW_TOK --max-generations $MAX_GEN \\
    --max-active $MAX_ACTIVE \\
    --shard $SHARD --num-shards $NUM_SHARDS \\
    --out $RUN_DIR"""
        ),
        code(
            """# Push state after the search, before doing anything else.
api.upload_folder(folder_path=RUN_DIR, path_in_repo=RUN_DIR,
                  repo_id=HF_REPO, repo_type="dataset")
import json; print(json.dumps(json.load(open(f"{RUN_DIR}/summary.json")), indent=1))"""
        ),
        md(
            """## Extraction

Cheap next to the search -- minutes, not hours -- so it runs in the same
session rather than needing its own GPU booking."""
        ),
        code(
            """!python scripts/build_and_extract.py \\
    --model $MODEL --pts-events $RUN_DIR/events.jsonl \\
    --tag $TAG --out data/acts_v2 --device cuda --dtype bfloat16"""
        ),
        code(
            """api.upload_folder(folder_path="data/acts_v2", path_in_repo="acts_v2",
                  repo_id=HF_REPO, repo_type="dataset")
print("done -- probe training runs on CPU, no GPU needed from here")"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for tag, model, n, note in MODELS:
        path = out_dir / f"pts_generate_{tag}.ipynb"
        path.write_text(json.dumps(notebook(tag, model, n, note), indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
