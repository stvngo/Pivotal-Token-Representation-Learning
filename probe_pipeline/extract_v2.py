"""Drive activation extraction from probe rows into a v2 cache.

One forward pass per branch, reading the residual stream at the labelled
positions only. The rows carry **token ids**, so nothing is re-tokenized here
-- that round trip is what shifted labels in the v1 pipeline
(``docs/issues.md`` Issue #9).

Typical use::

    rows, stats = build_probe_dataset(normalize_pts_rows(ds), tokenizer)
    train, test = split_by_query(rows)
    run_extraction(train, model, tokenizer, out_path="train.safetensors", ...)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from .activations_v2 import ActivationManifest, ActivationWriter
from .probe_dataset import ProbeRow


def resolve_layers(n_hidden_states: int, layers: Sequence[int] | None) -> list[int]:
    """Layers to keep, in ``outputs.hidden_states`` indexing.

    ``hidden_states`` has ``num_hidden_layers + 1`` entries: index 0 is the
    embedding output, index L is the output of decoder layer ``L-1``.
    """
    if layers is None:
        return list(range(n_hidden_states))
    bad = [L for L in layers if not 0 <= L < n_hidden_states]
    if bad:
        raise ValueError(
            f"layers {bad} out of range; hidden_states has {n_hidden_states} entries"
        )
    return sorted(set(int(L) for L in layers))


@torch.no_grad()
def run_extraction(
    rows: Sequence[ProbeRow],
    model: Any,
    tokenizer: Any,
    out_path: str | Path,
    *,
    split: str = "train",
    model_name: str = "",
    layers: Sequence[int] | None = None,
    dtype: str = "bfloat16",
    device: torch.device | str | None = None,
    max_seq_len: int | None = None,
    source_dataset: str = "",
    config_hash: str = "",
    logger: Any | None = None,
) -> dict[str, Any]:
    """Extract labelled-position activations for ``rows`` into one file.

    Returns a summary dict; the file itself carries a full manifest.
    """
    if not rows:
        raise ValueError("no rows to extract")

    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()

    # One probe forward pass to learn the shape of hidden_states.
    probe_ids = torch.tensor([rows[0].token_ids[:8]], dtype=torch.long, device=device)
    n_states = len(model(probe_ids, output_hidden_states=True).hidden_states)
    keep = resolve_layers(n_states, layers)

    hidden_size = int(model.config.hidden_size)
    manifest = ActivationManifest(
        model_name=model_name or getattr(model.config, "_name_or_path", "unknown"),
        n_layers=n_states,
        hidden_size=hidden_size,
        split=split,
        source_dataset=source_dataset,
        config_hash=config_hash,
        model_revision=str(getattr(model.config, "_commit_hash", "") or ""),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # hidden_states[-1] is post-final-norm for Llama/Qwen-style models, so the
    # LM head applies to it directly.
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("model has no output embeddings; cannot score uncertainty")

    writer = ActivationWriter(out_path, manifest, keep, dtype=dtype)
    n_truncated = 0
    started = time.time()

    for row in tqdm(rows, desc=f"Extracting ({split})"):
        token_ids = row.token_ids
        labels = row.labels
        if max_seq_len is not None and len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
            labels = labels[:max_seq_len]
            n_truncated += 1

        positions = [i for i, v in enumerate(labels) if v != 0]
        if not positions:
            continue

        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        # logits_to_keep=1 suppresses the full-sequence LM head. Materializing
        # (T, 151936) logits for every sequence dominated runtime -- 378 MB of
        # float32 for a 622-token branch -- and we only need a handful of rows.
        out = model(ids, output_hidden_states=True, logits_to_keep=1)

        # Next-token uncertainty at the labelled positions, computed by
        # applying the LM head to just those rows. These are the baselines the
        # probe has to beat, so capturing them here avoids a second pass.
        pos_t = torch.tensor(positions, dtype=torch.long, device=device)
        final_hidden = out.hidden_states[-1][0].index_select(0, pos_t)
        logits = lm_head(final_hidden).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(-1)
        top2 = probs.topk(2, dim=-1).values
        uncertainty = list(
            zip(
                entropy.cpu().tolist(),
                (top2[:, 0] - top2[:, 1]).cpu().tolist(),
                top2[:, 0].cpu().tolist(),
            )
        )

        writer.add_query(
            hidden_states=[out.hidden_states[L] for L in keep],
            positions=positions,
            labels=[labels[i] for i in positions],
            token_ids=[token_ids[i] for i in positions],
            prob_deltas=[row.prob_delta[i] for i in positions],
            query_id=f"{row.query_id}#b{row.branch}",
            uncertainty=uncertainty,
        )

    path = writer.close()
    elapsed = time.time() - started

    summary = {
        "path": str(path),
        "split": split,
        "n_rows": writer.n_rows,
        "n_sequences": len(rows),
        "n_layers_kept": len(keep),
        "hidden_size": hidden_size,
        "dtype": dtype,
        "max_abs_activation": writer.manifest.max_abs_activation,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "n_truncated": n_truncated,
        "seconds": round(elapsed, 1),
    }
    if logger:
        logger.info("Extraction complete: %s", summary)
    return summary
