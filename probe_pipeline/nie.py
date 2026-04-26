"""Per-token Natural Indirect Effect (NIE) for steering vectors.

Closes Issue #10 from docs/issues.md ("No per-token NIE metric"). GSM8K
accuracy lift is the right deployment metric, but it routes the steering
signal through hundreds of token decisions and many intervening computations.
NIE evaluates the *single-token* causal effect of a steering vector at exactly
the labelled t-1 position the probe was learned on.

For each labelled position ``p`` (where ``labels[p] == 1`` -- i.e. ``p`` is
the residual immediately preceding a pivotal token in the canonical sequence):

1. ``log_p_base = log_softmax(model(ids).logits[p])[ids[p+1]]``
2. ``log_p_steered = log_softmax(model_with_hook(ids).logits[p])[ids[p+1]]``
   where the hook applies the steering vector ONLY at position ``p``
   (``position_mask`` selects index ``p``).
3. ``nie_p = log_p_steered - log_p_base``.

Aggregate ``nie_p`` across positions/rows. Positive NIE means the steering
vector *increases* the model's probability of producing the actual pivot token;
negative NIE means it suppresses it.

The hook target is ``model.model.layers[layer-1]`` (post-Issue-#2 fix), so the
perturbed residual is exactly ``outputs.hidden_states[layer]`` -- the same
tensor the probe was trained on.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .steering import _get_decoder_layer, make_hook


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compute_token_nie(
    model: nn.Module,
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    layer: int,
    vector: np.ndarray | torch.Tensor,
    coef: float,
    mode: str = "additive_normalized",
    device: torch.device | str = "cpu",
    label_value: int = 1,
    max_rows: int | None = None,
    max_positions_per_row: int | None = None,
) -> dict[str, Any]:
    """Compute per-token NIE for ``vector`` at the labelled positions.

    Args:
        rows: iterable of ``{"text": str, "labels": list[int],
            "original_dataset_item_id": Any}`` (the schema produced by
            ``probe_pipeline.preprocess.create_probe_dataset``).
        layer: probe layer (post-Issue-#2 fix; hook attaches to
            ``layers[layer-1]``).
        vector: ``(hidden_dim,)`` steering direction.
        coef: scalar steering coefficient. Interpretation matches ``make_hook``
            (e.g. for ``additive_normalized`` it is the per-position fraction
            of ``||h||`` to add).
        mode: ``additive_raw`` / ``additive_normalized`` / ``projection``.
        label_value: which label to evaluate. ``1`` = pivotal positions
            (default). ``-1`` would be the matched non-pivotal control.
        max_rows / max_positions_per_row: optional caps for smoke testing.

    Returns a dict with per-row NIE samples plus aggregate statistics:
        {
          "n_rows": int, "n_positions": int,
          "nie_mean": float, "nie_std": float, "nie_median": float,
          "bootstrap_ci": [lo, hi],
          "log_p_base_mean": float, "log_p_steered_mean": float,
          "samples": [{row_id, position, pivot_token_id,
                       log_p_base, log_p_steered, nie}, ...]
        }
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device
    if isinstance(vector, np.ndarray):
        v = torch.tensor(vector, dtype=torch.float32)
    else:
        v = vector.detach().to(torch.float32)
    v = v.reshape(-1)

    target = _get_decoder_layer(model, layer)

    samples: list[dict[str, Any]] = []
    n_rows_processed = 0

    with torch.no_grad():
        for row in rows:
            if max_rows is not None and n_rows_processed >= max_rows:
                break
            text = row["text"]
            labels = list(row["labels"])
            row_id = row.get("original_dataset_item_id", n_rows_processed)

            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            ids = inputs["input_ids"][0].tolist()
            ids_t = inputs["input_ids"].to(device)
            seq_len = len(ids)
            if seq_len <= 1:
                n_rows_processed += 1
                continue

            usable = min(len(labels), seq_len)
            positions = [p for p in range(usable - 1) if labels[p] == label_value]
            if max_positions_per_row is not None:
                positions = positions[:max_positions_per_row]
            if not positions:
                n_rows_processed += 1
                continue

            base_out = model(ids_t)
            base_logits = base_out.logits[0]
            base_logp = F.log_softmax(base_logits.float(), dim=-1)

            for p in positions:
                pivot_id = int(ids[p + 1])
                lp_base = float(base_logp[p, pivot_id].item())

                mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
                mask[p] = True
                hook_fn = make_hook(v, coef, mode=mode, position_mask=mask)
                handle = target.register_forward_hook(hook_fn)
                try:
                    steer_out = model(ids_t)
                finally:
                    handle.remove()
                steer_logits = steer_out.logits[0]
                steer_logp = F.log_softmax(steer_logits.float(), dim=-1)
                lp_steer = float(steer_logp[p, pivot_id].item())

                samples.append({
                    "row_id": str(row_id),
                    "position": int(p),
                    "pivot_token_id": pivot_id,
                    "log_p_base": lp_base,
                    "log_p_steered": lp_steer,
                    "nie": lp_steer - lp_base,
                })
            n_rows_processed += 1

    if not samples:
        return {
            "n_rows": int(n_rows_processed),
            "n_positions": 0,
            "nie_mean": float("nan"),
            "nie_std": float("nan"),
            "nie_median": float("nan"),
            "bootstrap_ci": [float("nan"), float("nan")],
            "log_p_base_mean": float("nan"),
            "log_p_steered_mean": float("nan"),
            "samples": [],
        }

    nie_arr = np.asarray([s["nie"] for s in samples], dtype=np.float64)
    base_arr = np.asarray([s["log_p_base"] for s in samples], dtype=np.float64)
    steer_arr = np.asarray([s["log_p_steered"] for s in samples], dtype=np.float64)
    lo, hi = _bootstrap_ci(nie_arr)

    return {
        "n_rows": int(n_rows_processed),
        "n_positions": int(len(samples)),
        "nie_mean": float(nie_arr.mean()),
        "nie_std": float(nie_arr.std(ddof=1)) if len(nie_arr) > 1 else 0.0,
        "nie_median": float(np.median(nie_arr)),
        "bootstrap_ci": [lo, hi],
        "log_p_base_mean": float(base_arr.mean()),
        "log_p_steered_mean": float(steer_arr.mean()),
        "samples": samples,
    }


def cosine(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    """Helper used by nie_eval.ipynb for the cosine-vs-NIE scatter."""
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / n)
