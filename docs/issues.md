# Open issues and methodological concerns

This document catalogs problems and concerns identified during a verification pass
of the pivotal-token probing and steering pipeline. Each entry cites the specific
file and line where the issue lives, explains why it matters, and proposes a
concrete remediation. None of these are blocking the existing experiments from
producing data, but several invalidate or reduce the strength of the *causal
claims* we can make from that data.

Severity legend:

- **Blocking** — must be fixed before drawing absolute conclusions about steering
  effect sizes; relative comparisons across arms are still meaningful.
- **High** — can shift results materially; should be addressed before the next
  round of experiments.
- **Medium** — improves rigor; not currently producing wrong answers.
- **Low** — hygiene / future-proofing.

## 1. Qwen3-0.6B layer reference (verified)

Source verified against the local cached Qwen3-0.6B at
`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/`
and the Qwen3 modeling source in
[`venv/.../transformers/models/qwen3/modeling_qwen3.py`](../venv/lib/python3.13/site-packages/transformers/models/qwen3/modeling_qwen3.py).

Key facts:

- `num_hidden_layers = 28` (per `config.json`).
- `hidden_size = 1024`, `head_dim = 128`, `num_attention_heads = 16`,
  `num_key_value_heads = 8` (GQA).
- `model.model.layers` is an `nn.ModuleList` of `28` `Qwen3DecoderLayer`
  instances, indexed `0..27`. There is no `model.model.layers[28]`.
- After all blocks, `Qwen3Model.forward` applies `self.norm = Qwen3RMSNorm(...)`
  to produce `last_hidden_state`.
- `Qwen3PreTrainedModel` declares
  `_can_record_outputs = {"hidden_states": Qwen3DecoderLayer}`. Combined with the
  `@capture_outputs` decorator
  ([`venv/.../transformers/utils/output_capturing.py:206`](../venv/lib/python3.13/site-packages/transformers/utils/output_capturing.py)),
  this produces the following `outputs.hidden_states` tuple of length 29 when
  `output_hidden_states=True`:

  | Index `k` | Tensor identity | What this is |
  | --- | --- | --- |
  | `0` | `args[0]` of the first `Qwen3DecoderLayer.forward` | post-`embed_tokens`, pre-`layers[0]` |
  | `1` to `27` | `output[0]` of `layers[k-1]` | post-`layers[k-1]`, pre-`layers[k]` (the residual stream that *feeds into* `layers[k]`) |
  | `28` | overwritten by `last_hidden_state` (`tie_last_hidden_states=True`) | post-`layers[27]` AND post-final `RMSNorm` |

- The capture mechanism is in
  [`venv/.../transformers/utils/output_capturing.py`](../venv/lib/python3.13/site-packages/transformers/utils/output_capturing.py)
  lines 103-120 (forward hook) and 260-270 (last-state tying).
- Important consequence of the tying: **`hidden_states[28]` is post-RMSNorm, but
  `hidden_states[1..27]` are pre-norm raw block outputs.** The last cached layer
  is therefore not interchangeable with the others.

This is the reference model layout used by every issue below.

## 2. Off-by-one between probe layer and steering hook (Blocking)

**Where**

- [`probe_pipeline/activations.py:96-115`](../probe_pipeline/activations.py)
- [`probe_pipeline/steering.py:45-91`](../probe_pipeline/steering.py)
- [`scripts/smoke_colab_notebook.py:86-117`](../scripts/smoke_colab_notebook.py)
- All `notebooks/steering_*.ipynb` that import these helpers.

**Symptom**

A steering vector trained against `hidden_states[L]` is being added to
`hidden_states[L+1]`, i.e. one transformer block downstream of where the probe
was learned.

**Trace**

In [`probe_pipeline/activations.py`](../probe_pipeline/activations.py):

```96:99:probe_pipeline/activations.py
            for layer_num, hidden_states_layer in enumerate(all_hidden_states):
                hidden_states = hidden_states_layer.squeeze(0)
                min_len = min(hidden_states.shape[0], len(labels))
                if hidden_states.shape[0] != len(labels) and logger:
```

`layer_num = 14` ↔ `all_hidden_states[14]` = output of `model.layers[13]` =
input to `model.layers[14]`. Per the table in §1.

In [`probe_pipeline/steering.py`](../probe_pipeline/steering.py):

```45:53:probe_pipeline/steering.py
def _get_decoder_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """Get the decoder layer at index layer_idx. Handles common HuggingFace layouts."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
```

```86:91:probe_pipeline/steering.py
    decoder_layer = _get_decoder_layer(model, layer)
    hook_fn = _make_steering_hook(vector, strength, device)
    handle = decoder_layer.register_forward_hook(hook_fn)
    return [handle]
```

`register_forward_hook` on `model.layers[14]` modifies that layer's *output*,
which is `hidden_states[15]`. So when callers say "steer at layer 14", the
steering vector is added to the residual stream that feeds into
`model.layers[15]`, not the stream the probe was trained on.

**Why ARENA's GoT pattern is internally consistent and ours isn't**

[`docs/arena_linear_probes.md`](arena_linear_probes.md) line 401 uses the
convention `outputs.hidden_states[layer + 1]` to read "layer L's output", so
their "layer L" is `hidden_states[L+1]` = output of `layers[L]`. They then hook
`model.model.layers[layer_idx]` (line 1832, 2644) and modify that same tensor.
Both ends agree.

We use the convention `outputs.hidden_states[L]` for "layer L" in extraction but
hook `model.layers[L]` for steering. The two ends use the same numeric label for
two different tensors.

**Why effects were still visible**

Adjacent residual-stream tensors are highly correlated; the steering vector
remains partially aligned with the right subspace. This is consistent with the
empirical pattern we have already observed: CAA (centroid_diff) > probe_weights
> random for GSM8k accuracy lift, with all magnitudes smaller than what
GoT-style experiments report.

**Affected experiments**

Every steering run from CAA, probe_weights, random control, layer sweep,
greedy, additive multilayer, and codelion. The relative ordering across arms is
still meaningful because the bug applies uniformly. Absolute effect sizes are
likely understated.

**Remediation**

Pick one convention and apply it to both extraction and steering.

Option A (smaller code change, recommended): keep the extraction convention
("layer L" = `hidden_states[L]`) and fix steering.

```python
def _get_decoder_layer(model, layer_idx):
    if layer_idx == 0:
        # hook on the embedding to modify hidden_states[0]
        return model.model.embed_tokens
    return model.model.layers[layer_idx - 1]
```

…with a corresponding pre-hook variant when modifying the input rather than the
output, or alternatively use `register_forward_pre_hook` on
`model.layers[layer_idx]` (which receives the input = `hidden_states[layer_idx]`
for `layer_idx >= 1`).

Option B (larger code change): adopt the ARENA convention and store cached
activations under the new label scheme; this requires re-extracting or
re-numbering the cached activations. Not recommended given existing artifacts.

Whichever path, add a smoke test that asserts:

```python
captured = {}

def _save(_m, _i, out):
    captured["pre"] = out[0].detach()

h = model.model.layers[layer_idx - 1].register_forward_hook(_save)
out = model(**inputs, output_hidden_states=True)
h.remove()
assert torch.allclose(captured["pre"], out.hidden_states[layer_idx])
```

so the convention is locked in.

## 3. LR regularization is below community defaults (High)

**Where**

[`probe_pipeline/train_sklearn.py:20-30`](../probe_pipeline/train_sklearn.py).

```20:30:probe_pipeline/train_sklearn.py
def _build_estimator(cfg: dict[str, Any]) -> Pipeline | LogisticRegression:
    model = LogisticRegression(
        solver=cfg.get("solver", "saga"),
        max_iter=int(cfg.get("max_iter", 3000)),
        C=float(cfg.get("C", 1.0)),
        random_state=int(cfg.get("random_state", 42)),
        class_weight=cfg.get("class_weight"),
    )
```

Default `C=1.0`. No config in `configs/` overrides `sklearn.C`.

**What other work uses**

- ARENA Geometry of Truth probe ([arena_linear_probes.md](arena_linear_probes.md)
  line 952): `C=0.1` with `fit_intercept=False`.
- ARENA Apollo deception probe ([arena_linear_probes.md](arena_linear_probes.md)
  line 2370): `C=0.001`.

**Why it matters**

[`docs/arena_linear_probes.md`](arena_linear_probes.md) lines 1957-1959 are
explicit:

> *"In some cases, however, the direction identified by LR can fail to reflect
> an intuitive best guess for the feature direction, even in the absence of
> confounding features."* … *"high probe accuracy does not guarantee causal
> relevance"*.

Weakly-regularized LR is the regime that picks up small correlational features
that classify well but aren't causally implicated in computation. `prob_delta`
is itself estimated from rollouts (sampling noise), so the `is_pivotal` target
contains label noise that under-regularized LR can latch onto.

**Likely manifestation in our results**

Probe-weight steering had smaller effects than centroid_diff steering. This is
exactly the GoT pattern (LR < MM for causal effect) that the doc predicts. With
`C=0.1` or `C=0.01`, the LR direction is expected to look more like the
centroid_diff direction (cosine similarity should rise) and to produce stronger
causal effects on intervention.

**Remediation**

- Add `C` to the layer-14 sweep: train probes at `C ∈ {0.001, 0.01, 0.1, 1.0}`.
- Report cosine similarity between the LR weights and the
  `centroid_diff = mu_pos - mu_neg` direction at each `C`.
- Pick whichever `C` maximizes either (a) cosine similarity to centroid_diff or
  (b) the per-token NIE described in §10.
- Bump the default in [`probe_pipeline/train_sklearn.py`](../probe_pipeline/train_sklearn.py)
  to `C=0.1` once the sweep confirms it's at least as good. Cheap because the
  cache is already built.

## 4. Always-on additive steering at every position (High)

**Where**

[`probe_pipeline/steering.py:56-77`](../probe_pipeline/steering.py).

```56:77:probe_pipeline/steering.py
def _make_steering_hook(
    vector: torch.Tensor,
    strength: float,
    device: torch.device,
) -> Callable[..., Any]:
    """Create a forward hook that adds strength * vector to the layer output."""

    def hook(module: nn.Module, args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # hidden: (batch, seq, hidden_dim)
        v = vector.to(hidden.device).to(hidden.dtype)
        if v.dim() == 1:
            v = v.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        delta = strength * v
        if isinstance(output, tuple):
            return (hidden + delta,) + output[1:]
        return hidden + delta
```

The vector is added to every position of every prompt during prefill, and to
every newly generated token during decode.

**Why it matters**

ARENA's GoT intervention experiment patches at exactly two positions per
sequence ([arena_linear_probes.md](arena_linear_probes.md) line 1815-1827): the
final period of the statement, and "This" of the appended " This statement is:"
suffix. ARENA's deception steering hook
([arena_linear_probes.md](arena_linear_probes.md) line 2629-2641) supports an
`apply_to_all_tokens=False` mode that only perturbs the last position.

We perturb every position. This:

- Dilutes the localized effect at the actual decision point.
- Pushes irrelevant prompt tokens off-manifold, causing a fluency tax that
  reduces sample quality (the codelion run showed degraded outputs at higher
  factors).
- Makes the steering effect harder to attribute causally to the probe direction
  because the perturbation is smeared.

**Status**

Already documented as a known limitation in
[`docs/pivotal_probes.md`](pivotal_probes.md) §4 ("Reactive / token-conditional
steering") as the motivation for the planned reactive-steering notebook. Not
yet implemented.

**Remediation**

Implement the reactive hook described in
[`docs/pivotal_probes.md`](pivotal_probes.md) §4: at each generation step, run
the probe on the latest residual, gate steering by `P(pivotal) > τ`, and apply
the vector only to the new token's hidden state. ARENA's batch-aware hook
([arena_linear_probes.md](arena_linear_probes.md) line 1815-1827) is a useful
template for the position-selection logic.

## 5. Centroid_diff is computed on validation activations (Medium)

**Where**

- [`probe_pipeline/train_sklearn.py:103-104`](../probe_pipeline/train_sklearn.py)
- [`probe_pipeline/steering.py:21-33`](../probe_pipeline/steering.py)

```103:104:probe_pipeline/train_sklearn.py
        np.save(layer_dir / f"activations_layer_{layer_num}.npy", x_val)
        np.save(layer_dir / "labels.npy", y_val.astype(np.float32))
```

```26:33:probe_pipeline/steering.py
        x = np.load(x_path).astype(np.float32)
        y = np.load(y_path).reshape(-1)
        mask_pos = y >= 0.5
        mask_neg = y < 0.5
        mu_pos = x[mask_pos].mean(axis=0)
        mu_neg = x[mask_neg].mean(axis=0)
        diff = mu_pos - mu_neg
```

The MM steering vector is constructed from 65 validation activations, not 232
training activations.

**Why it matters**

- It's a methodological inconsistency with how the MM probe is normally
  defined ([arena_linear_probes.md](arena_linear_probes.md) line 919-928 uses
  `pos_acts.mean(0) - neg_acts.mean(0)` over training data).
- Smaller sample → higher variance in `mu_pos` and `mu_neg` → noisier direction.
- Not a label leak (we don't use validation labels to fit anything we then
  evaluate against), but the convention should match the literature.

**Remediation**

Save both `x_train`/`y_train` and `x_val`/`y_val` to `analysis_data/layer_X/`
and rebuild the centroid_diff from training data. The vectors saved into
`steering_configs/` should be re-derived from `train_acts` so historical
runs can be re-overlaid against a corrected baseline.

## 6. Single-layer steering rather than intervene→probe range (Medium)

**Where**

`probe_pipeline/steering.py` registers exactly one hook per call. All
notebooks except [`notebooks/steering_additive_multilayer.ipynb`](../notebooks/steering_additive_multilayer.ipynb)
hook a single layer.

**What ARENA does**

[`docs/arena_linear_probes.md`](arena_linear_probes.md) line 1688:

```python
# Intervene at all layers from INTERVENE_LAYER through PROBE_LAYER. This matches
# the paper's "group (b)" hidden states that were found to be causally implicated.
intervene_layer_list = list(range(INTERVENE_LAYER, PROBE_LAYER + 1))
```

GoT patches at layers 8 through 14 inclusive — *earlier* than the probe layer,
so downstream layers can actually use the modified residual stream when forming
the prediction.

**Why it matters**

Patching only at the probe layer (or later, due to issue #2) gives downstream
layers no opportunity to react to the perturbation. The model's output decision
is formed several layers later, so the intervention has to propagate.

**Status**

`steering_additive_multilayer.ipynb` exists but uses *neighbouring-layer
clusters* around the probe layer, not the GoT "from intervene to probe"
inclusive range pattern.

**Remediation**

Add a notebook arm that uses
`intervene_layer_list = list(range(intervene_layer, probe_layer + 1))` with a
small grid for `intervene_layer ∈ {probe_layer-2, probe_layer-4, probe_layer-6}`.
Apply the same vector at every layer in the range (or scale per-layer if the
norm-scaling fix in #7 is also adopted). Compare against single-layer steering
on the same examples.

## 7. No norm-scaled steering convention (Medium)

**Where**

All steering hooks (`probe_pipeline/steering.py`,
`scripts/smoke_colab_notebook.py`, all notebooks except the projection-scaling
arm in [`notebooks/steering_probe_weights.ipynb`](../notebooks/steering_probe_weights.ipynb)).

**What ARENA does**

[`docs/arena_linear_probes.md`](arena_linear_probes.md) line 2629-2641:

```python
def _hook_fn(self, module, input, output):
    hidden_states = output[0] if isinstance(output, tuple) else output
    v = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
    v_normed = v / (v.norm() + 1e-8)
    if self.apply_to_all_tokens:
        norm = t.norm(hidden_states, dim=-1, keepdim=True)
        hidden_states = hidden_states + self.steering_coef * norm * v_normed
```

The perturbation is `coef * ||h_pos|| * v̂`, scaled to the per-position
activation magnitude.

**Why it matters**

- Typical residual-stream norms differ by 10–100× across layers. With raw
  additive steering, the same `factor=1.4` produces very different perturbation
  magnitudes at different layers, which makes the layer sweep noisy and hard
  to interpret.
- Norm-scaling makes `coef` comparable across layers and across models.

**Remediation**

Add a `mode` parameter to the hook factory so callers can pick:

- `additive_raw` (current behavior, kept for back-compat).
- `additive_normalized` (`alpha * ||h|| * v̂`, ARENA's deception convention).
- `projection_scaling` (already exists in
  [`notebooks/steering_probe_weights.ipynb`](../notebooks/steering_probe_weights.ipynb)
  and `scripts/smoke_colab_notebook.py`).

Re-run the layer sweep under `additive_normalized` to get an interpretable
layer×coef plot.

## 8. Padding handling: currently safe, fragile under future batching (Low)

**Verification**

All current paths run with `batch_size=1` and therefore never trigger
padding:

- [`probe_pipeline/preprocess.py:73`](../probe_pipeline/preprocess.py): single
  text in `tokenizer(longest_text, return_tensors="pt", add_special_tokens=False)`.
- [`probe_pipeline/activations.py:92`](../probe_pipeline/activations.py): single
  text per loop iteration.
- [`probe_pipeline/steering.py:118`](../probe_pipeline/steering.py): single
  prompt for `generate_with_steering`.
- [`probe_pipeline/gsm8k_eval.py:112`](../probe_pipeline/gsm8k_eval.py): single
  prompt for both base and steered eval.
- [`scripts/smoke_colab_notebook.py:201`](../scripts/smoke_colab_notebook.py):
  single prompt.

The Qwen3 tokenizer config sets `"pad_token": "<|endoftext|>"` (token 151643)
out of the box, and [`probe_pipeline/modeling.py:33-34`](../probe_pipeline/modeling.py)
defensively re-aliases `pad_token = eos_token` if missing. All
`model.generate` calls pass `pad_token_id=tokenizer.pad_token_id or
tokenizer.eos_token_id`. None of the current code is exercising the padding
path, so there is no padding-related bug today.

**Where it becomes fragile**

If we ever batch GSM8k generation or batch the activation extraction for
speed, three things must be addressed simultaneously:

1. Use `padding=True` in the tokenizer call and capture
   `inputs["attention_mask"]`.
2. Use `tokenizer.padding_side = "left"` for batched `model.generate` (Qwen3
   is a causal LM; right-padding breaks autoregressive decoding because the
   real last-token isn't actually last).
3. In [`probe_pipeline/activations.py:96-115`](../probe_pipeline/activations.py),
   when iterating positions, only consider positions `idx` where
   `attention_mask[b, idx] == 1`, and align labels to *real* token positions
   (not the padded positions). Right-padding puts pads at the end (real tokens
   at indices `[0, end)` where `end = attention_mask.sum()`); left-padding puts
   them at the start (real tokens at `[seq_len - end, seq_len)`).
4. The steering hook in [`probe_pipeline/steering.py:62-77`](../probe_pipeline/steering.py)
   adds the vector at every position, including padded positions. With
   left-padding for generation, the perturbation on padded positions is
   masked out by the attention layer downstream so it's wasteful but not
   incorrect; with right-padding for inference, perturbing future positions
   would be similarly absorbed but is still wasteful and obscures the
   intended targeting.

**Remediation (preventive)**

Add an `attention_mask`-aware iteration block to
`extract_and_label_all_layers` *now*, gated behind a `batch_size > 1` path so
the single-sequence behavior is unchanged. Same for the steering hook: when
`attention_mask` is available, multiply the delta by the mask to only perturb
real positions.

## 9. Tokenizer round-trip can shift labels (Low)

**Where**

- Label construction: [`probe_pipeline/preprocess.py:43-101`](../probe_pipeline/preprocess.py).
  Builds `final_ids` then *decodes* them into `text` for storage:
  `text = tokenizer.decode(final_ids, skip_special_tokens=True)`.
- Activation extraction: [`probe_pipeline/activations.py:92`](../probe_pipeline/activations.py).
  Re-encodes via `tokenizer(text, return_tensors="pt", add_special_tokens=False)`.

**Why it matters**

Decode→encode is not always idempotent for BPE tokenizers, especially when
`skip_special_tokens=True` strips tokens like `<think>`, `<|endoftext|>` etc.
that may have been part of `final_ids`. If the round-trip produces a different
token count than `len(labels)`, the existing safeguard at
[`probe_pipeline/activations.py:99-107`](../probe_pipeline/activations.py)
silently truncates to `min_len`:

```96:108:probe_pipeline/activations.py
            for layer_num, hidden_states_layer in enumerate(all_hidden_states):
                hidden_states = hidden_states_layer.squeeze(0)
                min_len = min(hidden_states.shape[0], len(labels))
                if hidden_states.shape[0] != len(labels) and logger:
                    logger.warning(
                        "Layer %s mismatch for query %s: tokens=%s labels=%s -> truncating=%s",
                        layer_num,
                        sample_id,
                        hidden_states.shape[0],
                        len(labels),
                        min_len,
                    )
```

A mismatch of even 1 token shifts the t-1 alignment for the affected query —
the labels-1 position would no longer correspond to the residual stream just
before the pivot.

**Remediation**

- Store `final_ids` directly in the probe dataset (alongside `labels` and the
  decoded `text`), and use `tokenizer.prepare_for_model` /
  pre-tokenized inputs at extraction time instead of re-encoding from the
  decoded string. This eliminates the round-trip ambiguity.
- Until then, count and surface the warning rate from training runs. If
  the rate is non-trivial (>1% of queries), the affected rows should be
  excluded from the cache rather than silently truncated.

## 10. No per-token NIE metric (Medium)

**Where**

We currently measure GSM8k accuracy lift end-to-end in
[`probe_pipeline/gsm8k_eval.py`](../probe_pipeline/gsm8k_eval.py). There is no
metric that isolates the steering vector's effect on a *single* model decision.

**What ARENA does**

[`docs/arena_linear_probes.md`](arena_linear_probes.md) line 1865:

> *"the NIE for "add" on false statements = P_diff(add) - P_diff(none)"*

Natural Indirect Effect on a clean two-class logit decision (P(TRUE) − P(FALSE)
under intervention vs. without).

**Why it matters**

GSM8k accuracy is the right deployment metric but a noisy causality metric:
the path from steering vector to final accuracy goes through hundreds of token
decisions and many intervening computations. A per-token NIE on labelled
pivots gives a clean signal that:

- Confirms or refutes the GoT prediction (MM > LR for causal effect).
- Is much cheaper to compute than full GSM8k generation.
- Is interpretable per layer, per coefficient, per probe type.

**Remediation**

Add a small notebook that, for each labelled pivot position `t-1` in the
held-out set:

1. Computes `P(actual_pivot_token | prefix)` with no steering.
2. Computes the same probability with the steering vector applied at
   `model.layers[layer-1]` (post the §2 fix) at that single position.
3. Reports `NIE = log P(steered) - log P(base)` averaged across pivots, broken
   down by sign of `prob_delta` (positive vs. negative pivots).

This becomes the primary causal-effect metric for the signed-probe and
reactive-steering experiments to follow.

## 11. Implications for the signed-pivotal relabel plan

The plan in `signed-pivotal-relabel-plan_d3eda80e.plan.md` reuses the existing
extract-and-hook stack unchanged. Before running it, fold in:

- Fix §2 (off-by-one) so probe 2 is trained and applied at the same residual
  stream tensor.
- Sweep `C` per §3 instead of fixing `C=1.0` for the signed probe.
- Train *both* MM (`mu_pos − mu_neg`) and LR (probe weights) for the signed
  task and save both direction vectors. The Geometry of Truth finding (MM more
  causally implicated) is testable on Qwen3 directly via §10's NIE.
- Use training activations for centroid_diff (§5).
- Add the `attention_mask`-aware extraction code path (§8) before scaling to
  more queries; the signed cache is a natural place to introduce batching.

## 12. Summary table

| # | Severity | Area | One-line fix |
| --- | --- | --- | --- |
| 2 | Blocking | Steering hook off-by-one | hook `model.layers[L-1]` (or pre-hook on `model.layers[L]`) |
| 3 | High | LR `C` too large | sweep `C ∈ {0.001, 0.01, 0.1, 1.0}`, default to 0.1 |
| 4 | High | Always-on steering | reactive token-conditional hook (already planned) |
| 5 | Medium | centroid_diff on val data | compute from train activations |
| 6 | Medium | Single-layer steering | add `range(intervene, probe + 1)` arm |
| 7 | Medium | No norm-scaled steering | add `additive_normalized` mode |
| 8 | Low | Padding fragile under future batching | add `attention_mask` path now |
| 9 | Low | Decode/encode round-trip | persist `final_ids`, drop re-encode |
| 10 | Medium | No per-token NIE | small NIE notebook for clean causal effect |
