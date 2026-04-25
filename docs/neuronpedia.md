# Neuronpedia API reference

Working notes on Neuronpedia's API as a possible substitute for training our
own SAE in the SAE-projected probe-steering recipe (`v_steer = W_dec @ w_probe`).

> Source of truth: <https://www.neuronpedia.org/api-doc>. This file just
> condenses the subset relevant to PTS / activation-steering experiments and
> spells out the integration points with our existing pipeline.

## What Neuronpedia is

Neuronpedia hosts:

1. **Pretrained SAE features** for a curated set of base models. Each feature
   has a stored decoder column, top-activating tokens, and (often) an
   auto-generated explanation.
2. **An activation API** that lets you submit your own text and read back
   feature activations or residual-stream channel activations.
3. **A hosted steering API** (`/api/steer*`) that takes an existing prompt
   plus a list of `(modelId, layer, feature_index, strength)` triples and
   returns both the unsteered and steered completions, with logprobs.
4. **A user-vector store** (`/api/vector/*`) for uploading custom direction
   vectors and steering with them via the same endpoints.

Useful angles for our project:

- **Substitute for training a Qwen-specific SAE**: if Qwen3 (or a close
  proxy) is hosted, we can prototype `v_steer = W_dec @ w_probe` by querying
  feature decoder vectors instead of training our own.
- **Selective / per-prompt steering**: `/api/steer` already supports
  query-conditional steering — closer to codelion/optillm's intended usage
  than our always-on hook.
- **Sanity-check our pivotal tokens**: `/api/feature/{model}/{layer}/{idx}`
  returns top-activating tokens; we can spot-check whether features that
  fire on our pivot-token list match the auto-explanations.

## Endpoints we care about

All paths below are relative to `https://www.neuronpedia.org`. Anchors point
at the corresponding entry in the public API doc.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/activation/new` | Get activation values for a `(modelId, source, index)` over arbitrary input text. ([doc](https://www.neuronpedia.org/api-doc#tag/activations/POST/api/activation/new)) |
| `POST` | `/api/activation/get` | Return all stored activations for a `(modelId, source, index)`. ([doc](https://www.neuronpedia.org/api-doc#tag/activations/POST/api/activation/get)) |
| `GET`  | `/api/feature/{modelId}/{layer}/{index}` | Feature page payload: top-activating tokens, decoder vector (sometimes), explanations. ([doc](https://www.neuronpedia.org/api-doc#tag/features/GET/api/feature/{modelId}/{layer}/{index})) |
| `GET`  | `/api/sparsity/connected-neurons` | Connected residual-stream channels with top-2 explanations each (graph-style links between features). ([doc](https://www.neuronpedia.org/api-doc#tag/sparsity/GET/api/sparsity/connected-neurons)) |
| `POST` | `/api/steer` | Steer a *completion* with a feature set; returns default + steered text and per-token logprobs. ([doc](https://www.neuronpedia.org/api-doc#tag/steering/POST/api/steer)) |
| `POST` | `/api/steer-chat` | Same as above, for *chat* messages instead of raw completions. ([doc](https://www.neuronpedia.org/api-doc#tag/steering/POST/api/steer-chat)) |
| `POST` | `/api/vector/new` | Upload a custom direction vector. ([doc](https://www.neuronpedia.org/api-doc#tag/vectors/POST/api/vector/new)) |
| `POST` | `/api/vector/delete` | Delete one of your vectors. ([doc](https://www.neuronpedia.org/api-doc#tag/vectors/POST/api/vector/delete)) |
| `POST` | `/api/vector/get` | Fetch a stored vector by `(modelId, source, index)`. ([doc](https://www.neuronpedia.org/api-doc#tag/vectors/POST/api/vector/get)) |

## Auth

- Read-only endpoints (feature lookup, listing) work without auth.
- Steering endpoints and any vector write/delete require an API key,
  passed in the `x-api-key` header. Get one from your Neuronpedia account
  settings.
- Rate limits exist (not formally published as of writing). Keep batch
  sizes small until we know the ceiling.

## Minimal `curl` examples

Replace the placeholders with real values once we confirm Qwen coverage
(see "Open questions" below).

### 1. Fetch a feature

```bash
curl -s "https://www.neuronpedia.org/api/feature/<modelId>/<layer>/<index>" \
  | jq '{model: .modelId, layer: .layer, index: .index,
         top_tokens: (.activations.tokens // .topTokens // []) | .[0:5],
         explanation: .explanation, has_decoder: (.vector != null)}'
```

### 2. Steer a completion with a feature set

```bash
curl -s -X POST "https://www.neuronpedia.org/api/steer" \
  -H 'content-type: application/json' \
  -H "x-api-key: $NEURONPEDIA_API_KEY" \
  -d '{
    "modelId": "<modelId>",
    "prompt": "Question: There are 3 apples. I eat 2. How many remain?\n\nLet'\''s think step by step.\n\n",
    "features": [
      {"modelId": "<modelId>", "layer": "<layer>", "index": <feature_idx>, "strength": 4.0}
    ],
    "temperature": 0.6,
    "n_tokens": 256
  }' | jq '{default: .default.text, steered: .steered.text,
            steered_logprob_first: .steered.logprobs[0]}'
```

`strength` is multiplicative on the feature's decoder column inside their
hook. Values in the 1-10 range are typical; large values (>20) often
saturate or clip.

## Fit to our project

The recipe we previously sketched, `v_steer = W_dec @ w_probe`, has two
expensive parts:

1. **Train an SAE on Qwen3-0.6B's residual stream** at the target layer.
2. **Project a probe trained on the SAE's latents** back to residual space.

Neuronpedia removes the first if Qwen3-0.6B (or a near-equivalent base
model) is hosted. Workflow becomes:

1. For each hidden state `a` at our chosen layer, query
   `POST /api/activation/new` with the surrounding text to get a sparse
   feature vector `z` from Neuronpedia's SAE.
2. Locally fit a sparse logistic-regression probe on `z` separating
   pivotal vs non-pivotal labels (we already have those labels).
3. The probe weights `w_probe` are sparse over feature indices. For each
   non-zero `i`, fetch the decoder column via
   `GET /api/feature/{modelId}/{layer}/{i}` and accumulate
   `v_steer = sum_i w_probe[i] * decoder[i]`.
4. Either:
   - Use the resulting `v_steer` locally with our existing forward hook
     (skip Neuronpedia at inference), or
   - Upload it as a custom vector via `POST /api/vector/new` and let
     `POST /api/steer` do the steering server-side. The latter only works
     if we want hosted inference rather than local Qwen.

Two cheaper, near-term uses that don't require the full pipeline:

- **Feature interpretation overlay**: for our top-impact pivot tokens
  (e.g. the ones in `artifacts/notebook_runs/codelion/promoted_words_by_task.csv`),
  fetch the matching SAE features at `layer=14` and 19 to read the
  auto-explanations. This gives us a story for *why* the steering vector
  helps, beyond raw accuracy.
- **Steering API as a baseline**: send our GSM8K subset through
  `POST /api/steer` with a hand-picked feature set and compare its accuracy
  to our probe-additive / projection-scaling baseline.

## Open questions / TODO before relying on this

- [ ] **Confirm Qwen3-0.6B is on Neuronpedia.** As of writing it is not in
      the small set of headline models on the home page (Llama, Gemma,
      Pythia variants). If it's not hosted, the SAE-substitute path is
      blocked unless we accept using a different base model for the SAE
      and *transferring* features — almost certainly worse than just
      training a small Qwen SAE locally.
- [ ] **Check whether decoder vectors are returned by
      `GET /api/feature/...`** or whether we need a separate endpoint /
      bulk download. The doc page is ambiguous; some hosted SAEs expose
      decoder columns inline, others gate them behind `vector/get`.
- [ ] Confirm the steering API uses the standard "clamp feature value
      then SAE-decode" recipe rather than always-on additive. If it's the
      latter we lose the main reason to go hosted.

## Limitations

- **Model coverage**: hosted SAEs are limited to a curated list. If Qwen3
  isn't there, we can't substitute Neuronpedia for a locally-trained SAE.
- **Hosted inference**: `/api/steer` runs the model server-side. We give
  up control over decoding settings, the exact tokenizer template, and
  any local hooks. Best used as a baseline, not a replacement for our
  notebook runs.
- **Network in the loop**: per-token logprobs round-trip through HTTP.
  For our n=100 GSM8K subset this is fine; for larger sweeps a local SAE
  is probably faster.
- **Auth + rate limits**: write/steer endpoints require an API key.
  Public throttling thresholds are not documented; budget conservatively.
- **Reproducibility**: hosted models can change without us noticing. For
  any results we publish, prefer locally-pinned model + SAE checkpoints.

## References

- Neuronpedia API doc index: <https://www.neuronpedia.org/api-doc>
- Anchors used above:
  - Activations: [`/api/activation/new`](https://www.neuronpedia.org/api-doc#tag/activations/POST/api/activation/new),
    [`/api/activation/get`](https://www.neuronpedia.org/api-doc#tag/activations/POST/api/activation/get)
  - Features: [`/api/feature/{modelId}/{layer}/{index}`](https://www.neuronpedia.org/api-doc#tag/features/GET/api/feature/{modelId}/{layer}/{index})
  - Sparsity: [`/api/sparsity/connected-neurons`](https://www.neuronpedia.org/api-doc#tag/sparsity/GET/api/sparsity/connected-neurons)
  - Steering: [`/api/steer`](https://www.neuronpedia.org/api-doc#tag/steering/POST/api/steer),
    [`/api/steer-chat`](https://www.neuronpedia.org/api-doc#tag/steering/POST/api/steer-chat)
  - Vectors: [`/api/vector/new`](https://www.neuronpedia.org/api-doc#tag/vectors/POST/api/vector/new),
    [`/api/vector/delete`](https://www.neuronpedia.org/api-doc#tag/vectors/POST/api/vector/delete),
    [`/api/vector/get`](https://www.neuronpedia.org/api-doc#tag/vectors/POST/api/vector/get)
