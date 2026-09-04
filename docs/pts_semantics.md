# Upstream PTS: exact token-search semantics

Read from `codelion/pts` at commit **`8334808`**. This is the reference
for our harness (`pts_harness/`) and for how we consume PTS events in
`probe_pipeline/`. Anything here that changes upstream invalidates the
cost model in the project plan and the label logic in `preprocess.py`.

Files: `pts/searchers/token.py`, `pts/searchers/base.py`, `pts/cli.py`,
`pts/dataset.py`, `pts/oracle.py`.

---

## 1. The algorithm

### 1.1 Screening (`TokenPTSSearcher.search`, token.py:98-107)

```python
init_prob = self.estimate_success_probability(query, ...)
if not (min_prob <= init_prob <= max_prob):
    return          # query is skipped entirely
```

Defaults `[0.2, 0.8]`. **This is a load-bearing filter, not a tuning
knob**: a query the model always (or never) solves has no pivotal tokens,
because no single token can move a saturated probability. It also means

- pivot *density* is confounded with task difficulty relative to model
  capability, so the honest cross-model comparison is at matched
  difficulty band, not matched dataset; and
- larger models need proportionally more candidate queries on an easy
  task, because more of them screen out.

### 1.2 Generation (token.py:117-137)

`max_generations` sampled continuations per query, one at a time,
`do_sample=True` at `temperature/top_p/top_k/min_p`. Generations shorter
than **5 tokens are skipped** (`token.py:135`).

### 1.3 Bisection (`subdivide_sequence`, token.py:41-77)

```python
if len(sequence) <= 1:
    return [sequence]                      # returns WITHOUT scoring

prob_before = P(prefix)
prob_after  = P(prefix + sequence)

if abs(prob_after - prob_before) < prob_threshold:
    return [sequence]                      # prune: segment kept whole

mid = len(sequence) // 2
left, right = sequence[:mid], sequence[mid:]
return (subdivide(left,  prefix)
      + subdivide(right, prefix + left))
```

Note the recursion **only descends into segments that move the
probability**. Segments below threshold are returned whole and are never
split further. Singletons return before scoring.

### 1.4 Emission (token.py:148-204)

Walks the returned segments accumulating `current_prefix`. For each
**length-1** segment it re-scores `P(current_prefix)` and
`P(current_prefix + token)` and emits an event when
`abs(prob_delta) >= prob_threshold`.

---

## 2. The probability cache changes the cost model

`estimate_success_probability` (base.py:235-285) is memoized by
`(query, prefix, num_samples, system_prompt, category)` in a bounded LRU
(`max_cache_size=4096`, base.py:82).

Tracing the recursion against that cache:

| call | key | status |
| --- | --- | --- |
| node: `P(prefix)` | prefix | hit (parent computed it) |
| node: `P(prefix+seq)` | prefix+seq | hit (parent computed it) |
| left child: `P(prefix)` | prefix | hit |
| left child: `P(prefix+left)` | prefix+left | **new — the midpoint** |
| right child: `P(prefix+left)` | prefix+left | hit (left child) |
| right child: `P(prefix+left+right)` | = prefix+seq | hit (parent) |

**Each bisection node costs exactly one new probability estimate — its
midpoint.** The root costs two. The emit loop is almost entirely cache
hits. So the cost driver is the count of *unique midpoints*, i.e.

```
B  ~=  k * log2(L)          k = pivots found, L = generation length
```

With `k≈3`, `L≈320`: `B ≈ 25`. This matches the plan's assumption; the
plan's estimate stands.

**Total generated tokens:**

```
screening = Q         * S * L
search    = Q * f * G * B * S * L
```

At `Q=1500, f=0.4, G=1, B=25, S=40, L=240`: 14.4M + 144M ≈ **158M tokens
per model**, in the same range as the plan's 130M.

---

## 3. Where the vLLM backend goes

`generate_completions` (base.py:178-219) is the single hot spot:

```python
while remaining > 0:
    current = min(self.batch_size, remaining)      # batch_size default 5
    outputs = self.model.generate(..., num_return_sequences=current, ...)
```

With CLI defaults (`--num-samples 50 --batch-size 5`) that is **ten
sequential `generate` calls per probability estimate**, each with a batch
of five. On an A100 running a 0.6B model this leaves the GPU almost idle.

Replacing this one function with a vLLM backend is the whole speedup, but
note it only batches *within* one estimate (~50 sequences). The larger
win requires driving many queries' bisection frontiers concurrently — see
the plan, Phase 3.

---

## 4. Facts our pipeline depends on

### 4.1 `position` is the pivot's absolute index — so *t−1* is exact

```python
position=len(current_prefix),        # token.py:189
```

`current_prefix` is prompt + everything emitted before this token, so
`position` is the **absolute, prompt-inclusive token index of the pivotal
token**. Therefore:

> **t−1 = `position` − 1.**

No string matching, no `startswith` search, no re-tokenization to recover
indices. This removes the entire class of BPE-merge and round-trip
hazards documented in `docs/issues.md` §9 and in the project plan §3.3-3.4.

**Caveat: the released `codelion/*-pts` datasets predate this field.**
They carry only `pivot_context` / `pivot_token`, which is why the current
`preprocess.py` reverse-engineers positions from strings. Data we generate
ourselves does not need that path.

### 4.2 `context` already contains special tokens

`context=prefix_str` where `prefix_str = tokenizer.decode(current_prefix,
skip_special_tokens=False)`. So re-encode it with
**`add_special_tokens=False`** or you get a doubled BOS and score on a
prefix the model never saw. Our `preprocess.py` already does this
correctly (`add_special_tokens=False` everywhere).

### 4.3 Token PTS's prefix *replaces* the prompt

`build_conditioning_text` (base.py:159-174): a non-empty prefix is
returned as-is, because for token PTS it is a fully-decoded string that
already contains the formatted prompt. Sentence PTS overrides this to
append. Do not mix the two conventions.

### 4.4 No chat template unless a system prompt is passed

```python
def format_prompt(self, query, system_prompt=None, category=None):
    if system_prompt:
        return self.tokenizer.apply_chat_template(...)      # base.py:150
    if category and hasattr(self.oracle, "get_prompt_for_category"):
        return self.oracle.get_prompt_for_category(query, category)
    return query                                            # <-- default
```

**With no `--system-prompt`, the model is conditioned on the bare question
text.** This resolves an open question from the plan: `codelion/Qwen3-0.6B-pts`
shows raw GSM8K text in `pivot_context` with no `<|im_start|>`, so it was
generated on this default path — **no chat template, and therefore no
Qwen3 thinking mode**, since thinking is triggered through the template.

Consequence for us, and it is a real experimental decision:

- **Raw conditioning** reproduces codelion exactly, so it is what the
  validation run must use. But it is off-distribution for an instruct
  model and depresses success rates, which is part of why baselines here
  look low.
- **Chat-template conditioning** is more realistic and what any deployment
  would do, but breaks comparability with the released data.

Plan: validate at 0.6B with raw conditioning, then run the main ladder
with the chat template applied (still `enable_thinking=False`), and report
the conditioning mode explicitly. Do not silently mix them.

---

## 5. Defaults, and where they come from

`pts/cli.py` overrides several `BasePTSSearcher` defaults. The CLI values
are what produced the published datasets.

| Parameter | base.py | cli.py | ours |
| --- | --- | --- | --- |
| `num_samples` | 20 | **50** | 40 |
| `batch_size` | 5 | 5 | n/a (vLLM) |
| `prob_threshold` | 0.2 | 0.2 | 0.2 (post-hoc, see below) |
| `temperature` | 0.6 | 0.6 | 0.6 |
| `top_p` / `top_k` / `min_p` | 0.95 / 20 / 0.0 | same | same |
| `max_new_tokens` | 512 | 512 | 320 |
| `max_generations` | — | 10 | **1** |
| `min_prob` / `max_prob` | — | 0.2 / 0.8 | same |
| `max_examples` | — | 100 | 1500 |

`--max-examples 100` is why the published datasets have ~104 items.

**We store sufficient statistics (`n`, `n_success`) rather than `p̂` or the
accept/reject verdict**, which makes `prob_threshold` a post-hoc knob and
lets every event carry a Wilson CI. See the plan §1.3b: at `S=50` and
`τ=0.2` the acceptance threshold sits at only 2σ, so an estimated
~25–30% of the "pivotal" class is sampling noise.

---

## 6. Datasets and oracles

`create_oracle_from_dataset` (dataset.py:125+) special-cases
`codelion/optillmbench` and matches `"gsm8k" in dataset_id.lower()`,
extracting ground truth with `####\s*(-?[\d,]+(?:\.\d+)?)` and returning
`MathOracle(dataset_format="gsm8k")`. Everything else falls back to field
auto-detection. Adding MATH or MBPP means editing this function — a small
patch, not a redesign. `CodeOracle` (executes extracted Python in a
subprocess) already exists but is unused on our path.

---

## 7. Still to verify empirically

- `f`, the fraction of queries inside `[min_prob, max_prob]` — assumed
  0.4, must be measured per model in the pilot.
- `B`, unique midpoints per generation — derived above as `k·log2(L)`,
  but `k` is data-dependent. Measure it.
- Why upstream defaults `max_generations` to 10. If many generations fail
  the oracle outright, `G=1` wastes screened queries.
- Throughput (tok/s) per model under vLLM on the actual GPU.
