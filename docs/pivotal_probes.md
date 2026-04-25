# Pivotal probes: design notes and application surface

Working notes on the probe architecture, activation-extraction conventions,
and the steering / mech-interp applications we plan to build on top. This
is a design doc, not a tutorial — it assumes familiarity with the existing
notebooks (`notebooks/steering_*.ipynb`) and the PTS dataset
(`codelion/Qwen3-0.6B-pts-steering-vectors`).

Out of scope for this doc:

- Refusal-robustness-against-jailbreaks (covered separately).
- Neuronpedia API integration details (see [`neuronpedia.md`](./neuronpedia.md)).

---

## 1. Recap: what a "pivotal token" is, and what our current probe captures

PTS labels a generated token *t* as **pivotal** if the model's probability
of arriving at the correct final answer changes by more than a threshold
between just before and just after generating *t*. Concretely each row in
the PTS dataset has `prob_before`, `prob_after`, `prob_delta`. The label
is symmetric — `|prob_delta| > τ` — so it bundles together two distinct
populations:

- **Positive pivotal tokens**: `prob_delta > +τ`. Generating this token
  pushed the model *toward* getting the answer right.
- **Negative pivotal tokens**: `prob_delta < −τ`. Generating this token
  pushed the model *away* from the right answer.

Our existing probe at layer 14 (`artifacts/cached3/sklearn/analysis_data/
layer_14/probe_weights.npy`, val acc ≈ 74.6%) is a **binary** logistic
regression trained on `is_pivotal` vs `is_not_pivotal`. It cannot
distinguish positive from negative pivotal tokens — both are folded into
the positive class. So:

- We can detect "something is happening" at the next token.
- We cannot detect "something *bad* is about to happen".

This single fact drives most of the design below.

## 2. Probe architecture: cascade of two binary probes vs one 3-class probe

We need a way to recover the sign distinction. Two implementations are
possible.

### Option A (chosen): a cascade of two binary logistic-regression probes

- **Probe 1 — `is_pivotal`** (what we already have). Trained on
  `pivotal vs non_pivotal`, using activations at position *t−1* to
  predict pivotality of *t*.
- **Probe 2 — `signed`** (to be built). Trained on
  `positive_pivotal vs negative_pivotal`, on the **pivotal subset only**,
  same extraction position (*t−1*). Output: `P(positive | pivotal, h)`.

Inference-time gate:

```
p_steer = P_1(h)  ·  (1 − P_2_pos(h))     # asymmetric multiplicative
```

Steering fires only when *something* pivotal is predicted **and** it
looks like the negative kind. When `P_2 ≈ 0.5`, `p_steer` shrinks back
toward zero — we don't act on activations the signed probe is unsure
about. When `P_1 ≈ 0`, `p_steer` is also zero — we don't act on
activations that aren't pivotal at all.

### Option B (rejected): one multinomial 3-class probe over `{none, pos, neg}`

Mechanically possible (sklearn `LogisticRegression(multi_class=
"multinomial")`), and after fitting it produces 2 effective directions in
activation space (3 rows, minus 1 for softmax shift-invariance). Rejected
for four reasons:

1. **Independent objectives → cleaner stories per direction.** In the
   cascade each direction is the answer to one yes/no question. In the
   3-class softmax each row is "evidence for class k vs. a chosen
   reference", entangled with the other classes' boundaries.
2. **Independent training distributions.** Probe 2 sees only pivotal
   activations, so the pos/neg axis is not drowned by the ~95%
   non-pivotal majority. A 3-class softmax on the natural distribution
   spends most of its capacity on the easy boundary.
3. **Independent steering vectors.** `w_1` (or `μ_pivotal − μ_non`) is
   the "amplify pivotal-ness" direction. `w_2` (or `μ_pos − μ_neg`) is
   the "flip negative pivotal toward positive" direction. They're
   different vectors and we want to compose them independently. A
   3-class softmax constrains both to a 2D subspace optimized jointly.
4. **Independent calibration.** Probe 1 wants high recall; probe 2 wants
   high precision; cascade lets us tune two thresholds independently. A
   3-class argmax has to be defeated by post-hoc ratios like
   `P(neg)/(P(pos)+P(neg))` — i.e. probe 2 reinvented, fit worse.

### Implementation notes for probe 2

- Training data: PTS rows where the token was labelled pivotal. Define
  positive as `prob_delta > +τ_pos` and negative as `prob_delta < −τ_neg`,
  with a dead zone in between to avoid borderline noise.
- Extract at the **same position** probe 1 uses (the token *preceding*
  the pivot — see §3). Otherwise probe 2's distribution at inference
  time differs from training.
- Layer choice: start with the same layer as probe 1 (14 for our setup).
  Run a layer sweep separately — there is no a-priori reason the pos/neg
  signal peaks at the same layer as the pivotal/non-pivotal signal.
- Save artifacts symmetrically with probe 1:
  - `artifacts/cached3/sklearn/analysis_data/layer_X/probe_signed_weights.npy`
  - `artifacts/cached3/sklearn/analysis_data/layer_X/probe_signed_biases.npy`
  - `artifacts/cached3/sklearn/steering_configs/steering_layer14_signed_*.npy`
    (force-added so Colab can fetch them).

### Choice of steering vector to pair with probe 2

When probe 2 fires "negative", which direction do we steer along? Three
candidates worth comparing head-to-head, in order of expected strength:

1. **`μ_pos − μ_neg` (signed CAA).** The contrastive mean over the two
   manifolds the gate is distinguishing. Most principled choice — its
   purpose is exactly to push the residual stream from one cluster
   toward the other.
2. **`w_2` (probe-2 weight vector).** Same direction in spirit but
   shaped by L2 regularization and the LR loss; should be similar to
   (1) up to scaling and noise.
3. **Existing probe-1 / centroid-diff direction with sign flip.** Cheap
   reuse but worse: it encodes pivotal-ness, not pos-vs-neg, so flipping
   its sign is a coarser approximation.

(1) is the first one to train and ship.

## 3. Activation-extraction position: *t−1* predicts *t*

Critical design decision in the existing pipeline: probes are trained on
the activation at the token position **immediately preceding** the pivot
token, with the goal of predicting whether the *next* token is pivotal.
Two practical consequences follow.

### 3a. The probe is a causal detector, not a post-hoc classifier

Most published CAA / linear-probe work extracts at the target token
itself. That makes the probe useful for labelling training data, but
useless at inference time — by the time you have the activation at
*t* you have already produced *t*. Extracting at *t−1* means the
probe's prediction is available **before** the next forward pass
commits to a token, which is what makes selective intervention viable
at all.

### 3b. The probe is a lightweight PTS substitute

The original PTS algorithm requires *N* counterfactual rollouts per
candidate token to estimate `prob_after` — orders of magnitude more
expensive than a single forward pass. A trained 1024-dim linear probe
at *t−1* approximates the same decision in a few microseconds per
token. This is the reason "probe-only" is a real claim to bring back
to the PTS author, not just an engineering shortcut: it converts an
offline analysis tool into an online detector.

### 3c. Implication for hooks

The forward hook at layer L fires on *every* layer-L forward pass.
At step *t* the hook reads the residual stream that produced *t*'s
input — which is the activation at *t−1*. So the natural place to
*read* probe 1 is inside the hook before the layer's own computation
runs. To *act* on the prediction, the hook then modifies the residual
stream that feeds the next layer's attention/MLP at the same step,
which is what produces the logits for *t*. Read and act in the same
hook call.

## 4. Priority application: reactive (token-conditional) steering

This is the single highest-priority follow-up. Everything else in
this doc is either a special case of it (tool-use gating, confidence-
aware decoding) or a means to it (probe 2, SAE projection).

### Motivation

Always-on hooks (every token, every step) work but degrade fluency:
the steering vector adds energy at every position regardless of whether
the model is actually about to make a pivotal decision. Empirically
we see this in the existing notebooks — accuracy gains plateau or
reverse at high `α` because non-pivotal tokens get nudged off-manifold
for no reason.

A reactive hook fires **only at predicted pivot tokens**, and only in
the harmful direction. Same accuracy lift, far less collateral.

### Algorithm

```
for each forward pass at layer L, at step t:
    h = residual_stream                     # shape (B, T, d), use h[:, -1]

    p1 = sigmoid(w_1 . h_last + b_1)        # is next token pivotal?
    if p1 < tau_1: continue                 # do nothing

    p2_pos = sigmoid(w_2 . h_last + b_2)    # if so, will it be helpful?
    p_steer = p1 * (1.0 - p2_pos)           # asymmetric multiplicative gate

    if p_steer < tau_steer: continue
    if t - last_fired < hysteresis_k: continue   # avoid oscillation

    h_last_new = h_last + alpha(p_steer) * v_signed_caa
    write h_last_new back into residual_stream
    last_fired = t
```

### Knobs and how to tune them

- **`τ_1`**: set against probe 1's val PR curve at a fixed false-positive
  budget (e.g. ≤ 5% of non-pivotal tokens). High recall preferred at
  this stage.
- **`τ_steer`** (or equivalently a threshold on `p_steer`): tunes
  precision of the *combined* gate. Sweep on a held-out set of GSM8K
  prompts and pick the value that maximizes accuracy minus a fluency
  penalty (e.g. perplexity on a benign held-out corpus).
- **`α(p_steer)`**: linear in `p_steer` is a reasonable starting choice
  (`α = α_max · p_steer`). Constant `α` works too but wastes budget on
  borderline cases.
- **`hysteresis_k`**: 1–3. Without it the steered residual stream feeds
  back into probe 1 at the next step and can re-trigger immediately,
  causing oscillation. Easiest fix: lock the hook for `k` tokens after
  a fire.
- **Calibration of probe 2 conditional on probe 1**: temperature-scale
  probe 2 on a held-out set where probe 1 fires, *not* on the full
  signed-pivot test set. Otherwise the gate is miscalibrated in the
  exact regime where it matters.

### Notebook to build

`notebooks/steering_reactive.ipynb`, scaffolded after
`steering_probe_weights.ipynb`:

1. Load probe 1 + probe 2 + `v_signed_caa`.
2. Hook factory takes `(probe_1, probe_2, v_signed, α_max, τ_1, τ_steer,
   hysteresis_k)`.
3. Eval matrix: always-on baseline (current notebook) vs reactive at
   matched `α_max`, sweep `τ_steer` and `α_max`.
4. Headline metrics: GSM8K accuracy, **fire rate** (% of tokens where
   the hook fired), accuracy lift per fire, parse-rate delta. Plot
   accuracy-vs-fire-rate curves — this is the single graph that sells
   the approach.

## 5. Other applications of the (probe-1, probe-2) pair

These are smaller follow-ups; some are pure-monitor (probe 1 only),
some are read-and-act (cascade).

### 5a. Confidence-aware decoding (probe 1, monitor only)

When probe 1 fires above some threshold the model is at a
"high-leverage" token. Three deployment modes:

- **Verifier hand-off.** Route the prefix to a stronger / more
  expensive model only at pivot tokens.
- **Targeted retrieval.** Trigger a tool call (search, calculator,
  KB lookup) at pivots and continue with the result inserted.
- **Best-of-K decoding.** Sample N continuations only at pivot
  tokens, score them with a separate verifier, keep the best.
  Compute spent only where it matters.

This is a probe-only feature — no steering, no probe 2. Useful as a
fallback if probe 2 is too noisy.

### 5b. Tool-use gating (probe 1 + a multi-class action probe)

Treat `{plain_token, search_call, calculator_call, code_call, ...}` as
labels and train a small multinomial probe at *t−1*. Emit structured
"about to call tool X" events to an orchestrator. Same architecture
as our cascade, just with the second probe being multi-class over
action types instead of binary over signs. Closest commercial analog
to what Goodfire / latent_node demos look like.

### 5c. Causal attribution at token level

Pivot tokens identified by probe 1 are localized "decision moments".
Path-patching / attribution-graph analysis at *those* token positions
gives a circuit-level story for *why* the steering vector helps:

- Which attention heads write to `w_1` at *t−1*?
- Which earlier tokens (via attention) influence the residual stream
  at *t−1* most?
- Do MLPs at later layers undo / amplify that signal?

This is the bridge from PTS → mech-interp circuit work. The token-level
granularity is what makes it tractable; doing it on whole-prompt CAA
data smears the signal across all positions.

### 5d. Token-level localization for SAE feature labeling

SAE features have a labelling problem — feature `i` fires, but on
what concept? Conditional on PTS pivot tokens, those activations
are *labelled*. Three uses:

- For each SAE feature `i`, compare its activation distribution at
  pivot vs non-pivot positions. Features with high lift are
  candidates for "this feature represents *the pivotal concept*".
- Decompose `w_1` (or `w_2`) into its sparse SAE-feature
  representation: `w_probe ≈ Σ c_i · D[:, i]` where `D` is the SAE
  decoder. The non-zero `c_i` are the few features the probe is
  actually using, and their hosted explanations (Neuronpedia or
  local) become the probe's interpretation.
- Run cluster analysis on pivot-token SAE activations to look for
  multiple subtypes of pivotal moment (the codelion-dataset
  reasoning_pattern groups already hint at this).

## 6. SAE integration: feature amplification, clamping, patching, probing

A separate axis of work that composes naturally with everything above.
Two layers (no pun intended) of integration.

### 6a. SAE-projected steering: `v_steer = D · w_probe`

Recipe: train (or load) an SAE at layer L; train a **probe on the SAE
latents** rather than the residual stream; project the probe weights
back through the decoder to get a steering vector that lives in residual
space but only excites a sparse set of interpretable features.

Why bother:
- The standard probe direction is dense (1024 non-zero entries) and
  basically uninterpretable as a list of numbers. The SAE-projected
  version is sparse — typically 5–50 non-zero features — and each
  feature has a name.
- Steering with `D · w_probe` only excites those features, not arbitrary
  combinations. Fluency cost should drop because we're staying on the
  SAE-spanned manifold.
- It is the same operation Neuronpedia's `/api/steer` performs server-
  side, but with a probe-derived feature set instead of a hand-picked
  one.

Status: pending. Requires either training an SAE on Qwen3-0.6B or
verifying Qwen coverage on Neuronpedia (open question, see
[`neuronpedia.md`](./neuronpedia.md)).

### 6b. Feature amplification / clamping / patching at pivot positions

Once we have an SAE at layer L, three additional interventions become
available, all firing **only at probe-1-predicted pivot positions**
(reactive style):

- **Amplification.** `z_i ← α · z_i` for selected feature indices `i`.
  Picks out a single named feature to push harder.
- **Clamping.** `z_i ← c` (constant). Forces a feature to a fixed
  activation regardless of input. Useful for "always-on this concept"
  experiments.
- **Patching.** Replace `z_i` with the value it had on a different
  prompt. Standard activation-patching idiom; pivot-position-only
  patching tells you whether the *moment* of the decision is what
  matters or the whole context.
- **SAE-probing.** Train a probe directly on the SAE latents, predict
  pivotal vs non-pivotal — see how sparse the discriminator is. If
  it's a handful of features, you have a fully interpretable readout.

All four interventions reuse probe 1 as the gate; the "what to do" is
the SAE-side knob.

## 7. Open questions and experimental knobs

Tracked here so they don't get lost between notebook runs.

- **Layer choice for probe 2.** Probe 1 peaks at L14 for our setup.
  Pos vs neg pivotal might peak elsewhere. Run the sweep.
- **Threshold for "pivotal".** PTS uses `|prob_delta| > τ`. Currently
  we inherit τ from the dataset; revisit whether tightening τ changes
  probe accuracy in interesting ways.
- **Layer of intervention vs layer of detection.** The current pipeline
  reads probe 1 at L14 and steers at L14. Decoupling those (read at L_r,
  steer at L_s) is one experiment we have not run. Plausible that the
  cleanest read layer ≠ the best inject layer.
- **Sequence-level positivity.** A token can be locally negative-pivotal
  but globally helpful (e.g. an exploratory step that gets corrected
  later). Probe 2 trained on `prob_delta` of the immediate next-token
  rollout might be too myopic. Consider a longer-horizon signed label.
- **Hysteresis interaction with sampling temperature.** `T = 0.6` with
  `top_p = 0.9` is what our existing runs use. Hysteresis logic should
  hold across sampling decisions — add a sanity test that fire rates
  are not wildly different across seeds.
- **Position drift under steering.** When the hook fires, the steered
  residual stream changes the token actually generated, which may shift
  which positions become pivotal in the rest of the rollout. This is
  fine in expectation but means the offline "pivotal token list" we
  trained against is not the same as the online one. Worth quantifying.

## 8. Suggested project order

By falling-cost / rising-value:

1. **Build probe 2** (`signed` binary probe at L14). One short notebook,
   tiny artifact, no inference-time changes yet. Ship as
   `notebooks/probe_signed_train.ipynb`.
2. **Train signed CAA vector** `μ_pos − μ_neg` from the same data.
   Save under `steering_configs/steering_layer14_signed_vector.npy`.
3. **Reactive-steering notebook** (§4). Replaces the always-on hook
   with the cascade gate, evaluates accuracy-vs-fire-rate curve on
   GSM8K. This is the headline.
4. **Layer sweep for probe 2.** Cheap once (1) is done.
5. **SAE-projected steering.** Either train a small SAE on Qwen3-0.6B
   at L14, or substitute Neuronpedia (after confirming Qwen coverage).
6. **Causal attribution notebook** at probe-1-fired positions.
   Path-patch / attribution-graph at *t−1* across the candidate
   layers; identify the heads writing to `w_1`.
7. **Tool-use gating prototype** (5b). Outside of GSM8K — needs a
   tool-using benchmark (τ-bench or similar).

## 9. Files touched / planned

Existing:
- `artifacts/cached3/sklearn/analysis_data/layer_14/probe_weights.npy`
  (probe 1 weights).
- `artifacts/cached3/sklearn/steering_configs/steering_layer14_*.npy`
  (probe 1 weights, biases, centroid-diff vector — Colab-fetchable).
- `notebooks/steering_probe_weights.ipynb` (additive + projection-scaling
  arms with probe 1).
- `notebooks/steering_random_control.ipynb` (null control).

Planned (in order of build):
- `notebooks/probe_signed_train.ipynb` — train probe 2.
- `artifacts/cached3/sklearn/steering_configs/steering_layer14_signed_*.npy`
  — probe 2 weights + signed CAA vector.
- `notebooks/steering_reactive.ipynb` — the cascade hook + GSM8K eval.
- `notebooks/sae_projected_steering.ipynb` — SAE recipe (depends on
  SAE availability).
- `notebooks/attribution_at_pivots.ipynb` — circuit work at pivot tokens.

---

*Last revised: this doc tracks the design conversation through the
probe-weights notebook (commit `40a1d37`) and predates the reactive
notebook implementation. Update as the items in §8 land.*
