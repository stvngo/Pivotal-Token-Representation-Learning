"""GSM8K evaluation for causal validation. Batched, hookable, honestly paired.

The v1 evaluator was confounded on four independent axes, each of which moved
accuracy without touching reasoning quality (see the audit in the project
plan, Part 4):

* no chat template, so an instruct model was scored off-distribution;
* 256 new tokens, which truncated correct answers mid-sentence;
* the answer parsed from the *full decoded sequence*, prompt included, so
  14/100 predictions were numbers lifted out of the question;
* one RNG seed per arm consumed across variable-length generations, so base
  and steered arms were genuinely paired only for the first example.

All four are fixed here. Two design choices are worth stating because they
are not the obvious ones.

**HuggingFace, not vLLM, for everything.** Steering needs residual-stream
hooks, which vLLM does not expose. Running the baseline under vLLM and the
steered arms under HF would make the comparison a comparison of inference
stacks. Batching (left-padded, attention-masked) recovers most of the speed;
the v1 code ran batch size 1.

**Greedy decoding is the primary metric.** It is fully deterministic, so
base and steered arms are exactly paired and every flipped answer is
attributable to the intervention rather than to resampling. Sampled arms are
run too, over several seeds, to show the effect survives realistic decoding
and to establish the noise band -- but a causal claim should not rest on a
comparison whose noise floor is +-4 points.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

# Ask for the format we parse. The v1 prompt never mentioned ####, so its
# primary branch fired 0/100 times and everything fell through to "last
# number anywhere", including numbers in the question.
PROMPT_SUFFIX = (
    "\n\nSolve this step by step. End your response with the final numeric "
    "answer on its own line in the form: #### <answer>"
)

_GSM8K = re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)")
_LAST_NUMBER = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


def extract_answer(response: str) -> str | None:
    """Parse the model's answer from the RESPONSE ONLY.

    Never pass the prompt in. The fallback exists because models sometimes
    omit the marker, but it is scoped to the generated text so it cannot
    read a number out of the question.
    """
    m = _GSM8K.search(response)
    if m:
        return m.group(1).replace(",", "")
    nums = _LAST_NUMBER.findall(response)
    return nums[-1].replace(",", "") if nums else None


def extract_gold(answer_field: str) -> str | None:
    m = _GSM8K.search(answer_field)
    return m.group(1).replace(",", "") if m else None


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


@dataclass
class ArmResult:
    name: str
    accuracy: float
    n: int
    n_correct: int
    marker_rate: float          # fraction using the #### branch, not the fallback
    truncation_rate: float      # fraction that hit the token cap
    mean_new_tokens: float
    correct_mask: list[bool] = field(default_factory=list)
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("correct_mask")
        return d


def build_prompts(questions: Sequence[str], tokenizer: Any) -> list[str]:
    """Chat-templated prompts. Thinking off, to match the PTS generation."""
    out = []
    for q in questions:
        msgs = [{"role": "user", "content": q + PROMPT_SUFFIX}]
        try:
            out.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:      # tokenizers without the thinking switch
            out.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )
    return out


@torch.no_grad()
def generate_batched(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 640,
    batch_size: int = 32,
    greedy: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 0,
    device: Any = None,
) -> tuple[list[str], list[int]]:
    """Batched generation. Returns (responses, new-token counts).

    Left padding is required: with right padding a decoder-only model
    continues from pad tokens rather than from the prompt.
    """
    device = device or next(model.parameters()).device
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses: list[str] = []
    n_new: list[int] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = list(prompts[i : i + batch_size])
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            torch.manual_seed(seed + i)      # deterministic per batch
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            for row in gen:
                ids = row.tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[: ids.index(tokenizer.eos_token_id)]
                n_new.append(len(ids))
                responses.append(tokenizer.decode(ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = saved_side
    return responses, n_new


def score(
    responses: Sequence[str],
    golds: Sequence[str],
    n_new: Sequence[int],
    *,
    name: str,
    max_new_tokens: int,
    seed: int | None = None,
) -> ArmResult:
    mask, markers = [], 0
    for r, g in zip(responses, golds):
        markers += bool(_GSM8K.search(r))
        mask.append(is_correct(extract_answer(r), g))
    n = len(mask)
    return ArmResult(
        name=name,
        accuracy=sum(mask) / n if n else 0.0,
        n=n,
        n_correct=sum(mask),
        marker_rate=markers / n if n else 0.0,
        truncation_rate=sum(1 for k in n_new if k >= max_new_tokens) / n if n else 0.0,
        mean_new_tokens=sum(n_new) / n if n else 0.0,
        correct_mask=mask,
        seed=seed,
    )


def mcnemar(base: Sequence[bool], other: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar on paired correctness."""
    from math import comb

    gained = sum(1 for b, o in zip(base, other) if o and not b)
    lost = sum(1 for b, o in zip(base, other) if b and not o)
    n = gained + lost
    if n == 0:
        p = 1.0
    else:
        k = min(gained, lost)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"gained": gained, "lost": lost, "net": gained - lost, "p": p}
