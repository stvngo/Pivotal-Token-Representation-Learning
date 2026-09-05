"""vLLM rollout backend -- the production path.

Rollout generation is essentially all of PTS's cost, and upstream issues it
as ``model.generate(num_return_sequences=5)`` in a Python loop: ten
sequential calls of batch five per probability estimate, which leaves an
A100 nearly idle. Handing the whole wave to a continuous batcher instead is
what takes a Qwen3-0.6B run from tens of hours to a few.

Two details do real work here:

``n`` per request
    Asking vLLM for ``n=num_samples`` in one request lets it compute the
    node's prefill once and fork the KV cache, instead of re-prefilling a
    long shared prefix dozens of times.

``enable_prefix_caching``
    A bisection node's prefix extends its parent's. With prefix caching the
    shared portion is not recomputed, which is the difference between
    prefill being negligible and being a third of the run.

Determinism: ``SamplingParams(seed=...)`` is per request, unlike HF's global
RNG, so a node's samples do not depend on what else shared its batch. Exact
bitwise reproducibility across different batch shapes is not achievable with
a continuous batcher and should not be chased -- re-runnability comes from
the checkpoint, which replays every node from stored counts.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Rollout, RolloutRequest, RolloutResult


class VLLMRolloutBackend:
    """Continuous-batched sampling through vLLM."""

    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 2048,
        enable_prefix_caching: bool = True,
        max_num_seqs: int = 256,
        seed: int = 42,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        llm: Any = None,
    ) -> None:
        self.model = model
        if llm is not None:
            self.llm = llm
        else:
            from vllm import LLM

            self.llm = LLM(
                model=model,
                revision=revision,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                seed=seed,
                enforce_eager=enforce_eager,
                trust_remote_code=trust_remote_code,
            )
        self._tokenizer = None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = self.llm.get_tokenizer()
        return self._tokenizer

    def generate(self, requests: Sequence[RolloutRequest]) -> list[RolloutResult]:
        if not requests:
            return []
        from vllm import SamplingParams, TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_token_ids)) for r in requests]
        params = [
            SamplingParams(
                n=r.n,
                seed=r.seed,
                max_tokens=r.max_new_tokens,
                temperature=r.temperature,
                top_p=r.top_p,
                top_k=r.top_k,
            )
            for r in requests
        ]

        # One call for the whole wave; vLLM schedules across all of it.
        outputs = self.llm.generate(prompts, params)

        results: list[RolloutResult] = []
        for req, out in zip(requests, outputs):
            results.append(
                RolloutResult(
                    request_id=req.request_id,
                    rollouts=[
                        Rollout(
                            text=c.text,
                            token_ids=tuple(c.token_ids),
                            finish_reason=c.finish_reason or "stop",
                        )
                        for c in out.outputs
                    ],
                )
            )
        return results

    def detokenize(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def close(self) -> None:
        self.llm = None
