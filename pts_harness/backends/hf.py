"""HuggingFace rollout backend.

The portable fallback: works on MPS and CPU, which is what makes the harness
testable end-to-end on a laptop. It is also what upstream PTS uses, so it is
the reference for checking that the vLLM backend agrees.

Not the production path. ``model.generate`` with a small
``num_return_sequences`` leaves a GPU mostly idle, which is exactly the
bottleneck the vLLM backend exists to remove.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Rollout, RolloutRequest, RolloutResult


class HFRolloutBackend:
    """Batched-ish sampling through ``transformers``."""

    name = "hf"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: Any = None,
        micro_batch_size: int = 8,
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.micro_batch_size = micro_batch_size
        self._torch = torch
        model.eval()

    def generate(self, requests: Sequence[RolloutRequest]) -> list[RolloutResult]:
        torch = self._torch
        out: list[RolloutResult] = []

        for req in requests:
            ids = torch.tensor(
                [list(req.prompt_token_ids)], dtype=torch.long, device=self.device
            )
            rollouts: list[Rollout] = []
            remaining = req.n
            # Chunked so that a large num_samples does not blow up memory on
            # a 16 GB laptop.
            while remaining > 0:
                take = min(self.micro_batch_size, remaining)
                torch.manual_seed(req.seed + len(rollouts))
                with torch.no_grad():
                    gen = self.model.generate(
                        ids,
                        do_sample=True,
                        num_return_sequences=take,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        pad_token_id=self.tokenizer.pad_token_id
                        or self.tokenizer.eos_token_id,
                    )
                for seq in gen:
                    new = seq[ids.shape[1]:].tolist()
                    rollouts.append(
                        Rollout(
                            text=self.tokenizer.decode(new, skip_special_tokens=True),
                            token_ids=tuple(new),
                            finish_reason="stop",
                        )
                    )
                remaining -= take
            out.append(RolloutResult(request_id=req.request_id, rollouts=rollouts))
        return out

    def detokenize(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def close(self) -> None:
        del self.model
