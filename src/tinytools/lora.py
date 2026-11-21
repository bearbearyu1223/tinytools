"""LoRA from the two matrices in the paper, ported from the series' post 6 code
(llm-architectures-refresher ``lora.py``; the parts training needs, unchanged in behaviour).

A frozen ``nn.Linear`` ``W`` gets a trainable low-rank pair beside it::

    y = W x  +  (alpha / r) * B A x

``A`` is ``(r, in)``, ``B`` is ``(out, r)`` and starts at zero, so the adapted
model is the base model until the first step. :func:`inject_lora` freezes the
whole model first (embeddings, norms and LM head included), so the only
trainable tensors are the adapters it adds.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRALinear", "inject_lora", "trainable_parameters", "ATTENTION_MLP"]

# Qwen2's attention and MLP projections: "attention + MLP" in the plan.
ATTENTION_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable rank-``r`` update beside it."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.A = nn.Parameter(torch.empty(rank, base.in_features, **kw))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, **kw))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


def inject_lora(model: nn.Module, targets: Iterable[str], *, rank: int,
                alpha: float) -> dict[str, LoRALinear]:
    """Freeze ``model`` and wrap every child ``nn.Linear`` named in ``targets``."""
    wanted = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped: dict[str, LoRALinear] = {}
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in wanted and isinstance(child, nn.Linear):
                adapter = LoRALinear(child, rank, alpha)
                setattr(parent, child_name, adapter)
                wrapped[f"{parent_name}.{child_name}" if parent_name else child_name] = adapter
    missing = wanted - {name.rsplit(".", 1)[-1] for name in wrapped}
    if missing:
        raise ValueError(f"no nn.Linear named {sorted(missing)} found in the model")
    return wrapped


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
