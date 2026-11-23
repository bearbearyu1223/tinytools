"""LoRA from the two matrices in the paper, ported from the series' post 6 code
(llm-architectures-refresher ``lora.py``; the parts training needs).

A frozen ``nn.Linear`` ``W`` gets a trainable low-rank pair beside it::

    y = W x  +  (alpha / r) * B A x

``A`` is ``(r, in)``, ``B`` is ``(out, r)`` and starts at zero, so the adapted
model is the base model until the first step. :func:`inject_lora` freezes the
whole model first (embeddings, norms and LM head included), so the only
trainable tensors are the adapters it adds.

**Precision (post 9's T8 spec).** ``A`` and ``B`` are float32 whatever the base
weights are (bf16 on the GPU), so their gradients and the optimizer's state, made in
the parameters' dtype, are float32 too: an Adam step of ~1e-4 against an ``A``
entry of ~0.03 would round away in bf16. The two matmuls of the low-rank path run
in the activations' dtype, with ``A`` and ``B`` cast to it for the product (what
``torch.autocast`` does with float32 parameters): running them in float32 would
keep a float32 copy of every adapted layer's input for the backward pass, about
2 MB per token on 1.5B, which the token budget can't hold. The cast's backward
returns float32 gradients, so accumulation and the update stay float32.

**Seeded init.** With ``generator``, every ``A`` is drawn on the CPU, in
``named_modules`` order, from that generator (then moved to the device), so a
seed gives the same adapters on a Mac and on a GPU.

**Scoring.** :func:`merge_into` folds a saved adapter into a model's weights:
``W + (alpha / r) B A`` computed in float32 and cast to the weight's dtype once,
so the merged model runs through the batched engine like any other.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRALinear", "inject_lora", "trainable_parameters", "ATTENTION_MLP",
           "ADAPTER_DTYPE", "adapter_state", "merge_into", "save_adapter", "load_adapter"]

# Qwen2's attention and MLP projections: "attention + MLP" in the plan.
ATTENTION_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ADAPTER_DTYPE = torch.float32


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` with a trainable rank-``r`` update beside it."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 generator: torch.Generator | None = None,
                 dtype: torch.dtype = ADAPTER_DTYPE) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        a = torch.empty(rank, base.in_features, dtype=dtype)  # on the CPU: the same draw anywhere
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        device = base.weight.device
        self.A = nn.Parameter(a.to(device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype).to(device))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        return self.base(x) + F.linear(F.linear(x, self.A.to(dt)), self.B.to(dt)) * self.scaling


def inject_lora(model: nn.Module, targets: Iterable[str], *, rank: int, alpha: float,
                generator: torch.Generator | None = None,
                dtype: torch.dtype = ADAPTER_DTYPE) -> dict[str, LoRALinear]:
    """Freeze ``model`` and wrap every child ``nn.Linear`` named in ``targets``.

    Returns the adapters by module name (``model.layers.0.self_attn.q_proj``), in
    ``named_modules`` order, which is also the order their ``A`` were drawn in."""
    wanted = set(targets)
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped: dict[str, LoRALinear] = {}
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in wanted and isinstance(child, nn.Linear):
                adapter = LoRALinear(child, rank, alpha, generator, dtype)
                setattr(parent, child_name, adapter)
                wrapped[f"{parent_name}.{child_name}" if parent_name else child_name] = adapter
    missing = wanted - {name.rsplit(".", 1)[-1] for name in wrapped}
    if missing:
        raise ValueError(f"no nn.Linear named {sorted(missing)} found in the model")
    return wrapped


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# -- saving, loading, merging -------------------------------------------------------------


def adapter_state(wrapped: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    """``{"<module>.A": A, "<module>.B": B}`` on the CPU, detached (what is saved)."""
    out = {}
    for name, m in wrapped.items():
        out[f"{name}.A"] = m.A.detach().to("cpu").contiguous()
        out[f"{name}.B"] = m.B.detach().to("cpu").contiguous()
    return out


def save_adapter(path, state: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    """The unmerged adapter as safetensors (post 10 needs it unmerged). safetensors
    writes ``metadata`` in no fixed order, so compare adapters by their tensors
    (``train.tensors_sha256``), not by the file's bytes."""
    from safetensors.torch import save_file

    save_file(state, str(path), metadata=metadata)


def load_adapter(path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def module_names(state: dict[str, torch.Tensor]) -> list[str]:
    """The adapted modules of a saved state, in its order; every one has an A and a B."""
    names = list(dict.fromkeys(k.rsplit(".", 1)[0] for k in state))
    for n in names:
        if f"{n}.A" not in state or f"{n}.B" not in state:
            raise ValueError(f"adapter for {n} lacks A or B")
    if len(state) != 2 * len(names):
        raise ValueError("adapter has keys other than <module>.A / <module>.B")
    return names


def merged_weight(weight: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                  scaling: float) -> torch.Tensor:
    """``W + scaling * B A``, computed in float32 and cast to ``weight``'s dtype once."""
    a = A.to(torch.float32).to(weight.device)  # dtype, then device: two steps
    b = B.to(torch.float32).to(weight.device)
    return (weight.float() + scaling * (b @ a)).to(weight.dtype)


@torch.no_grad()
def merge_into(model: nn.Module, state: dict[str, torch.Tensor], scaling: float) -> int:
    """Fold a saved adapter into ``model``'s plain ``nn.Linear`` weights, in place.
    Returns how many modules were merged (every adapted module must exist)."""
    names = module_names(state)
    for n in names:
        lin = model.get_submodule(n)
        if not isinstance(lin, nn.Linear):
            raise ValueError(f"{n} is not an nn.Linear")
        A, B = state[f"{n}.A"], state[f"{n}.B"]
        r = A.shape[0]
        if tuple(A.shape) != (r, lin.in_features) or tuple(B.shape) != (lin.out_features, r):
            raise ValueError(f"{n}: adapter shapes {tuple(A.shape)}, {tuple(B.shape)} don't fit "
                             f"{lin.out_features} x {lin.in_features}")
        lin.weight.copy_(merged_weight(lin.weight, A, B, scaling))
    return len(names)
