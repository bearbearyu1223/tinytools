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

**Embedding rows (post 9's base arm, plan 3f).** In Qwen2.5 base, ``<|im_start|>``,
``<|im_end|>``, ``<tool_call>`` and ``</tool_call>`` have the same embedding, and the
embedding is tied to the output head, so from any hidden state "end the turn" and
"start a call" score the same and no LoRA update can separate them.
:func:`inject_rows` adds one trainable float32 ``delta`` (k rows x hidden, zero at the
start) that both sides see: the input embedding of token ``ids[j]`` becomes
``E[ids[j]] + delta[j]`` (:class:`DeltaEmbedding`) and its output logit
``h . (E[ids[j]] + delta[j])`` (:class:`DeltaHead`), one parameter for both, as the
tied weight is. Both additions are a one-hot matrix product (no scatter or
``index_add``, which have no deterministic CUDA kernel), in the activations' dtype as
LoRA's path is. The delta is saved in the adapter file (``embed_rows.delta`` and
``embed_rows.ids``) and :func:`merge_into` writes ``E + delta`` into the tied weight's
rows, in float32, cast once. Adapters without those keys (the Instruct arm's) load and
merge exactly as before.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRALinear", "inject_lora", "trainable_parameters", "ATTENTION_MLP",
           "ADAPTER_DTYPE", "adapter_state", "merge_into", "save_adapter", "load_adapter",
           "RowDelta", "inject_rows", "merge_rows", "split_rows", "ROWS_DELTA", "ROWS_IDS"]

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


# -- embedding rows (the base arm) ---------------------------------------------------------

ROWS_DELTA = "embed_rows.delta"
ROWS_IDS = "embed_rows.ids"


class RowDelta(nn.Module):
    """A trainable float32 addition to ``len(ids)`` rows of a tied embedding, zero at the
    start. ``enabled = False`` turns it off (the merge check's base model)."""

    def __init__(self, ids: list[int], hidden: int, vocab: int, device,
                 dtype: torch.dtype = ADAPTER_DTYPE) -> None:
        super().__init__()
        if len(set(ids)) != len(ids) or not all(0 <= i < vocab for i in ids):
            raise ValueError(f"row ids must be distinct and below {vocab}: {ids}")
        ids_t = torch.tensor(ids, dtype=torch.int64)
        self.register_buffer("ids", ids_t.to(device), persistent=False)  # int64: move only
        select = torch.zeros(len(ids), vocab, dtype=torch.float32)
        select[torch.arange(len(ids)), ids_t] = 1.0  # row j: one-hot at ids[j]
        self.register_buffer("select", select.to(device), persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(ids), hidden, dtype=dtype).to(device))
        self.enabled = True


class DeltaEmbedding(nn.Module):
    """The input embedding with ``delta[j]`` added wherever the token is ``ids[j]``."""

    def __init__(self, base: nn.Embedding, rows: RowDelta) -> None:
        super().__init__()
        self.base = base
        self.rows = rows

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        if not self.rows.enabled:
            return out
        dt = out.dtype
        hit = (input_ids.unsqueeze(-1) == self.rows.ids).to(dt)  # (..., k) one-hot
        return out + hit @ self.rows.delta.to(dt)


class DeltaHead(nn.Module):
    """The tied output head with ``h . delta[j]`` added to the logit of ``ids[j]``."""

    def __init__(self, base: nn.Linear, rows: RowDelta) -> None:
        super().__init__()
        self.base = base
        self.rows = rows

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.rows.enabled:
            return out
        dt = x.dtype
        extra = F.linear(x, self.rows.delta.to(dt))  # (..., k)
        return out + extra @ self.rows.select.to(dt)


def tied(model: nn.Module) -> bool:
    """The output head's weight is the input embedding's (one tensor)."""
    head = model.get_output_embeddings()
    return head is not None and head.weight is model.get_input_embeddings().weight


def inject_rows(model: nn.Module, ids: list[int]) -> RowDelta:
    """Wrap a tied model's input embedding and output head so both see ``E + delta`` on
    the rows ``ids``. Call after :func:`inject_lora` (which freezes every parameter it
    finds); the delta is the only new trainable tensor."""
    emb = model.get_input_embeddings()
    if not isinstance(emb, nn.Embedding) or not tied(model):
        raise ValueError("embedding rows need a plain, tied input embedding and output head")
    head = model.get_output_embeddings()
    rows = RowDelta(ids, emb.embedding_dim, emb.num_embeddings, emb.weight.device)
    model.set_input_embeddings(DeltaEmbedding(emb, rows))
    model.set_output_embeddings(DeltaHead(head, rows))
    return rows


@torch.no_grad()
def randomize_rows(rows: RowDelta, scale: float, generator: torch.Generator) -> None:
    """The delta set to ``scale`` x a standard normal draw (CPU, then moved)."""
    d = torch.randn(tuple(rows.delta.shape), generator=generator, dtype=torch.float32) * scale
    rows.delta.copy_(d.to(rows.delta.dtype).to(rows.delta.device))


def row_similarity(rows: torch.Tensor) -> dict:
    """Norms of k embedding rows, their pairwise cosines (off the diagonal) and the
    largest absolute difference between any two, in float32. Pure."""
    r = rows.detach().float().cpu()
    n = r.norm(dim=1)
    cos = (r @ r.T) / (n[:, None] * n[None, :])
    off = ~torch.eye(len(r), dtype=torch.bool)
    diff = (r[:, None, :] - r[None, :, :]).abs().amax(dim=-1)
    return {"norms": [float(x) for x in n], "cos_min": float(cos[off].min()),
            "cos_max": float(cos[off].max()), "max_abs_diff": float(diff[off].max()),
            "cos_first": [float(x) for x in cos[0]]}


# -- saving, loading, merging -------------------------------------------------------------


def adapter_state(wrapped: dict[str, LoRALinear],
                  rows: RowDelta | None = None) -> dict[str, torch.Tensor]:
    """``{"<module>.A": A, "<module>.B": B}`` on the CPU, detached (what is saved); with
    ``rows``, also ``embed_rows.delta`` (float32) and ``embed_rows.ids`` (int64)."""
    out = {}
    for name, m in wrapped.items():
        out[f"{name}.A"] = m.A.detach().to("cpu").contiguous()
        out[f"{name}.B"] = m.B.detach().to("cpu").contiguous()
    if rows is not None:
        out[ROWS_DELTA] = rows.delta.detach().to("cpu").contiguous()
        out[ROWS_IDS] = rows.ids.detach().to("cpu").contiguous()
    return out


def split_rows(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor],
                                                        tuple[torch.Tensor, torch.Tensor] | None]:
    """(the LoRA tensors, (ids, delta) or ``None``) of a saved state."""
    has = (ROWS_DELTA in state, ROWS_IDS in state)
    if has[0] != has[1]:
        raise ValueError(f"adapter has one of {ROWS_DELTA} and {ROWS_IDS} without the other")
    lora = {k: v for k, v in state.items() if k not in (ROWS_DELTA, ROWS_IDS)}
    return lora, ((state[ROWS_IDS], state[ROWS_DELTA]) if has[0] else None)


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
    """The adapted modules of a saved state, in its order; every one has an A and a B
    (embedding rows, if any, are not modules: :func:`split_rows`)."""
    state, _ = split_rows(state)
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
def randomize_b(wrapped: dict[str, LoRALinear], scale: float,
                generator: torch.Generator) -> None:
    """Every ``B`` set to ``scale`` x a standard normal draw (on the CPU from
    ``generator``, in module order, then moved): an adapter whose merge can be wrong in
    a way B = 0 hides (sft-check's merge check, T8 review S3)."""
    for m in wrapped.values():
        b = torch.randn(tuple(m.B.shape), generator=generator, dtype=torch.float32) * scale
        m.B.copy_(b.to(m.B.dtype).to(m.B.device))


@torch.no_grad()
def merge_rows(model: nn.Module, ids: torch.Tensor, delta: torch.Tensor) -> int:
    """Write ``E[ids[j]] + delta[j]`` (float32, cast to the weight's dtype once) into the
    tied embedding, so the input embedding and the output head both get it. Row by row
    with plain indexing (no index kernel). Returns the rows written."""
    if not tied(model):
        raise ValueError("embedding rows merge into a tied embedding only")
    w = model.get_input_embeddings().weight
    if delta.dim() != 2 or tuple(delta.shape) != (len(ids), w.shape[1]):
        raise ValueError(f"row delta {tuple(delta.shape)} doesn't fit {len(ids)} x {w.shape[1]}")
    for j, i in enumerate(ids.tolist()):
        new = w[i].float() + delta[j].to(torch.float32).to(w.device)  # dtype, then device
        w[i].copy_(new.to(w.dtype))
    return len(ids)


@torch.no_grad()
def merge_into(model: nn.Module, state: dict[str, torch.Tensor], scaling: float) -> int:
    """Fold a saved adapter into ``model``'s plain ``nn.Linear`` weights, in place, and
    its embedding rows, if it has any, into the tied embedding (:func:`merge_rows`).
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
    _, rows = split_rows(state)
    if rows is not None:
        merge_rows(model, *rows)
    return len(names)
