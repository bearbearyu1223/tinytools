"""The SFT trainer for post 9 (plan 3e, "T8 spec"; pre-registered, the same for both arms).

One run trains a LoRA adapter on one SFT set (``data/sft/real.jsonl``, or a seed's
``control-seed<N>.jsonl``) and saves it unmerged, with its loss log and a record.

The settings (:data:`MAIN`) are fixed before any result and printed in r4c:

* LoRA rank 16, alpha 32, dropout 0, on q, k, v, o, gate, up and down in every layer;
  A, B, their gradients and AdamW's state in float32, base weights in the device's
  dtype (bf16 on the GPU), frozen (``lora.py``);
* 2 epochs; AdamW (0.9, 0.999), eps 1e-8, weight decay 0; lr 2e-4, cosine with 3%
  warmup (Hugging Face's ``get_cosine_schedule_with_warmup``: warmup steps =
  ceil(3% of the steps), the multiplier step / warmup during warmup, so the first
  step runs at lr 0, then a half cosine to 0); no gradient clipping;
* 16 rows per optimizer step (the effective batch); a step's rows are split into
  micro-batches by a token budget (rows sorted by length within the step, longest
  first, packed while rows x longest <= budget; gradients accumulated), so a step
  is the same 16 rows whatever the budget;
* the loss is the mean over the step's scored tokens (every assistant turn from
  after its header through ``<|im_end|>``, as ``sft-data`` counted them), so a
  step's gradient doesn't depend on how it was split into micro-batches;
* max length 3,072 tokens, no truncation: a longer row stops the run (0 are
  longer; the longest SFT row is 2,975).

**What a seed does.** The real set has no randomness, so seed N sets two things:
LoRA A's initial values (a CPU ``torch.Generator`` seeded N, drawn in module order)
and the shuffle order (``Random("<N>:shuffle:<epoch>")``, a permutation of row
positions per epoch). Real N and control N share both (their rows are twins in the
same order, checked), which is what makes them a pair.

Everything above the ``-- running`` line is pure and tested in ``tests/test_train.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .lora import ATTENTION_MLP

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QUICK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ARMS = ("real", "control")


@dataclass(frozen=True)
class Settings:
    model: str = MODEL
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ATTENTION_MLP
    epochs: int = 2
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    batch_rows: int = 16
    max_length: int = 3072
    token_budget: int = 8192
    grad_checkpointing: bool = False
    rows: int | None = None  # use only the first N rows of the set (quick mode)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


MAIN = Settings()
# Quick mode (a Mac check, not results): the small model, the first 32 rows (2 steps an
# epoch, the first at lr 0 as the schedule starts; ~30 s a step in float32 on MPS), a
# smaller budget for MPS memory; every other setting as MAIN.
QUICK = replace(MAIN, model=QUICK_MODEL, rows=32, token_budget=4096)

# The settings two runs must share to be compared (a pair, or a run and its repeat).
PAIRED = ("model", "rank", "alpha", "dropout", "targets", "epochs", "lr", "betas", "eps",
          "weight_decay", "warmup_ratio", "batch_rows", "max_length", "rows")


# -- planning (pure) ----------------------------------------------------------------------


def shuffle_order(n: int, seed: int, epoch: int) -> list[int]:
    """Row positions 0..n-1 in seed N's order for one epoch."""
    order = list(range(n))
    random.Random(f"{seed}:shuffle:{epoch}").shuffle(order)
    return order


@dataclass(frozen=True)
class Step:
    epoch: int
    rows: tuple[int, ...]  # row positions, in shuffled order


def plan_steps(n: int, seed: int, epochs: int, batch_rows: int) -> list[Step]:
    """Every optimizer step: each epoch's shuffled rows, ``batch_rows`` at a time (the
    last step of an epoch takes what is left)."""
    if n < 1 or batch_rows < 1 or epochs < 1:
        raise ValueError("need rows, a batch and an epoch")
    out = []
    for e in range(epochs):
        order = shuffle_order(n, seed, e)
        out += [Step(e, tuple(order[k:k + batch_rows])) for k in range(0, n, batch_rows)]
    return out


def micro_batches(rows: tuple[int, ...] | list[int], lengths: list[int],
                  budget: int) -> list[list[int]]:
    """A step's rows split by a token budget: sorted longest first (ties by position),
    packed while (rows in the micro-batch) x (its longest row) <= ``budget``. A row
    longer than the budget would be alone; the caller keeps the budget >= max length."""
    order = sorted(rows, key=lambda i: (-lengths[i], i))
    out: list[list[int]] = []
    for i in order:
        if out and (len(out[-1]) + 1) * lengths[out[-1][0]] <= budget:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def warmup_steps(total: int, ratio: float) -> int:
    """ceil(ratio x total), as Hugging Face's Trainer computes it from warmup_ratio."""
    return math.ceil(total * ratio)


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    """Hugging Face's cosine schedule with warmup (num_cycles 0.5), for optimizer step
    ``step`` (0-based): linear from 0 during warmup, then half a cosine to 0."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def padded_tokens(batches: list[list[int]], lengths: list[int]) -> int:
    """Token slots the micro-batches compute (rows x longest row, summed)."""
    return sum(len(b) * max(lengths[i] for i in b) for b in batches)


def rows_over(lengths: list[int], max_length: int) -> int:
    return sum(1 for n in lengths if n > max_length)


def labels_for(ids: list[int], mask: list[bool], ignore: int = -100) -> list[int]:
    """The token ids where the loss is computed, ``ignore`` elsewhere."""
    if len(ids) != len(mask):
        raise ValueError("one mask entry per token")
    return [t if m else ignore for t, m in zip(ids, mask, strict=True)]


def same_rows(a: list[dict], b: list[dict]) -> bool:
    """Two sets hold the same rows in the same order (a real set and its control twins)."""
    return [r["id"] for r in a] == [r["id"] for r in b]


# -- the deadline (pure) --------------------------------------------------------------------

SAFETY = 1.5  # an estimate is multiplied by this before it is compared with the time left
# Before any training is measured: Milestone 0's A100 probe, 0.297 s per step of 4 x 1,024
# tokens (1.5B LoRA r16, bf16, full-vocabulary logits), per token.
PRIOR_S_PER_TOKEN = 0.297 / 4096
REPORT_ALLOW_S = 900  # kept back for the report and the untrained rescoring after the pairs


def estimate_seconds(train_rates: list[float], score_walls: list[float], padded_tokens: int,
                     trainings: int, scorings: int, checkpointing: bool,
                     score_prior_s: float) -> float:
    """Seconds for ``trainings`` runs of ``padded_tokens`` token slots and ``scorings``
    scorings: at the slowest measured seconds per token slot (loading included) else the
    prior (+40% with gradient checkpointing), and the slowest measured scoring else
    ``score_prior_s``. :func:`fits` applies the safety factor."""
    rate = max(train_rates) if train_rates else PRIOR_S_PER_TOKEN * (1.4 if checkpointing
                                                                     else 1.0)
    score = max(score_walls) if score_walls else score_prior_s
    return trainings * rate * padded_tokens + scorings * score


def fits(now: float, deadline: float | None, need_s: float, keep_s: float = 0.0) -> bool:
    """Can work of ``need_s`` x SAFETY start now and end ``keep_s`` before the deadline?"""
    return deadline is None or now + SAFETY * need_s + keep_s <= deadline


# -- running ---------------------------------------------------------------------------------


@dataclass
class Encoded:
    ids: list[list[int]]
    masks: list[list[bool]]
    lengths: list[int]
    scored: list[int]


def encode(tok, records: list[dict]) -> Encoded:
    """Token ids and the scored mask of each record, as ``sft-data`` counted them
    (checked: the same token count and scored count as the record says)."""
    from .milestone0 import token_mask

    out = Encoded([], [], [], [])
    for r in records:
        enc = tok(r["text"], return_offsets_mapping=True, add_special_tokens=False)
        mask = token_mask([tuple(o) for o in enc["offset_mapping"]],
                          [tuple(x) for x in r["train_spans"]])
        if len(enc["input_ids"]) != r["n_tokens"] or sum(mask) != r["n_trained"]:
            raise ValueError(f"{r['id']}: {len(enc['input_ids'])} tokens / {sum(mask)} scored, "
                             f"but the record says {r['n_tokens']} / {r['n_trained']}")
        out.ids.append(enc["input_ids"])
        out.masks.append(mask)
        out.lengths.append(len(enc["input_ids"]))
        out.scored.append(sum(mask))
    return out


def load_set(path: Path, rows: int | None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if rows is not None and len(out) >= rows:
                break
            out.append(json.loads(line))
    return out


def _collate(enc: Encoded, batch: list[int], pad: int, device):
    """Right-padded ids, attention mask and labels for one micro-batch (int64 built on
    the CPU, then moved: device only, no dtype change, the MPS bug)."""
    import torch

    width = max(enc.lengths[i] for i in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.int64)
    attn = torch.zeros((len(batch), width), dtype=torch.int64)
    labels = torch.full((len(batch), width), -100, dtype=torch.int64)
    for k, i in enumerate(batch):
        n = enc.lengths[i]
        ids[k, :n] = torch.tensor(enc.ids[i], dtype=torch.int64)
        attn[k, :n] = 1
        labels[k, :n] = torch.tensor(labels_for(enc.ids[i], enc.masks[i]), dtype=torch.int64)
    return ids.to(device), attn.to(device), labels.to(device)


def token_loss_sum(model, ids, attn, labels):
    """Summed cross-entropy over the scored tokens. Logits only at the scored positions
    (about 10% of tokens), in float32: the full (tokens x 152k) logits would not fit
    the token budget."""
    import torch.nn.functional as F

    hidden = model.model(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
    target = labels[:, 1:]
    keep = target != -100
    logits = model.lm_head(hidden[:, :-1][keep]).float()
    return F.cross_entropy(logits, target[keep], reduction="sum")


@dataclass
class Result:
    arm: str
    seed: int
    settings: dict
    rows: int
    steps: int
    warmup: int
    micro_batches: int
    tokens: int
    scored: int
    padded_tokens: int
    rows_cut: int
    longest: int
    trainable: int
    dtypes: dict
    first_step_rows: list[str]  # the first step's row ids, in order (the seed-pair test)
    init_sha256: str  # tensors_sha256 of A's initial values (B starts at 0)
    adapter_sha256: str  # tensors_sha256 of the trained adapter
    data_sha256: str
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""


def tensors_sha256(state: dict) -> str:
    """A digest of an adapter's tensors (sorted names, dtypes, shapes, bytes). Not the
    file's sha256: safetensors writes its metadata map in no fixed order, so two equal
    adapters can be different files."""
    import torch

    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().to("cpu").contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train(s: Settings, arm: str, seed: int, set_path: Path, out_dir: Path, device,
          real_path: Path | None = None, on_step=None, label: str | None = None) -> Result:
    """Train one adapter and write ``adapter.safetensors``, ``losses.jsonl`` and
    ``train.json`` to ``out_dir`` (each written whole, then renamed into place)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import progress
    from .device import dtype_name, preferred_dtype
    from .lora import adapter_state, inject_lora, save_adapter, trainable_parameters

    t0 = time.perf_counter()
    records = load_set(set_path, s.rows)
    if real_path is not None and not same_rows(load_set(real_path, s.rows), records):
        raise ValueError(f"{set_path.name} does not hold the real set's rows in its order")
    data_sha = hashlib.sha256(set_path.read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(s.model)
    enc = encode(tok, records)
    cut = rows_over(enc.lengths, s.max_length)
    if cut:
        raise ValueError(f"{cut} rows are longer than {s.max_length} tokens; nothing is truncated")
    if s.token_budget < max(enc.lengths):
        raise ValueError(f"token budget {s.token_budget} < the longest row ({max(enc.lengths)})")
    steps = plan_steps(len(records), seed, s.epochs, s.batch_rows)
    warm = warmup_steps(len(steps), s.warmup_ratio)
    batches = [micro_batches(st.rows, enc.lengths, s.token_budget) for st in steps]

    torch.manual_seed(seed)  # nothing else draws from it (dropout 0); set anyway
    model = AutoModelForCausalLM.from_pretrained(s.model, dtype=preferred_dtype(device))
    model.to(device)
    gen = torch.Generator().manual_seed(seed)
    wrapped = inject_lora(model, s.targets, rank=s.rank, alpha=s.alpha, generator=gen)
    init_sha = tensors_sha256({k: v for k, v in adapter_state(wrapped).items()
                               if k.endswith(".A")})
    params = trainable_parameters(model)
    if s.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # no dropout in Qwen2 (attention_dropout 0); gates checkpointing only
    opt = torch.optim.AdamW(params, lr=s.lr, betas=s.betas, eps=s.eps,
                            weight_decay=s.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lr_multiplier(k, warm, len(steps)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    on_step = on_step or progress.training(label or f"{arm} seed {seed}", len(steps))
    out_dir.mkdir(parents=True, exist_ok=True)
    losses_tmp = out_dir / ".losses.jsonl.tmp"
    note = ""
    losses: list[float] = []
    dtypes: dict = {}
    with open(losses_tmp, "w", encoding="utf-8") as log:
        for k, (st, mbs) in enumerate(zip(steps, batches, strict=True)):
            n_scored = sum(enc.scored[i] for i in st.rows)
            total = 0.0
            for mb in mbs:
                ids, attn, labels = _collate(enc, mb, pad, device)
                try:
                    loss = token_loss_sum(model, ids, attn, labels)
                    (loss / n_scored).backward()
                except RuntimeError as e:  # an op with no deterministic CUDA kernel
                    # the same ops run in every micro-batch: only the first can hit this
                    if "deterministic" not in str(e) or note or k or mb is not mbs[0]:
                        raise
                    note = "warn_only: " + str(e).split("\n")[0][:120]
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    opt.zero_grad(set_to_none=True)
                    loss = token_loss_sum(model, ids, attn, labels)
                    (loss / n_scored).backward()
                total += float(loss.detach())
            lr_now = opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
            if k == 0:
                a = next(iter(wrapped.values()))
                st0 = opt.state[a.A]
                dtypes = {"base weights": dtype_name(a.base.weight.dtype),
                          "activations (compute)": dtype_name(a.base.weight.dtype),
                          "LoRA A, B": dtype_name(a.A.dtype),
                          "gradients of A, B": dtype_name(a.A.grad.dtype),
                          "AdamW state": dtype_name(st0["exp_avg"].dtype)}
            opt.zero_grad(set_to_none=True)
            losses.append(total / n_scored)
            log.write(json.dumps({"step": k + 1, "epoch": st.epoch + 1, "loss": losses[-1],
                                  "lr": lr_now, "rows": len(st.rows), "scored": n_scored,
                                  "micro_batches": len(mbs)}) + "\n")
            on_step(k, losses[-1])
    state = adapter_state(wrapped)
    meta = {"base_model": s.model, "arm": arm, "seed": str(seed), "rank": str(s.rank),
            "alpha": str(s.alpha), "targets": ",".join(s.targets), "data_sha256": data_sha}
    tmp = out_dir / ".adapter.safetensors.tmp"
    save_adapter(tmp, state, meta)
    adapter_sha = tensors_sha256(state)
    tmp.replace(out_dir / "adapter.safetensors")
    losses_tmp.replace(out_dir / "losses.jsonl")
    res = Result(arm, seed, {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in asdict(s).items()},
                 len(records), len(steps), warm, sum(len(b) for b in batches),
                 sum(enc.lengths), sum(enc.scored),
                 sum(padded_tokens(b, enc.lengths) for b in batches),
                 cut, max(enc.lengths), sum(p.numel() for p in params), dtypes,
                 [records[i]["id"] for i in steps[0].rows], init_sha, adapter_sha, data_sha,
                 losses, time.perf_counter() - t0, note)
    write_json(out_dir / "train.json", asdict(res))
    if note and device.type == "cuda":
        torch.use_deterministic_algorithms(True)  # back to strict for what follows
    del model, opt
    return res


def write_json(path: Path, obj) -> None:
    """Written whole, then renamed into place: a fetch never sees half a file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
