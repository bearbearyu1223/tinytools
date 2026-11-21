"""The one generation engine: batched greedy decoding and T=1 sampling, our own loop.

Every score this repo prints (``score``, ``bfcl-score``, ``milestone0``, and later the
training evals) generates through :func:`generate`. It replaces Hugging Face
``generate`` called one item at a time (kept as ``bfcl.generate_text`` only so
``engine-check`` can compare the two).

How a call runs:

* Prompts are tokenized, and items are sorted by prompt length, longest first (so a
  batch holds prompts of similar length, padding is short, and the largest batch,
  the one most likely to run out of memory, runs first). Results come back in the
  caller's order.
* A batch is ``batch_size`` sequences (rows). With ``samples`` replies per item, an
  item's rows stay together in one batch: ``batch_size // samples`` items per batch
  (at least one).
* Rows are padded on the left, with the attention mask, and positions counted from
  each row's first real token, so a padded row sees what an unpadded one does.
* One forward pass per new token, with the KV cache. A row stops at the model's
  eos or ``<|im_end|>`` (``bfcl.stop_ids``); a finished row is fed padding and no
  longer counts; the batch ends when every row has finished or at the token cap.
* Greedy is the argmax of the float32 logits. Sampling at temperature T is the
  full softmax of logits / T (top_p 1, top_k off, repetition penalty 1).

**Per-item reproducibility for sampling.** Each row (item ``id``, sample ``j``) has
its own random number generator, Python's ``random.Random`` seeded with the string
``f"{seed}:{id}:{j}"``, and draws one integer from it per token it generates,
nothing else: always 96 random bits per token. So a reply's draws depend only on the
seed, its item id and sample number: not on its batch, its position in it, the batch
size, or what ran before. The draw picks a token by inverse CDF over the
probabilities in fixed point: each probability is floored to a multiple of 2^-40, the
integers are summed with an integer cumulative sum on the device (exact, and allowed
under CUDA's deterministic mode, where a floating-point cumsum raises), the 96 bits
are scaled to a uniform integer below the total (:func:`scale_draw`), and the token
is the first whose running total exceeds it (:func:`pick_token`).
Tokens below 2^-40 (about 1e-12) can't be drawn; every other token is drawn with
its probability to within 2^-40.

What this does not promise: the probabilities themselves come from a forward pass
whose floating-point rounding can depend on the batch's shape (padding, number of
rows), so across batch sizes a reply can still differ, rarely, where a draw lands
on a boundary or greedy's top two logits are within rounding. Same batch size, same
items, same device: byte-identical (checked by ``engine-check``).

The arithmetic (:func:`plan_batches`, :func:`fixed_point`, :func:`pick_token`,
:func:`first_stop`, :func:`row_seed`, :func:`scale_draw`) is pure and tested in ``tests/test_engine.py``.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch

# Fixed-point scale for sampling: probabilities are floored to multiples of 2^-SCALE_BITS.
SCALE_BITS = 40
# Random bits per token drawn: always this many, so a reply's generator advances the same
# way whatever the probabilities (a draw by randrange(total) would not: its bit count and
# its rejection loop depend on total, which rounding can move across 2^40).
DRAW_BITS = 96

# Default batch sizes (sequences per forward pass), by device type. CUDA: an A100 holds
# 64 rows of a 3B model's KV cache at our prompt lengths (~0.4-2k tokens) many times
# over, and a decode step at 0.5-3B is bound by per-step overhead, so more rows are
# nearly free. MPS (0.5B, fp32, measured 2026-11-21): a decode step slows with every
# row once a batch has padding (16 rows: 53 ms unpadded, 130 ms padded), so greedy on
# 60 tasks took 61 / 56 / 55 s at 8 / 16 / 32 rows; 8 keeps a sampled item's 8 rows
# alone in their batch (equal lengths: no padding). CPU: tests only.
DEFAULT_BATCH = {"cuda": 64, "mps": 8, "cpu": 4}

ENGINE = "batched"


def default_batch_size(device_type: str) -> int:
    return DEFAULT_BATCH.get(device_type, DEFAULT_BATCH["cpu"])


# -- pure parts --------------------------------------------------------------------------


def row_seed(seed: int, item_id: str, sample: int) -> str:
    """The string a reply's generator is seeded with (``random.Random`` hashes it with
    SHA-512, so it is the same on every platform and run)."""
    return f"{seed}:{item_id}:{sample}"


def plan_batches(lengths: list[int], samples: int, batch_size: int) -> list[list[int]]:
    """Item indices per batch: sorted by prompt length, longest first (ties in the
    caller's order), ``max(1, batch_size // samples)`` items per batch."""
    if batch_size < 1 or samples < 1:
        raise ValueError("batch size and samples must be at least 1")
    per = max(1, batch_size // samples)
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    return [order[k:k + per] for k in range(0, len(order), per)]


def first_stop(tokens: list[int], stop: set[int]) -> int | None:
    """Index of the first stop token, or ``None`` (the reply ran to the cap)."""
    return next((i for i, t in enumerate(tokens) if t in stop), None)


def fixed_point(probs: torch.Tensor) -> torch.Tensor:
    """Float probabilities (rows) -> int64 running totals in units of 2^-SCALE_BITS.

    ``probs * 2^40`` is exact in float32 (a power of two); the floor is converted
    to int64 on the tensor's own device (dtype only, no device move: the MPS bug
    is in the fused device-and-dtype move)."""
    q = torch.floor(probs.float() * float(2**SCALE_BITS)).to(torch.int64)
    return torch.cumsum(q, dim=-1)


def scale_draw(bits: int, total: int) -> int:
    """A ``DRAW_BITS``-bit random integer -> an integer in ``[0, total)``: floor(bits x
    total / 2^DRAW_BITS). Every value gets floor or ceil of 2^96 / total of the 2^96
    inputs, so it is uniform to within total / 2^96 (about 2^-56) relative."""
    return (bits * total) >> DRAW_BITS


def pick_token(running: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    """Per row, the first index whose running total exceeds the draw.

    ``running``: (rows, vocab) int64 cumulative sums; ``draws``: (rows,) int64, each
    in ``[0, total)``. A token with running increment q is picked by exactly q of
    the ``total`` possible draws, so with a uniform draw its chance is q / total.
    """
    idx = torch.searchsorted(running, draws.unsqueeze(-1), right=True).squeeze(-1)
    return idx.clamp(max=running.shape[-1] - 1)


# -- the loop ------------------------------------------------------------------------------


@dataclass
class Generated:
    """One item's replies (``samples`` of them, in sample order)."""

    texts: list[str]
    new_tokens: list[int]
    capped: list[bool]
    seconds: float  # its batch's wall seconds x (its rows / the batch's rows)


@dataclass(frozen=True)
class BatchStat:
    rows: int
    items: int
    longest_prompt: int
    shortest_prompt: int
    steps: int  # forward passes after the prefill (new tokens of the longest row)
    seconds: float


@dataclass
class Run:
    items: list[Generated]
    batches: list[BatchStat] = field(default_factory=list)


@dataclass(frozen=True)
class Decoding:
    samples: int
    temperature: float
    max_new_tokens: int
    stop: tuple[int, ...]
    pad: int

    @property
    def greedy(self) -> bool:
        return self.samples == 1 and self.temperature == 0


def decoding(tok, model, samples: int, temperature: float, max_new_tokens: int) -> Decoding:
    from .bfcl import END_OF_TURN, stop_ids

    stop = stop_ids(model.generation_config.eos_token_id, tok.convert_tokens_to_ids(END_OF_TURN))
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop[0]
    return Decoding(samples, temperature, max_new_tokens, tuple(stop), pad)


def _left_pad(rows: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.int64)
    mask = torch.zeros((len(rows), width), dtype=torch.int64)
    for i, r in enumerate(rows):
        ids[i, width - len(r):] = torch.tensor(r, dtype=torch.int64)
        mask[i, width - len(r):] = 1
    return ids, mask


@torch.no_grad()
def _decode(model, rows: list[list[int]], dec: Decoding,
            rngs: list[random.Random] | None) -> tuple[list[list[int]], int]:
    """New tokens per row (through its stop token or to the cap) and the steps taken."""
    device = next(model.parameters()).device
    ids, mask = _left_pad(rows, dec.pad)
    ids, mask = ids.to(device), mask.to(device)  # int64 stays int64: move only
    pos = (mask.cumsum(-1) - 1).clamp(min=0)  # integer cumsum: deterministic on CUDA
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                logits_to_keep=1)
    stop = torch.tensor(dec.stop, dtype=torch.int64).to(device)
    finished = torch.zeros(len(rows), dtype=torch.bool, device=device)
    next_pos = pos[:, -1:] + 1
    cache = out.past_key_values
    produced = []
    steps = 0
    for step in range(dec.max_new_tokens):
        logits = out.logits[:, -1, :].float()
        if rngs is None:
            tok = logits.argmax(dim=-1)
        else:
            running = fixed_point(torch.softmax(logits / dec.temperature, dim=-1))
            both = torch.stack([running[:, -1], finished.to(torch.int64)]).tolist()  # one sync
            totals, done = both[0], both[1]
            draws = [0 if d else scale_draw(rngs[i].getrandbits(DRAW_BITS), totals[i])
                     for i, d in enumerate(done)]
            tok = pick_token(running, torch.tensor(draws, dtype=torch.int64).to(device))
        tok = torch.where(finished, torch.full_like(tok, dec.pad), tok)
        produced.append(tok)
        finished = finished | (tok.unsqueeze(-1) == stop).any(-1)
        steps = step + 1
        if bool(finished.all()) or steps == dec.max_new_tokens:
            break
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
        out = model(input_ids=tok.unsqueeze(-1), attention_mask=mask, position_ids=next_pos,
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_pos = next_pos + 1
    new = torch.stack(produced, dim=1).cpu().tolist()
    return new, steps


def generate(tok, model, keys: list[str], prompts: list[str], *, samples: int,
             temperature: float, seed: int, batch_size: int, max_new_tokens: int,
             on_done=None) -> Run:
    """Generate ``samples`` replies for each prompt; ``keys[i]`` (the item id) seeds its
    sampling. Returns one :class:`Generated` per prompt, in the given order, and one
    :class:`BatchStat` per batch. ``on_done(k)``: k items finished so far."""
    from .device import sync

    if len(keys) != len(prompts):
        raise ValueError("one key per prompt")
    dec = decoding(tok, model, samples, temperature, max_new_tokens)
    if samples > 1 and temperature <= 0:
        raise ValueError("sampling needs a temperature above 0")
    if dec.greedy and samples != 1:
        raise ValueError("greedy is one reply per item")
    device = next(model.parameters()).device
    encoded = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    plan = plan_batches([len(e) for e in encoded], samples, batch_size)
    stop = set(dec.stop)
    results: list[Generated | None] = [None] * len(prompts)
    stats = []
    done = 0
    for batch in plan:
        rows = [encoded[i] for i in batch for _ in range(samples)]
        rngs = None if dec.greedy else [random.Random(row_seed(seed, keys[i], j))
                                        for i in batch for j in range(samples)]
        sync(device)
        t0 = time.perf_counter()
        new, steps = _decode(model, rows, dec, rngs)
        sync(device)
        seconds = time.perf_counter() - t0
        for k, i in enumerate(batch):
            texts, lengths, capped = [], [], []
            for row in new[k * samples:(k + 1) * samples]:
                n = first_stop(row, stop)
                capped.append(n is None)
                n = len(row) if n is None else n
                lengths.append(n)
                texts.append(tok.decode(row[:n], skip_special_tokens=False))
            results[i] = Generated(texts, lengths, capped, seconds * samples / len(rows))
        lens = [len(encoded[i]) for i in batch]
        stats.append(BatchStat(len(rows), len(batch), max(lens), min(lens), steps, seconds))
        done += len(batch)
        if on_done:
            on_done(done)
    return Run(results, stats)  # type: ignore[arg-type]


# -- what a capture's Setup says ----------------------------------------------------------


def batch_text(batch_size: int, samples: int) -> str:
    """The batch line's value: rows per batch, and items x samples when sampling."""
    if samples == 1:
        return f"{batch_size} sequences"
    per = max(1, batch_size // samples)
    return f"{per * samples} sequences ({per} item{'s' if per > 1 else ''} x {samples})"


def print_setup(rep, batch_size: int, samples: tuple[int, ...] = (1,)) -> None:
    """The Setup lines a batched capture carries (one batch line per decoding used)."""
    rep.kv("engine", f"{ENGINE}, sorted by length")
    for g in samples:
        label = "  batch size" if len(samples) == 1 else (
            "  batch size, greedy" if g == 1 else f"  batch size, {g} samples")
        rep.kv(label, batch_text(batch_size, g))
    rep.note("Left padding with the attention mask; a reply ends at the model's")
    rep.note("eos or <|im_end|>. Seconds per item: the batch's seconds x the")
    rep.note("item's share of its rows.")


def recorded(rows: list[dict]) -> tuple[str, int] | None:
    """(engine, batch size) a replies file was generated with; ``None`` for files from
    the one-at-a-time engine, whose rows carry neither field."""
    engines = {(r.get("engine"), r.get("batch_size")) for r in rows}
    if engines == {(None, None)}:
        return None
    if len(engines) != 1 or None in next(iter(engines)):
        raise ValueError(f"mixed or partial engine fields in one replies file: {engines}")
    return next(iter(engines))
