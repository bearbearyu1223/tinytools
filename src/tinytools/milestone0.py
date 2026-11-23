"""Milestone 0: which model post 9 trains, decided by a rule fixed before the numbers.

Six models (Qwen2.5 0.5B / 1.5B / 3B, base and instruct) are scored on the dev
split only: our 200 dev tasks greedy, the same tasks with 8 samples at T=1, and
the BFCL dev slice greedy. The rule (plan 3a, thresholds adopted by the author):

    a candidate qualifies when, all four,
    (a) greedy right by execution match on our dev tasks      <= 70%
    (b) BFCL right on the dev slice's call categories          <= 80%
    (c) mixed groups (some of 8 at T=1 right, some wrong)      >= 10%
        of the dev tasks that need a call
    (d) greedy right by execution match on the dev tasks       >= 20%
        that need a call

(c) and (d) leave out the "no tool fits" tasks, where a reply without a call is
right: a model that never makes a right call must not qualify (review of f13ccbd;
the author's decision of 2026-11-21). A task whose kind is unknown counts as not
needing a call, so it can't help a model qualify.

Pick the smaller of 1.5B / 3B instruct that qualifies; else base 1.5B, then
base 3B, by the same rule; else stop and bring it to the author. 0.5B is scored
as comparison rows only. Also: one model's greedy dev pass run twice (are the
replies byte-identical?), and a training-step timing probe (LoRA vs full).

With ``--deadline-epoch`` (the Lambda runner passes when its remote time limit fires),
a model whose scoring can't finish before it (estimated from what was measured so far,
conservatively) is not started: the rule section is printed from what was measured, with
the verdict "incomplete" if the rule reaches an unscored model. SIGTERM (the time limit,
or the runner stopping the job) does the same with whatever had been measured.

Everything above the ``-- running`` line is model-free and unit-tested in
``tests/test_milestone0.py``.
"""

from __future__ import annotations

import gc
import json
import math
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# -- the models and the rule (pure) ------------------------------------------------


@dataclass(frozen=True)
class Model:
    size: str  # "0.5B", "1.5B", "3B"
    kind: str  # "base" or "instruct"

    @property
    def hf_id(self) -> str:
        return f"Qwen/Qwen2.5-{self.size}" + ("-Instruct" if self.kind == "instruct" else "")

    @property
    def label(self) -> str:
        return f"{self.size} {self.kind}"

    @property
    def key(self) -> str:
        return f"{self.size}-{self.kind}"


I05, B05 = Model("0.5B", "instruct"), Model("0.5B", "base")
I15, B15 = Model("1.5B", "instruct"), Model("1.5B", "base")
I3, B3 = Model("3B", "instruct"), Model("3B", "base")
ALL_MODELS = (I05, B05, I15, B15, I3, B3)
CANDIDATES = (I15, I3)  # in the rule's order: the smaller first
FALLBACKS = (B15, B3)
RULE_ORDER = CANDIDATES + FALLBACKS
# Scoring order: what the rule needs first, so a run cut short by its time
# limit still has the candidates. 0.5B (comparison rows) last.
RUN_ORDER = (I15, I3, B15, B3, I05, B05)
REPEAT_MODEL = I15  # the model whose greedy dev pass is run twice
TIMING_SIZES = ("1.5B", "3B")  # the candidate sizes

OURS_RIGHT_MAX = 70  # percent, (a)
BFCL_RIGHT_MAX = 80  # percent, (b)
MIXED_MIN = 10  # percent, (c)
CALLS_RIGHT_MIN = 20  # percent, (d)
CALL_KINDS = ("choose+fill", "tricky arguments")  # tasks.CHOOSE, tasks.TRICKY: need a call
SAMPLES = 8
TEMPERATURE = 1.0
SEED = 0
BFCL_CALL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple")
BFCL_CATEGORIES = BFCL_CALL_CATEGORIES + ("irrelevance",)


def role(m: Model) -> str:
    if m in CANDIDATES:
        return "candidate"
    if m in FALLBACKS:
        return "fallback"
    return "comparison"


def parse_models(text: str) -> tuple[Model, ...]:
    """``"1.5B-instruct,3B-base"`` -> those models, in :data:`RUN_ORDER`."""
    keys = {m.key: m for m in ALL_MODELS}
    wanted = [k.strip() for k in text.split(",") if k.strip()]
    unknown = [k for k in wanted if k not in keys]
    if unknown or not wanted:
        raise ValueError(f"unknown models {unknown}; choose from {list(keys)}")
    return tuple(m for m in RUN_ORDER if m.key in wanted)


def at_most(k: int, n: int, percent: int) -> bool:
    """k / n <= percent%, in integers (no rounding at the threshold)."""
    return n > 0 and 100 * k <= percent * n


def at_least(k: int, n: int, percent: int) -> bool:
    return n > 0 and 100 * k >= percent * n


@dataclass(frozen=True)
class Measured:
    """The four numbers the rule reads, with their denominators."""

    model: Model
    ours_right: int  # greedy, right by execution match, all dev tasks
    ours_n: int
    bfcl_right: int  # greedy, call categories of the dev slice
    bfcl_n: int
    mixed: int  # call-needing dev tasks with some of 8 right and some wrong
    groups: int  # call-needing dev tasks sampled
    calls_right: int  # greedy, right by execution match, call-needing dev tasks
    calls_n: int

    @property
    def a(self) -> bool:
        return at_most(self.ours_right, self.ours_n, OURS_RIGHT_MAX)

    @property
    def b(self) -> bool:
        return at_most(self.bfcl_right, self.bfcl_n, BFCL_RIGHT_MAX)

    @property
    def c(self) -> bool:
        return at_least(self.mixed, self.groups, MIXED_MIN)

    @property
    def d(self) -> bool:
        return at_least(self.calls_right, self.calls_n, CALLS_RIGHT_MIN)

    @property
    def qualifies(self) -> bool:
        return self.a and self.b and self.c and self.d


@dataclass(frozen=True)
class Verdict:
    outcome: str  # "pick", "stop" or "incomplete"
    model: Model | None
    text: str


def verdict(measured: dict[Model, Measured]) -> Verdict:
    """The rule, applied in its order. A model the rule reaches but that was not
    scored makes the verdict incomplete (the rule can't skip it)."""
    for m in RULE_ORDER:
        if m not in measured:
            return Verdict("incomplete", None,
                           f"no verdict: {m.hf_id} was not scored, and the rule reaches it")
        if measured[m].qualifies:
            step = "candidate" if m in CANDIDATES else "fallback"
            return Verdict("pick", m, f"pick {m.hf_id} (the first {step} that qualifies)")
    return Verdict("stop", None, "stop: no candidate or fallback qualifies; bring it to the author")


def failed_criteria(m: Measured) -> list[str]:
    return [name for name, ok in (("a", m.a), ("b", m.b), ("c", m.c), ("d", m.d)) if not ok]


def takeaway(measured: dict[Model, Measured], v: Verdict) -> str:
    """The sentence under the verdict, built from the numbers."""
    from .ours import share

    def nums(m: Measured) -> str:
        return (f"{share(m.ours_right, m.ours_n)} right on our dev tasks, "
                f"{share(m.bfcl_right, m.bfcl_n)} on BFCL's dev calls, "
                f"{share(m.mixed, m.groups)} mixed groups and "
                f"{share(m.calls_right, m.calls_n)} right on the tasks that need a call")

    reached = []
    for m in RULE_ORDER:
        if m not in measured or measured[m].qualifies:
            break
        reached.append(f"{m.label} fails ({', '.join(failed_criteria(measured[m]))}: "
                       f"{nums(measured[m])})")
    parts = reached[:]
    if v.outcome == "pick":
        parts.append(f"{v.model.label} qualifies ({nums(measured[v.model])})")
    elif v.outcome == "incomplete":
        parts.append(v.text)
    return "; ".join(parts) + "."


def skip_reason(model: Model, measured: dict[Model, Measured],
                base_when_needed: bool) -> str | None:
    """With ``--base-when-needed``: why a base model is not scored (``None``: score it).
    0.5B base is a comparison row only; a fallback is skipped once the rule has picked."""
    if not base_when_needed or model.kind != "base":
        return None
    if model not in FALLBACKS:
        return "a comparison row only"
    v = verdict(measured)
    if v.outcome == "pick":
        return f"the rule had already picked {v.model.hf_id}"
    return None


# -- the deadline (pure) ---------------------------------------------------------------------

SAFETY = 1.5  # an estimate is multiplied by this before it is compared with the time left
DEADLINE_MARGIN_S = 120  # kept back before the deadline: printing the rule, writing records
PROBE_ALLOW_S = 600  # the training-step probe: not started with less than this left


@dataclass(frozen=True)
class Timing:
    """What scoring one model took: wall clock (loading included), the item-passes it
    scored, and its greedy generation (seconds, decode steps) for a per-step rate."""

    model: Model
    wall_s: float
    item_passes: int
    gen_s: float  # all generation, every pass
    greedy_s: float
    greedy_steps: int  # decode steps of the greedy passes' batches (one forward pass each)


def item_passes(model: Model, n_ours: int, n_bfcl: int, repeat_model: Model) -> int:
    """Items a model's scoring generates for: ours greedy, ours sampled (one group of
    8 per task), BFCL greedy, and ours greedy again for the repeat model."""
    return 2 * n_ours + n_bfcl + (n_ours if model == repeat_model else 0)


def batch_count(n_ours: int, n_bfcl: list[int], batch: int, samples: int,
                again: bool) -> int:
    """Batches a model's scoring runs (the batched engine): ours greedy, ours sampled
    (``batch // samples`` tasks per batch), each BFCL category greedy (one call each),
    and ours greedy again for the repeat model."""
    def n(items: int, per: int) -> int:
        return math.ceil(items / max(1, per))
    greedy = n(n_ours, batch)
    return greedy * (2 if again else 1) + n(n_ours, batch // samples) + sum(
        n(c, batch) for c in n_bfcl)


def estimate_seconds(model: Model, passes: int, timings: list[Timing],
                     max_new_tokens: int, batches: int | None = None) -> float | None:
    """A conservative guess at how long scoring ``model`` takes. With a model of the same
    kind measured: the slowest such model's seconds per item-pass. Otherwise (a base
    model after instruct ones, whose replies run to the cap): every batch (``batches``;
    without it, every item-pass as its own sequence) running to the token cap at the
    slowest measured greedy seconds per decode step, plus the largest measured
    loading and checking time. ``None``: nothing measured yet. Times :data:`SAFETY`."""
    if not timings:
        return None
    same = [t for t in timings if t.model.kind == model.kind and t.item_passes]
    if same:
        return SAFETY * passes * max(t.wall_s / t.item_passes for t in same)
    rates = [t.greedy_s / t.greedy_steps for t in timings if t.greedy_steps]
    if not rates:
        return None
    overhead = max(max(0.0, t.wall_s - t.gen_s) for t in timings)
    units = passes if batches is None else batches
    return SAFETY * (units * max_new_tokens * max(rates) + overhead)


def fits(now: float, deadline: float | None, need_s: float | None,
         margin_s: float = DEADLINE_MARGIN_S) -> bool:
    """Can work of ``need_s`` seconds start now and end ``margin_s`` before the deadline?
    No deadline, or no estimate yet: yes (the first model always starts)."""
    if deadline is None:
        return True
    if need_s is None:
        return now + margin_s < deadline
    return now + need_s + margin_s <= deadline


def differing(first: list[tuple[str, str]], second: list[tuple[str, str]]) -> list[str]:
    """Ids whose reply differs between two passes of ``(id, reply)``, in order.
    The passes must cover the same ids in the same order."""
    if [i for i, _ in first] != [i for i, _ in second]:
        raise ValueError("the two passes scored different tasks")
    return [i for (i, a), (_, b) in zip(first, second, strict=True) if a != b]


# -- the timing probe's arithmetic (pure) ------------------------------------------------


def token_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[bool]:
    """Per token: is it scored (wholly inside an assistant span)? As ``sft.count_tokens``."""
    return [any(a >= s and b <= e for s, e in spans) for a, b in offsets]


def last_window(ids: list[int], mask: list[bool], length: int) -> tuple[list[int], list[bool]]:
    """The last ``length`` tokens (a row's assistant turn is at its end)."""
    return ids[-length:], mask[-length:]


def epoch_steps(rows: int, batch: int) -> int:
    return math.ceil(rows / batch)


def epoch_hours(seconds_per_step: float, rows: int, batch: int) -> float:
    return seconds_per_step * epoch_steps(rows, batch) / 3600


# -- settings ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    quick: bool
    models: tuple[Model, ...]
    ours_limit: int | None  # per kind
    bfcl_limit: int | None  # per category
    repeat_model: Model
    timing_sizes: tuple[str, ...]
    batch: int
    seq_len: int
    warmup_steps: int
    timed_steps: int
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lr: float = 1e-5
    base_when_needed: bool = False
    gen_batch: int | None = None  # generation batch size; None: the device's default

    @property
    def tag(self) -> str:
        return "milestone0-quick" if self.quick else "milestone0"


def settings(quick: bool, models: tuple[Model, ...] | None = None,
             base_when_needed: bool = False, gen_batch: int | None = None) -> Settings:
    if quick:  # one item per kind and category: a base model's replies run to the cap
        return Settings(True, models or (I05, B05), ours_limit=1, bfcl_limit=1,
                        repeat_model=I05, timing_sizes=("0.5B",), batch=2, seq_len=256,
                        warmup_steps=1, timed_steps=2, base_when_needed=base_when_needed,
                        gen_batch=gen_batch)
    return Settings(False, models or RUN_ORDER, ours_limit=None, bfcl_limit=None,
                    repeat_model=REPEAT_MODEL, timing_sizes=TIMING_SIZES, batch=4,
                    seq_len=1024, warmup_steps=1, timed_steps=3,
                    base_when_needed=base_when_needed, gen_batch=gen_batch)


# -- running (needs the models) ------------------------------------------------------------


def _free(device) -> None:
    gc.collect()
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()
    elif device.type == "mps":
        import torch

        torch.mps.empty_cache()


def _versions() -> dict:
    import platform
    from importlib.metadata import version

    import torch

    v = {p: version(p) for p in ("torch", "transformers", "bfcl-eval", "numpy")}
    v["python"] = sys.version.split()[0]
    v["platform"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    v["cuda (torch build)"] = torch.version.cuda or "-"
    return v


@dataclass
class ModelResult:
    model: Model
    ours: list  # ours.Scored, greedy
    sampled: list  # ours.Scored, 8 samples
    bfcl: list  # bfcl.CategoryResult per category, greedy
    measured: Measured
    timing: Timing | None = None


def needs_call(scored) -> bool:
    """Does this scored task need a call? Unknown kind: no (it can't help a model qualify)."""
    task = getattr(scored, "task", None)
    return isinstance(task, dict) and task.get("kind") in CALL_KINDS


def _measure(model: Model, greedy, sampled, bfcl_results) -> Measured:
    from .classify import mixed_share

    calls = [r for r in bfcl_results if r.category in BFCL_CALL_CATEGORIES]
    call_sampled = [s for s in sampled if needs_call(s)]
    call_greedy = [s for s in greedy if needs_call(s)]
    _, mixed, _ = mixed_share([s.right for s in call_sampled], SAMPLES)
    return Measured(model, ours_right=sum(s.right for s in greedy), ours_n=len(greedy),
                    bfcl_right=sum(sum(r.right_per_task) for r in calls),
                    bfcl_n=sum(len(r.right_per_task) for r in calls),
                    mixed=mixed, groups=len(call_sampled),
                    calls_right=sum(s.right for s in call_greedy), calls_n=len(call_greedy))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ours_rows(label: str, res: list, setup: dict) -> list[dict]:
    rows = []
    for s in res:
        for i, (text, c, capped) in enumerate(zip(s.replies, s.checks, s.capped, strict=True)):
            rows.append({"id": s.task["id"], "pass": label, "sample": i, "kind": s.task["kind"],
                         "tool": s.task["tool"], "reply": text, "outcome": c.outcome,
                         "execution": c.execution, "exact": c.exact, "hit_cap": capped,
                         "task_seconds": s.seconds, "task_new_tokens": s.new_tokens, **setup})
    return rows


def _print_model(rep, r: ModelResult) -> None:
    from . import tasks
    from .classify import CALL_OUTCOMES, NO_FIT_OUTCOMES, short_label
    from .ours import MIXED_HEADERS, mixed_rows, outcome_rows, pct, total_row

    rep.section(f"{r.model.label} ({role(r.model)}): {r.model.hf_id}")
    rows = outcome_rows(r.ours)
    calls = [x for x in rows if x.kind not in (tasks.NO_FIT, "total")]
    no_fit = [x for x in rows if x.kind == tasks.NO_FIT]
    rep.note("Our dev tasks, greedy: replies by outcome")
    rep.table(["kind", "n", *[short_label(o) for o in CALL_OUTCOMES]],
              [[tasks.SHORT_KIND.get(x.kind, x.kind), x.n,
                *[x.outcomes.get(o, 0) for o in CALL_OUTCOMES]]
               for x in [*calls, total_row(calls)]])
    if no_fit:
        rep.blank()
        rep.table(["kind", "n", *[short_label(o) for o in NO_FIT_OUTCOMES]],
                  [[tasks.SHORT_KIND[x.kind], x.n, *[x.outcomes.get(o, 0) for o in NO_FIT_OUTCOMES]]
                   for x in no_fit])
    rep.blank()
    rep.table(["kind", "right by execution", "right by exact match"],
              [[tasks.SHORT_KIND.get(x.kind, x.kind), pct(x.execution, x.n), pct(x.exact, x.n)]
               for x in rows])

    rep.blank()
    rep.note("BFCL dev slice, greedy: replies by outcome")
    rep.table(["category", "n", *[short_label(o) for o in CALL_OUTCOMES]],
              [[_short(c.category), c.n, *[c.outcomes[o] for o in CALL_OUTCOMES]]
               for c in r.bfcl if c.category in BFCL_CALL_CATEGORIES])
    irr = [c for c in r.bfcl if c.category == "irrelevance"]
    if irr:
        rep.blank()
        rep.table(["category", "n", *[short_label(o) for o in NO_FIT_OUTCOMES]],
                  [[c.category, c.n, *[c.outcomes[o] for o in NO_FIT_OUTCOMES]] for c in irr])
    rep.blank()
    m = r.measured
    rep.kv("BFCL right, call categories", pct(m.bfcl_right, m.bfcl_n))
    rep.kv("BFCL reference answers pass", pct(sum(c.oracle_right for c in r.bfcl),
                                              sum(c.oracle_n for c in r.bfcl)))

    rep.blank()
    rep.note(f"Our dev tasks, {SAMPLES} samples at T={TEMPERATURE:g}: groups with signal")
    right = {k: [s.right for s in r.sampled if s.task["kind"] == k] for k in tasks.KINDS}
    rep.table(["kind", *MIXED_HEADERS],
              mixed_rows([*[(tasks.SHORT_KIND[k], right[k]) for k in tasks.KINDS if right[k]],
                          ("total", [s.right for s in r.sampled])], SAMPLES))

    rep.blank()
    rep.note("Wall clock (varies run to run)")
    wall = []
    for label, res in (("ours greedy", r.ours), (f"ours x{SAMPLES}", r.sampled)):
        n = max(1, len(res))
        wall.append([label, len(res), f"{sum(s.seconds for s in res) / n:.2f}",
                     f"{sum(s.new_tokens for s in res) / n:.0f}", sum(sum(s.capped) for s in res)])
    n = max(1, sum(len(c.right_per_task) for c in r.bfcl))
    wall.append(["bfcl greedy", n, f"{sum(c.seconds for c in r.bfcl) / n:.2f}",
                 f"{sum(c.new_tokens for c in r.bfcl) / n:.0f}", sum(c.hit_cap for c in r.bfcl)])
    rep.table(["pass", "items", "s/item", "new tokens/item", "hit token cap"], wall)


def _short(cat: str) -> str:
    return {"parallel_multiple": "par_multi"}.get(cat, cat)


def _score_model(model: Model, s: Settings, device, dev_tasks: list, setup: dict, out_dir: Path,
                 repeat: dict, batch: int) -> ModelResult:
    from . import bfcl, ours, progress

    t0 = time.perf_counter()
    progress.status(f"loading {model.hf_id} on {device}")
    tok, lm = bfcl.load_model(model.hf_id, device)
    greedy_stats: list = []  # engine.BatchStat of the greedy passes: decode steps
    try:
        greedy = ours.score(tok, lm, dev_tasks, samples=1, temperature=0.0, seed=SEED,
                            batch_size=batch, stats=greedy_stats,
                            on_item=progress.items(f"{model.label}: our dev, greedy",
                                                   len(dev_tasks)))
        if model == s.repeat_model:
            again = ours.score(tok, lm, dev_tasks, samples=1, temperature=0.0, seed=SEED,
                               batch_size=batch,
                               on_item=progress.items(f"{model.label}: our dev, greedy again",
                                                      len(dev_tasks)))
            repeat["model"] = model
            repeat["first"] = [(x.task["id"], x.replies[0]) for x in greedy]
            repeat["second"] = [(x.task["id"], x.replies[0]) for x in again]
            repeat["seconds"] = sum(x.seconds for x in again)
            repeat["outcomes_differ"] = sum(a.checks[0].outcome != b.checks[0].outcome
                                            for a, b in zip(greedy, again, strict=True))
        sampled = ours.score(tok, lm, dev_tasks, samples=SAMPLES, temperature=TEMPERATURE,
                             seed=SEED, batch_size=batch, on_item=progress.items(
                                 f"{model.label}: our dev, {SAMPLES} samples", len(dev_tasks)))
        records: list[dict] = []
        results = []
        for cat in BFCL_CATEGORIES:
            items = bfcl.load_items(cat, s.bfcl_limit, "dev")
            results.append(bfcl.score(tok, lm, items, samples=1, temperature=0.0, seed=SEED,
                                      batch_size=batch, stats=greedy_stats,
                                      on_item=progress.items(f"{model.label}: bfcl dev {cat}",
                                                             len(items)),
                                      records=records, setup=setup))
    finally:
        del lm
        _free(device)
    stem = f"{s.tag}-{model.hf_id.split('/')[-1]}"
    _write_jsonl(out_dir / f"{stem}-ours-replies.jsonl",
                 _ours_rows("greedy", greedy, setup)
                 + _ours_rows(f"samples{SAMPLES}-t{TEMPERATURE:g}", sampled, setup))
    _write_jsonl(out_dir / f"{stem}-bfcl-replies.jsonl", records)
    again = repeat.get("model") == model
    greedy_s = sum(x.seconds for x in greedy) + sum(c.seconds or 0 for c in results)
    timing = Timing(model, time.perf_counter() - t0,
                    2 * len(greedy) + (len(repeat["second"]) if again else 0)
                    + sum(c.n for c in results),
                    greedy_s + (repeat["seconds"] if again else 0.0)
                    + sum(x.seconds for x in sampled),
                    greedy_s, sum(b.steps for b in greedy_stats))
    return ModelResult(model, greedy, sampled, results,
                       _measure(model, greedy, sampled, results), timing)


# -- the training-step probe ----------------------------------------------------------


def _probe_batches(tok, rows: list[dict], s: Settings, device):
    """``timed + warmup`` batches of ``batch`` rows: the first rows in file order with at
    least ``seq_len`` tokens, each cut to its last ``seq_len`` tokens (no padding)."""
    import torch

    need = s.batch * (s.warmup_steps + s.timed_steps)
    picked = []
    for r in rows:
        if r["n_tokens"] < s.seq_len:
            continue
        enc = tok(r["text"], return_offsets_mapping=True, add_special_tokens=False)
        mask = token_mask([tuple(o) for o in enc["offset_mapping"]],
                          [tuple(x) for x in r["train_spans"]])
        ids, m = last_window(enc["input_ids"], mask, s.seq_len)
        picked.append((r["id"], ids, m))
        if len(picked) == need:
            break
    if len(picked) < need:
        raise ValueError(f"only {len(picked)} SFT rows have {s.seq_len}+ tokens; need {need}")
    batches = []
    for k in range(0, need, s.batch):
        chunk = picked[k:k + s.batch]
        ids = torch.tensor([c[1] for c in chunk], dtype=torch.long)
        labels = ids.clone()
        labels[~torch.tensor([c[2] for c in chunk])] = -100
        # move only: int64 stays int64 (the MPS fused .to(device, dtype) bug)
        batches.append((ids.to(device), labels.to(device)))
    scored = sum(sum(c[2]) for c in picked)
    return batches, [c[0] for c in picked], scored


@dataclass
class StepTiming:
    size: str
    method: str  # "LoRA r16" or "full"
    trainable: int
    total: int
    seconds_per_step: float | None
    peak_bytes: int | None
    losses: list[float]
    note: str = ""


def _time_steps(size: str, method: str, s: Settings, device, batches) -> StepTiming:
    import torch

    from . import bfcl, progress
    from .device import peak_memory_bytes, reset_peak_memory, sync
    from .lora import ATTENTION_MLP, inject_lora, trainable_parameters

    model_id = f"Qwen/Qwen2.5-{size}-Instruct"
    progress.status(f"timing probe: {size} {method}, loading {model_id}")
    _, lm = bfcl.load_model(model_id, device)
    note = ""
    try:
        lm.train()
        total = sum(p.numel() for p in lm.parameters())
        if method == "full":
            params = list(lm.parameters())
        else:
            # the probe as it ran (its capture says "dtype (weights, grads, optimizer)
            # bfloat16"): adapters in the base dtype, not lora.py's fp32 default
            inject_lora(lm, ATTENTION_MLP, rank=s.lora_rank, alpha=s.lora_alpha,
                        dtype=next(lm.parameters()).dtype)
            params = trainable_parameters(lm)
        trainable = sum(p.numel() for p in params)
        torch.manual_seed(SEED)
        opt = torch.optim.AdamW(params, lr=s.lr)
        reset_peak_memory(device)
        losses, times = [], []
        on_step = progress.training(f"{size} {method}", len(batches))
        for k, (ids, labels) in enumerate(batches):
            sync(device)
            t0 = time.perf_counter()
            try:
                loss = lm(input_ids=ids, labels=labels).loss
                loss.backward()
            except RuntimeError as e:  # an op with no deterministic CUDA kernel
                if "deterministic" not in str(e) or note:
                    raise
                note = "warn_only: " + str(e).split("\n")[0][:120]
                torch.use_deterministic_algorithms(True, warn_only=True)
                opt.zero_grad(set_to_none=True)
                loss = lm(input_ids=ids, labels=labels).loss
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sync(device)
            if k >= s.warmup_steps:
                times.append(time.perf_counter() - t0)
            losses.append(float(loss.detach()))
            on_step(k, losses[-1])
        peak = peak_memory_bytes(device)
        return StepTiming(size, method, trainable, total, sum(times) / len(times), peak, losses,
                          note)
    except torch.OutOfMemoryError:
        return StepTiming(size, method, 0, 0, None, None, [], "out of memory")
    finally:
        if note and device.type == "cuda":
            torch.use_deterministic_algorithms(True)  # back to strict for what follows
        del lm
        _free(device)


# -- the report ---------------------------------------------------------------------------


class Stopped(Exception):
    """SIGTERM arrived while a model was being scored (or the probe ran)."""


class _Sigterm:
    """SIGTERM handler: always notes the request; raises :class:`Stopped` only inside
    ``raising()`` (scoring, the probe), so a report line is never cut in half."""

    def __init__(self) -> None:
        self.requested = False
        self.armed = False

    def __call__(self, signum, frame) -> None:
        self.requested = True
        if self.armed:
            self.armed = False
            raise Stopped()

    def raising(self):
        from contextlib import contextmanager

        @contextmanager
        def armed():
            if self.requested:
                raise Stopped()
            self.armed = True
            try:
                yield
            finally:
                self.armed = False
        return armed()


def _minutes(seconds: float) -> str:
    return f"{max(0.0, seconds) / 60:.0f} min"


def run(quick: bool, models: tuple[Model, ...] | None, root: Path,
        base_when_needed: bool = False, deadline: float | None = None,
        gen_batch: int | None = None) -> int:
    """Score, print, record. Returns the exit code: 0; 1 if the deadline stopped the
    scoring before the rule could decide; 143 after SIGTERM."""
    from . import bfcl, ours, progress, tasks
    from .device import dtype_name, get_device, preferred_dtype, set_determinism
    from .engine import ENGINE, default_batch_size, print_setup
    from .report import Report

    s = settings(quick, models, base_when_needed, gen_batch)
    out_dir = root / "outputs"
    device = get_device()
    batch = s.gen_batch or default_batch_size(device.type)
    determinism = set_determinism(device)  # before any model loads (cuBLAS workspace)
    dtype = dtype_name(preferred_dtype(device))
    splits = tasks.generate(tasks.DEFAULT_SEED)
    dev = splits["dev"]
    path = root / "data" / "tasks" / "dev.jsonl"
    if path.exists() and path.read_bytes() != tasks.jsonl_bytes(dev):
        sys.exit(f"{path} differs from what seed {tasks.DEFAULT_SEED} generates; rerun `tinytools tasks`")
    dev_tasks = ours.limit_per_kind(dev, s.ours_limit)
    versions = _versions()
    setup = {"dtype": dtype, "device": device.type, "engine": ENGINE, "batch_size": batch}
    record = {"command": "tinytools milestone0" + (" --quick" if quick else "")
              + (" --base-when-needed" if base_when_needed else "")
              + (" --deadline-epoch ..." if deadline is not None else ""),
              "deadline_epoch": deadline,
              "models": [m.hf_id for m in s.models], "device": device.type, "dtype": dtype,
              "determinism": determinism, "seeds": {"generation": SEED, "probe": SEED},
              "engine": ENGINE, "batch_size": batch,
              "samples": SAMPLES, "temperature": TEMPERATURE,
              "max_new_tokens": bfcl.MAX_NEW_TOKENS, "versions": versions,
              "dev_tasks_sha256": tasks.sha256(dev), "bfcl_dev_seed": bfcl.DEV_SEED,
              "probe": {k: v for k, v in asdict(s).items() if k not in ("models", "repeat_model")}}
    setup_path = out_dir / f"{s.tag}-setup.json"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text(json.dumps(record, indent=2) + "\n")

    rep = Report("tinytools milestone0" + (" --quick (not for publication)" if quick else ""),
                 out=out_dir / f"{s.tag}.txt")
    rep.header()
    rep.section("Setup")
    rep.kv("models", f"{len(s.models)}, in this order:")
    for m in s.models:
        rep.note(f"  {m.hf_id:<28} {role(m)}")
    if base_when_needed:
        rep.note("--base-when-needed: a base model is scored only if the rule")
        rep.note("reaches it; 0.5B base not at all.")
    rep.kv("our tasks", f"dev split, seed {tasks.DEFAULT_SEED}")
    rep.kv("  sha256 (first 16)", tasks.sha256(dev)[:16])
    rep.kv("  scored", f"{len(dev_tasks)} " + ("(all)" if s.ours_limit is None
                                               else f"(first {s.ours_limit} of each kind)"))
    rep.kv("BFCL", f"dev slice, {bfcl.DEV_SHARE}% per category, seed {bfcl.DEV_SEED}")
    rep.kv("  scored", "all of it" if s.bfcl_limit is None
           else f"first {s.bfcl_limit} per category")
    rep.kv("decoding", f"greedy; and {SAMPLES} samples at T={TEMPERATURE:g} (ours)")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("stop at", "the model's eos, and <|im_end|>")
    rep.kv("seed", f"{SEED} (+ item id and sample number)")
    print_setup(rep, batch, (1, SAMPLES))
    rep.kv("deadline", "none" if deadline is None else
           time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(deadline)))
    if deadline is not None:
        rep.note("A model whose scoring can't finish by then (a conservative estimate")
        rep.note("from what is measured) is not started.")
    rep.kv("dtype", dtype)
    rep.kv("determinism", determinism, wrap=True)
    for name, v in versions.items():
        rep.kv(name, v)

    results: dict[Model, ModelResult] = {}
    repeat: dict = {}
    probe: list[StepTiming] = []
    order = list(s.models)
    # the probe runs once the candidates (or, in --quick, the first model) are scored
    probe_after = next((m for m in reversed(order) if m in CANDIDATES), order[0])
    skipped: dict[Model, str] = {}
    stopped: str | None = None  # why scoring ended early
    term = _Sigterm()
    try:
        old_term = signal.signal(signal.SIGTERM, term)
    except ValueError:  # not the main thread
        old_term = None

    def save() -> None:
        # after each model, so a run stopped early still leaves the numbers
        record["measured"] = {k.hf_id: asdict(r.measured) | {"model": k.hf_id}
                              for k, r in results.items()}
        record["skipped"] = {k.hf_id: why for k, why in skipped.items()}
        record["probe_results"] = [asdict(t) for t in probe]
        record["timings"] = [asdict(r.timing) | {"model": k.hf_id}
                             for k, r in results.items() if r.timing]
        record["stopped"] = stopped
        setup_path.write_text(json.dumps(record, indent=2) + "\n")

    for idx, m in enumerate(order):
        current = m
        try:
            if term.requested:
                raise Stopped()
            measured_now = {x: r.measured for x, r in results.items()}
            why = skip_reason(m, measured_now, base_when_needed)
            if not why and deadline is not None:
                timings = [r.timing for r in results.values() if r.timing]
                cats = next(([c.n for c in r.bfcl] for r in results.values()), None)
                need = None if cats is None else estimate_seconds(
                    m, item_passes(m, len(dev_tasks), sum(cats), s.repeat_model), timings,
                    bfcl.MAX_NEW_TOKENS,
                    batch_count(len(dev_tasks), cats, batch, SAMPLES, m == s.repeat_model))
                if not fits(time.time(), deadline, need):
                    left = deadline - time.time()
                    needs = f"~{_minutes(need)}" if need is not None else "unknown"
                    stopped = (f"the deadline: {m.label} would need {needs} (estimate), "
                               f"{_minutes(left)} were left")
                    progress.status(f"stopping before {m.hf_id}: {stopped}")
                    for rest in order[idx:]:
                        skipped[rest] = (
                            f"not started: the deadline (needs {needs}, {_minutes(left)} "
                            "left)" if rest == m
                            else skip_reason(rest, measured_now, base_when_needed)
                            or "not started: the deadline")
                    save()
                    break
            if why:
                skipped[m] = why
                progress.status(f"{m.hf_id}: not scored ({why})")
            else:
                with term.raising():
                    results[m] = _score_model(m, s, device, dev_tasks, setup, out_dir, repeat,
                                              batch)
                _print_model(rep, results[m])
                if m == s.repeat_model:
                    _print_repeat(rep, repeat)
            if m == probe_after:
                current = None
                if fits(time.time(), deadline, PROBE_ALLOW_S):
                    with term.raising():
                        probe = _probe(rep, s, device, root)
                else:
                    rep.section("Training step timing: LoRA vs full fine-tuning")
                    rep.note(f"(skipped: less than {PROBE_ALLOW_S // 60} minutes were left before "
                             "the deadline)")
            save()
        except Stopped:
            stopped = "SIGTERM (the time limit, or the run was stopped)"
            progress.status("stopped by SIGTERM; printing what was measured")
            if current is None:
                rep.note("(the probe was stopped by SIGTERM)")
            for rest in order[idx:]:
                if rest not in results and rest not in skipped:
                    skipped[rest] = ("stopped by SIGTERM while being scored" if rest == current
                                     else "not started: SIGTERM")
            break
    try:  # what's left takes a second; the remote timeout's KILL comes 60 s after its TERM
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except ValueError:
        pass

    if skipped:
        rep.section("Not scored")
        for m, why in skipped.items():
            rep.kv(m.label, why if len(why) <= 37 else "")
            if len(why) > 37:
                rep.note("  " + why)
    _print_rule(rep, {m: r.measured for m, r in results.items()}, quick, stopped)
    rep.close()
    v = verdict({m: r.measured for m, r in results.items()})
    record["verdict"] = v.text
    save()
    progress.status(f"setup and results written to {setup_path.relative_to(root)}")
    if old_term is not None:
        try:
            signal.signal(signal.SIGTERM, old_term)
        except ValueError:
            pass
    if term.requested:
        return 128 + signal.SIGTERM
    return 1 if stopped and v.outcome == "incomplete" else 0


def _print_repeat(rep, repeat: dict) -> None:
    rep.section("The same greedy pass twice")
    m = repeat["model"]
    diff = differing(repeat["first"], repeat["second"])
    n = len(repeat["first"])
    rep.kv("model", m.hf_id)
    rep.kv("tasks", f"{n} of our dev tasks, greedy")
    rep.kv("replies byte-identical", f"{n - len(diff)} of {n}")
    rep.kv("outcomes that changed", repeat["outcomes_differ"])
    if diff:
        rep.note("replies that differ:")
        import textwrap

        for line in textwrap.wrap(", ".join(diff), 70):
            rep.print("    " + line)
    rep.note("Both passes ran in one process, one after the other; a second")
    rep.note("process (a rerun of the command) is the stronger test.")


def _probe(rep, s: Settings, device, root: Path) -> list[StepTiming]:
    from transformers import AutoTokenizer

    from .device import dtype_name, preferred_dtype

    rep.section("Training step timing: LoRA vs full fine-tuning")
    path = root / "data" / "sft" / "real.jsonl"
    if not path.exists():
        rep.note(f"(skipped: {path.relative_to(root)} is missing; run toolace-convert and")
        rep.note("sft-data first)")
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    tok = AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-{s.timing_sizes[0]}-Instruct")
    batches, ids, scored = _probe_batches(tok, rows, s, device)
    rep.kv("rows", f"{path.relative_to(root)} ({len(rows)} rows)")
    rep.kv("  used", f"{len(ids)}, the first with {s.seq_len}+ tokens")
    rep.kv("batch x sequence length", f"{s.batch} x {s.seq_len}, no padding")
    rep.kv("  scored tokens in those rows", scored)
    rep.note(f"Each row is cut to its last {s.seq_len} tokens (its assistant turn is")
    rep.note("at the end); masked tokens as in training (sft.py).")
    rep.kv("steps", f"{s.warmup_steps} warm-up + {s.timed_steps} timed")
    rep.note("A step: forward, backward, AdamW update; timed after a sync.")
    rep.kv("dtype (weights, grads, optimizer)", dtype_name(preferred_dtype(device)))
    rep.kv("LoRA", f"rank {s.lora_rank}, alpha {s.lora_alpha:g}, attention + MLP")
    rep.note("LoRA on q, k, v, o, gate, up and down in every layer.")
    rep.kv("gradient checkpointing", "off")
    rep.kv("model", "the instruct model of each size")
    timings = []
    for size in s.timing_sizes:
        for method in (f"LoRA r{s.lora_rank}", "full"):
            timings.append(_time_steps(size, "full" if method == "full" else method, s, device,
                                       batches))
    rep.blank()
    mem_label = {"cuda": "peak GB", "mps": "GB now*"}.get(device.type, "GB")
    rows_out = []
    for t in timings:
        rows_out.append([t.size, t.method,
                         f"{t.trainable / 1e6:.1f}M" if t.trainable else "-",
                         f"{t.seconds_per_step:.3f}" if t.seconds_per_step else "-",
                         f"{t.peak_bytes / 1e9:.1f}" if t.peak_bytes else "-",
                         f"{t.losses[0]:.3f}" if t.losses else "-"])
    rep.table(["size", "method", "trainable", "s/step", mem_label, "loss, step 1"], rows_out)
    if device.type == "mps":
        rep.note("* MPS has no peak counter: memory allocated after the last step,")
        rep.note("  a lower bound on the peak.")
    for t in timings:
        if t.note:
            rep.note(f"{t.size} {t.method}: {t.note}")
    rows_ft = len(rows)  # the real arm's training rows (the control arm has as many)
    rep.blank()
    rep.note(f"One epoch of {rows_ft} rows at this rate, if every batch had this")
    rep.note(f"shape ({epoch_steps(rows_ft, s.batch)} steps of {s.batch}); arithmetic, not a measurement:")
    rep.table(["size", "method", "hours per epoch"],
              [[t.size, t.method, f"{epoch_hours(t.seconds_per_step, rows_ft, s.batch):.2f}"]
               for t in timings if t.seconds_per_step])
    return timings


def criteria(m: Measured) -> list[tuple[str, str, int, int, str, bool]]:
    """The rule's four criteria for one model: (name, what, k, n, threshold, met)."""
    return [
        ("a", "our dev tasks, greedy, right by execution", m.ours_right, m.ours_n,
         f"<= {OURS_RIGHT_MAX}%", m.a),
        ("b", "BFCL dev slice, call categories, greedy right", m.bfcl_right, m.bfcl_n,
         f"<= {BFCL_RIGHT_MAX}%", m.b),
        ("c", f"call-needing dev tasks, mixed groups of {SAMPLES} at T={TEMPERATURE:g}",
         m.mixed, m.groups, f">= {MIXED_MIN}%", m.c),
        ("d", "call-needing dev tasks, greedy, right by execution", m.calls_right, m.calls_n,
         f">= {CALLS_RIGHT_MIN}%", m.d),
    ]


def _print_rule(rep, measured: dict[Model, Measured], quick: bool,
                stopped: str | None = None) -> None:
    from .ours import pct

    rep.section("The rule (fixed before these numbers)")
    rep.note(f"(a) our dev tasks, greedy, right by execution: at most {OURS_RIGHT_MAX}%")
    rep.note(f"(b) BFCL dev slice, call categories, right: at most {BFCL_RIGHT_MAX}%")
    rep.note(f"(c) dev tasks that need a call, {SAMPLES} samples at T={TEMPERATURE:g}, mixed")
    rep.note(f"    groups: at least {MIXED_MIN}%")
    rep.note("(d) dev tasks that need a call, greedy, right by execution: at")
    rep.note(f"    least {CALLS_RIGHT_MIN}%")
    rep.note("Tasks that need a call: choose+fill and tricky arguments (not")
    rep.note("'no tool fits', where declining is right).")
    rep.note("Pick the smaller of 1.5B / 3B instruct meeting all four; else")
    rep.note("base 1.5B, then base 3B; else stop and bring it to the author.")
    order = [m for m in (*RULE_ORDER, I05, B05) if m in measured]
    for k in range(4):
        rows = []
        for m in order:
            name, what, n_right, n, threshold, met = criteria(measured[m])[k]
            rows.append([m.label, role(m), pct(n_right, n), threshold,
                         "met" if met else "not met"])
        rep.blank()
        rep.note(f"({name}) {what}")
        rep.table(["model", "role", "count", "needs", "met"], rows)
    rep.blank()
    rep.table(["model", "role", "(a)", "(b)", "(c)", "(d)", "qualifies"],
              [[m.label, role(m), *["met" if c[5] else "not" for c in criteria(measured[m])],
                "yes" if measured[m].qualifies else "no"] for m in order])
    v = verdict(measured)
    rep.blank()
    import textwrap

    if stopped:
        for line in textwrap.wrap(f"Scoring stopped early, by {stopped}; the rule reads "
                                  "only what was measured.", 70):
            rep.note(line)
        rep.blank()
    rep.kv("verdict", v.outcome)
    for line in textwrap.wrap(v.text, 68):
        rep.note("  " + line)
    if quick:
        rep.note("--quick: 0.5B only, a few items each; no verdict is meant.")
    rep.takeaway(takeaway(measured, v))
