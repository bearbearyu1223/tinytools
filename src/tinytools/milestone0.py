"""Milestone 0: which model post 9 trains, decided by a rule fixed before the numbers.

Six models (Qwen2.5 0.5B / 1.5B / 3B, base and instruct) are scored on the dev
split only: our 200 dev tasks greedy, the same tasks with 8 samples at T=1, and
the BFCL dev slice greedy. The rule (plan 3a, thresholds adopted by the author):

    a candidate qualifies when, all three,
    (a) greedy right by execution match on our dev tasks <= 70%
    (b) BFCL right on the dev slice's call categories     <= 80%
    (c) mixed groups (some of 8 right, some wrong)        >= 10% of dev tasks

Pick the smaller of 1.5B / 3B instruct that qualifies; else base 1.5B, then
base 3B, by the same rule; else stop and bring it to the author. 0.5B is scored
as comparison rows only. Also: one model's greedy dev pass run twice (are the
replies byte-identical?), and a training-step timing probe (LoRA vs full).

Everything above the ``-- running`` line is model-free and unit-tested in
``tests/test_milestone0.py``.
"""

from __future__ import annotations

import gc
import json
import math
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
    """The three numbers the rule reads, with their denominators."""

    model: Model
    ours_right: int  # greedy, right by execution match, all dev tasks
    ours_n: int
    bfcl_right: int  # greedy, call categories of the dev slice
    bfcl_n: int
    mixed: int  # dev tasks with some of 8 right and some wrong
    groups: int

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
    def qualifies(self) -> bool:
        return self.a and self.b and self.c


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
    return [name for name, ok in (("a", m.a), ("b", m.b), ("c", m.c)) if not ok]


def takeaway(measured: dict[Model, Measured], v: Verdict) -> str:
    """The sentence under the verdict, built from the numbers."""
    from .ours import share

    def nums(m: Measured) -> str:
        return (f"{share(m.ours_right, m.ours_n)} right on our dev tasks, "
                f"{share(m.bfcl_right, m.bfcl_n)} on BFCL's dev calls, "
                f"{share(m.mixed, m.groups)} mixed groups")

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

    @property
    def tag(self) -> str:
        return "milestone0-quick" if self.quick else "milestone0"


def settings(quick: bool, models: tuple[Model, ...] | None = None,
             base_when_needed: bool = False) -> Settings:
    if quick:  # one item per kind and category: a base model's replies run to the cap
        return Settings(True, models or (I05, B05), ours_limit=1, bfcl_limit=1,
                        repeat_model=I05, timing_sizes=("0.5B",), batch=2, seq_len=256,
                        warmup_steps=1, timed_steps=2, base_when_needed=base_when_needed)
    return Settings(False, models or RUN_ORDER, ours_limit=None, bfcl_limit=None,
                    repeat_model=REPEAT_MODEL, timing_sizes=TIMING_SIZES, batch=4,
                    seq_len=1024, warmup_steps=1, timed_steps=3,
                    base_when_needed=base_when_needed)


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


def _measure(model: Model, greedy, sampled, bfcl_results) -> Measured:
    from .classify import mixed_share

    calls = [r for r in bfcl_results if r.category in BFCL_CALL_CATEGORIES]
    _, mixed, _ = mixed_share([s.right for s in sampled], SAMPLES)
    return Measured(model, ours_right=sum(s.right for s in greedy), ours_n=len(greedy),
                    bfcl_right=sum(sum(r.right_per_task) for r in calls),
                    bfcl_n=sum(len(r.right_per_task) for r in calls),
                    mixed=mixed, groups=len(sampled))


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
                 repeat: dict) -> ModelResult:
    from . import bfcl, ours, progress

    progress.status(f"loading {model.hf_id} on {device}")
    tok, lm = bfcl.load_model(model.hf_id, device)
    try:
        greedy = ours.score(tok, lm, dev_tasks, samples=1, temperature=0.0, seed=SEED,
                            on_item=progress.items(f"{model.label}: our dev, greedy",
                                                   len(dev_tasks)))
        if model == s.repeat_model:
            again = ours.score(tok, lm, dev_tasks, samples=1, temperature=0.0, seed=SEED,
                               on_item=progress.items(f"{model.label}: our dev, greedy again",
                                                      len(dev_tasks)))
            repeat["model"] = model
            repeat["first"] = [(x.task["id"], x.replies[0]) for x in greedy]
            repeat["second"] = [(x.task["id"], x.replies[0]) for x in again]
            repeat["outcomes_differ"] = sum(a.checks[0].outcome != b.checks[0].outcome
                                            for a, b in zip(greedy, again, strict=True))
        sampled = ours.score(tok, lm, dev_tasks, samples=SAMPLES, temperature=TEMPERATURE,
                             seed=SEED, on_item=progress.items(
                                 f"{model.label}: our dev, {SAMPLES} samples", len(dev_tasks)))
        records: list[dict] = []
        results = []
        for cat in BFCL_CATEGORIES:
            items = bfcl.load_items(cat, s.bfcl_limit, "dev")
            results.append(bfcl.score(tok, lm, items, samples=1, temperature=0.0, seed=SEED,
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
    return ModelResult(model, greedy, sampled, results, _measure(model, greedy, sampled, results))


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
            inject_lora(lm, ATTENTION_MLP, rank=s.lora_rank, alpha=s.lora_alpha)
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


def run(quick: bool, models: tuple[Model, ...] | None, root: Path,
        base_when_needed: bool = False) -> None:
    from . import bfcl, ours, progress, tasks
    from .device import dtype_name, get_device, preferred_dtype, set_determinism
    from .report import Report

    s = settings(quick, models, base_when_needed)
    out_dir = root / "outputs"
    device = get_device()
    determinism = set_determinism(device)  # before any model loads (cuBLAS workspace)
    dtype = dtype_name(preferred_dtype(device))
    splits = tasks.generate(tasks.DEFAULT_SEED)
    dev = splits["dev"]
    path = root / "data" / "tasks" / "dev.jsonl"
    if path.exists() and path.read_bytes() != tasks.jsonl_bytes(dev):
        sys.exit(f"{path} differs from what seed {tasks.DEFAULT_SEED} generates; rerun `tinytools tasks`")
    dev_tasks = ours.limit_per_kind(dev, s.ours_limit)
    versions = _versions()
    setup = {"dtype": dtype, "device": device.type}
    record = {"command": "tinytools milestone0" + (" --quick" if quick else "")
              + (" --base-when-needed" if base_when_needed else ""),
              "models": [m.hf_id for m in s.models], "device": device.type, "dtype": dtype,
              "determinism": determinism, "seeds": {"generation": SEED, "probe": SEED},
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
    rep.kv("seed", f"{SEED} (+ crc32 of each item id)")
    rep.kv("dtype", dtype)
    rep.kv("determinism", determinism)
    for name, v in versions.items():
        rep.kv(name, v)

    results: dict[Model, ModelResult] = {}
    repeat: dict = {}
    probe: list[StepTiming] = []
    order = list(s.models)
    # the probe runs once the candidates (or, in --quick, the first model) are scored
    probe_after = next((m for m in reversed(order) if m in CANDIDATES), order[0])
    skipped: dict[Model, str] = {}
    for m in order:
        why = skip_reason(m, {k: r.measured for k, r in results.items()}, base_when_needed)
        if why:
            skipped[m] = why
            progress.status(f"{m.hf_id}: not scored ({why})")
        else:
            results[m] = _score_model(m, s, device, dev_tasks, setup, out_dir, repeat)
            _print_model(rep, results[m])
            if m == s.repeat_model:
                _print_repeat(rep, repeat)
        if m == probe_after:
            probe = _probe(rep, s, device, root)
        # after each model, so a run stopped by its time limit still leaves the numbers
        record["measured"] = {k.hf_id: asdict(r.measured) | {"model": k.hf_id}
                              for k, r in results.items()}
        record["skipped"] = {k.hf_id: why for k, why in skipped.items()}
        record["probe_results"] = [asdict(t) for t in probe]
        setup_path.write_text(json.dumps(record, indent=2) + "\n")

    if skipped:
        rep.section("Not scored")
        for m, why in skipped.items():
            rep.kv(m.label, why if len(why) <= 37 else "")
            if len(why) > 37:
                rep.note("  " + why)
    _print_rule(rep, {m: r.measured for m, r in results.items()}, quick)
    rep.close()
    record["verdict"] = verdict({m: r.measured for m, r in results.items()}).text
    setup_path.write_text(json.dumps(record, indent=2) + "\n")
    progress.status(f"setup and results written to {setup_path.relative_to(root)}")


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


def _print_rule(rep, measured: dict[Model, Measured], quick: bool) -> None:
    from .ours import pct

    rep.section("The rule (fixed before these numbers)")
    rep.note(f"(a) our dev tasks, greedy, right by execution: at most {OURS_RIGHT_MAX}%")
    rep.note(f"(b) BFCL dev slice, call categories, right: at most {BFCL_RIGHT_MAX}%")
    rep.note(f"(c) our dev tasks, {SAMPLES} samples at T={TEMPERATURE:g}, mixed groups: "
             f"at least {MIXED_MIN}%")
    rep.note("Pick the smaller of 1.5B / 3B instruct meeting all three; else")
    rep.note("base 1.5B, then base 3B; else stop and bring it to the author.")
    order = [m for m in (*RULE_ORDER, I05, B05) if m in measured]
    for title, fn in (("(a) our dev tasks, greedy right",
                       lambda x: (pct(x.ours_right, x.ours_n), x.a)),
                      ("(b) BFCL dev calls, greedy right",
                       lambda x: (pct(x.bfcl_right, x.bfcl_n), x.b)),
                      ("(c) mixed groups of 8",
                       lambda x: (pct(x.mixed, x.groups), x.c))):
        rep.blank()
        rep.table(["model", "role", title, "met"],
                  [[m.label, role(m), *fn(measured[m])] for m in order])
    rep.blank()
    rep.table(["model", "role", "(a)", "(b)", "(c)", "qualifies"],
              [[m.label, role(m), measured[m].a, measured[m].b, measured[m].c,
                measured[m].qualifies] for m in order])
    v = verdict(measured)
    rep.blank()
    import textwrap

    rep.kv("verdict", v.outcome)
    for line in textwrap.wrap(v.text, 68):
        rep.note("  " + line)
    if quick:
        rep.note("--quick: 0.5B only, a few items each; no verdict is meant.")
    rep.takeaway(takeaway(measured, v))
