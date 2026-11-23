"""``tinytools engine-check``: the batched engine against the one-at-a-time engine it replaced.

On one model (0.5B instruct; the Mac, fp32), the same items through both engines:

* greedy on our test tasks and on BFCL eval items: how many replies are byte-identical,
  and for each that differs, where, and whether its outcome changed;
* the control for those: the batched engine at batch size 1 (no padding, no other
  rows) against the old engine, so a difference above can be laid on batching;
* 8 samples at T=1: outcome counts and groups side by side (the two engines draw
  different random numbers, so equal replies are not expected), and the batched
  engine at a second batch size, which uses the same draws per reply: how many
  replies survive a change of batch;
* the batched engine run twice: byte-identical or not;
* wall clock per pass, and the decode steps the batched passes took (the A100
  estimate's input).

The comparison arithmetic (:func:`compare`, :func:`divergence`) is pure and tested in
``tests/test_engine_check.py``.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
SAMPLES = 8
TEMPERATURE = 1.0


@dataclass(frozen=True)
class Sizes:
    ours_per_kind: int  # greedy: first N test tasks of each kind
    bfcl_per_category: int  # greedy: first N eval items of each category
    sampled_tasks: int  # sampled: the first N test tasks by id
    alt_batch: int  # the second batch size for the sampled pass


FULL = Sizes(20, 8, 10, 16)  # 20 sampled tasks: ~19 min on the Mac, over the 15-min limit
QUICK = Sizes(2, 1, 3, 16)


# -- pure ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Compared:
    n: int
    identical: int
    differ: list[str]  # ids, in order
    outcome_changed: list[str]


def compare(ids: list[str], a: list[str], b: list[str], out_a: list[str],
            out_b: list[str]) -> Compared:
    """Replies of two passes over the same ids (one reply per id): identical, differing,
    and differing with a different outcome."""
    if not (len(ids) == len(a) == len(b) == len(out_a) == len(out_b)):
        raise ValueError("the passes must cover the same items")
    differ = [i for i, x, y in zip(ids, a, b, strict=True) if x != y]
    changed = [i for i, x, y, oa, ob in zip(ids, a, b, out_a, out_b, strict=True)
               if x != y and oa != ob]
    return Compared(len(ids), len(ids) - len(differ), differ, changed)


def divergence(a: str, b: str) -> int:
    """Index of the first character where two different strings differ (or the
    shorter one's length, when one is a prefix of the other)."""
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def same_count(a: list[list[str]], b: list[list[str]]) -> int:
    """Replies equal position by position, over groups of replies."""
    return sum(x == y for ga, gb in zip(a, b, strict=True) for x, y in zip(ga, gb, strict=True))


def speedup(old_s: float, new_s: float) -> str:
    return f"{old_s / new_s:.1f}x" if new_s > 0 else "-"


# -- running ------------------------------------------------------------------------------


@dataclass
class Pass:
    """One engine's pass: replies per item (lists of samples), outcomes, seconds, and
    for the batched engine its batches."""

    replies: list[list[str]]
    outcomes: list[list[str]]
    seconds: float
    new_tokens: int
    batches: list | None = None


def _old(tok, model, keys, prompts, judge, samples, temperature, on_item) -> Pass:
    from . import bfcl

    cfg = bfcl.generation_config(tok, model, samples, temperature)
    replies, outcomes, secs, toks = [], [], 0.0, 0
    for k, (key, prompt) in enumerate(zip(keys, prompts, strict=True), start=1):
        g = bfcl.generate_text(tok, model, prompt, key, cfg, SEED)
        replies.append(g.texts)
        outcomes.append([judge(k - 1, t) for t in g.texts])
        secs += g.seconds
        toks += sum(g.new_tokens)
        on_item(k)
    return Pass(replies, outcomes, secs, toks)


def _new(tok, model, keys, prompts, judge, samples, temperature, batch, on_item) -> Pass:
    from . import bfcl, engine

    run = engine.generate(tok, model, keys, prompts, samples=samples, temperature=temperature,
                          seed=SEED, batch_size=batch, max_new_tokens=bfcl.MAX_NEW_TOKENS,
                          on_done=on_item)
    replies = [g.texts for g in run.items]
    outcomes = [[judge(k, t) for t in g.texts] for k, g in enumerate(run.items)]
    return Pass(replies, outcomes, sum(b.seconds for b in run.batches),
                sum(sum(g.new_tokens) for g in run.items), run.batches)


def sizes_for(quick: bool, ours: int | None, bfcl: int | None, sampled: int | None) -> Sizes:
    """The item counts: the full check's or --quick's, each overridable."""
    base = QUICK if quick else FULL
    return Sizes(ours or base.ours_per_kind, bfcl or base.bfcl_per_category,
                 sampled or base.sampled_tasks, base.alt_batch)


def tag_for(quick: bool, sizes: Sizes) -> str:
    """The capture's name: engine-check(-quick), or with the counts when they differ."""
    stem = "engine-check-quick" if quick else "engine-check"
    if sizes == (QUICK if quick else FULL):
        return stem
    return f"{stem}-{sizes.ours_per_kind}-{sizes.bfcl_per_category}-{sizes.sampled_tasks}"


def run(quick: bool, batch: int | None, root: Path, sizes: Sizes | None = None) -> None:
    from . import bfcl, ours, progress, tasks
    from .checkers import check_reply
    from .device import dtype_name, get_device, preferred_dtype, set_determinism
    from .engine import default_batch_size, print_setup
    from .report import Report

    sizes = sizes or (QUICK if quick else FULL)
    device = get_device()
    determinism = set_determinism(device)
    dtype = dtype_name(preferred_dtype(device))
    batch = batch or default_batch_size(device.type)
    tag = tag_for(quick, sizes)
    rep = Report("tinytools engine-check" + (" --quick" if quick else "")
                 + ("" if tag in ("engine-check", "engine-check-quick")
                    else " (not the full check's counts)"),
                 out=root / "outputs" / f"{tag}.txt")

    test = tasks.generate(tasks.DEFAULT_SEED)["test"]
    greedy_tasks = ours.limit_per_kind(test, sizes.ours_per_kind)
    sampled_tasks = sorted(test, key=lambda t: t["id"])[:sizes.sampled_tasks]
    items = [it for cat in bfcl.CATEGORIES
             for it in bfcl.load_items(cat, sizes.bfcl_per_category, "eval")]

    rep.header()
    rep.section("Setup")
    rep.kv("model", MODEL)
    rep.kv("dtype", dtype)
    rep.kv("determinism", determinism, wrap=True)
    rep.kv("greedy, our tasks", f"{len(greedy_tasks)}: first {sizes.ours_per_kind} test "
           "tasks per kind")
    rep.kv("greedy, BFCL", f"{len(items)}: first {sizes.bfcl_per_category} eval items "
           "per category")
    rep.kv("sampled, our tasks", f"{len(sampled_tasks)}: the first test tasks by id")
    rep.kv("  decoding", f"{SAMPLES} samples, temperature {TEMPERATURE:g}")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("seed", SEED)
    rep.note("One at a time (the replaced engine): Hugging Face generate on one")
    rep.note("prompt, torch.manual_seed(seed + crc32(item id)) before it; its")
    rep.note("8 samples are one generate call with 8 return sequences.")
    rep.note("Batched (the engine every capture now uses):")
    print_setup(rep, batch, (1, SAMPLES))
    rep.note("Its samples: each reply draws from its own generator seeded with")
    rep.note("the seed, item id and sample number (engine.py says how).")

    progress.status(f"loading {MODEL} on {device}")
    tok, model = bfcl.load_model(MODEL, device)
    h = bfcl.handler()

    def ours_judge(task_list):
        return lambda k, text: check_reply(text, task_list[k]["reference"]).outcome

    def bfcl_judge(k, text):
        return bfcl.judge(h, items[k], text).outcome

    sets = {
        "ours": ([t["id"] for t in greedy_tasks], [ours.prompt(tok, t) for t in greedy_tasks],
                 ours_judge(greedy_tasks)),
        "bfcl": ([it.id for it in items], [bfcl.prompt_text(tok, it) for it in items],
                 bfcl_judge),
        "sampled": ([t["id"] for t in sampled_tasks],
                    [ours.prompt(tok, t) for t in sampled_tasks], ours_judge(sampled_tasks)),
    }

    def go(name, label, engine_batch):
        keys, prompts, judge = sets[name]
        samples, temp = (SAMPLES, TEMPERATURE) if name == "sampled" else (1, 0.0)
        on = progress.items(f"{name}, {label}", len(keys))
        if engine_batch is None:
            return _old(tok, model, keys, prompts, judge, samples, temp, on)
        return _new(tok, model, keys, prompts, judge, samples, temp, engine_batch, on)

    t0 = time.perf_counter()
    old = {n: go(n, "one at a time", None) for n in sets}
    new = {n: go(n, f"batched {batch}", batch) for n in sets}
    again = {n: go(n, f"batched {batch}, again", batch) for n in sets}
    one = {n: go(n, "batched 1", 1) for n in ("ours", "bfcl")}
    alt = go("sampled", f"batched {sizes.alt_batch}", sizes.alt_batch)
    progress.status(f"all passes in {time.perf_counter() - t0:.0f} s")

    # -- greedy --------------------------------------------------------------------------
    rep.section("Greedy: one at a time vs batched")
    rows = []
    found = {}
    for name, label in (("ours", "our tasks"), ("bfcl", "BFCL")):
        keys = sets[name][0]
        for engine_label, p in ((f"batched, {batch}", new[name]), ("batched, 1", one[name])):
            c = compare(keys, [r[0] for r in old[name].replies], [r[0] for r in p.replies],
                        [o[0] for o in old[name].outcomes], [o[0] for o in p.outcomes])
            rows.append([label, engine_label, c.n, c.identical, len(c.differ),
                         len(c.outcome_changed)])
            if engine_label != "batched, 1":
                found[name] = c
    rep.table(["items", "vs one at a time", "n", "identical", "differ", "outcome changed"],
              rows)
    rep.note("identical: the same reply, byte for byte. batched, 1: the batched")
    rep.note("engine one item per batch (no padding): the control, which")
    rep.note("separates the new loop from the effect of batching.")
    for name in ("ours", "bfcl"):
        keys = sets[name][0]
        for i in found[name].differ:
            k = keys.index(i)
            a, b = old[name].replies[k][0], new[name].replies[k][0]
            at = divergence(a, b)
            rep.blank()
            rep.note(f"{i}: {old[name].outcomes[k][0]} -> {new[name].outcomes[k][0]}"
                     f" (first difference at character {at})", wrap=True)
            for who, text in (("one at a time", a), ("batched", b)):
                rep.note(f"  {who:<13}: ..." + _clip(text[max(0, at - 30):at + 40]))

    # -- sampled -------------------------------------------------------------------------
    from .classify import OUTCOMES, is_right

    def counts(p: Pass) -> Counter:
        return Counter(o for os_ in p.outcomes for o in os_)

    s_old, s_new = old["sampled"], new["sampled"]
    n_replies = sum(len(r) for r in s_new.replies)
    rep.section(f"Sampled ({SAMPLES} at T={TEMPERATURE:g}): one at a time vs batched")
    c_old, c_new, c_alt = counts(s_old), counts(s_new), counts(alt)
    shown = [o for o in OUTCOMES if c_old[o] or c_new[o] or c_alt[o]]
    rep.table(["outcome", "one at a time", f"batched, {batch}", f"batched, {sizes.alt_batch}"],
              [[o, c_old[o], c_new[o], c_alt[o]] for o in shown]
              + [["total", sum(c_old.values()), sum(c_new.values()), sum(c_alt.values())]])
    rep.blank()

    def right_per_task(p: Pass) -> list[int]:
        return [sum(is_right(o) for o in os_) for os_ in p.outcomes]

    hist = [right_per_task(p) for p in (s_old, s_new, alt)]
    rep.table(["right of 8", "one at a time", f"batched, {batch}", f"batched, {sizes.alt_batch}"],
              [[str(k), *[h.count(k) for h in hist]] for k in range(SAMPLES + 1)]
              + [["mixed", *[sum(0 < x < SAMPLES for x in h) for h in hist]]])
    rep.note("tasks with k of 8 replies right; mixed: some right, some wrong.")
    rep.blank()
    rep.note(f"Replies identical to batched, {batch}:")
    rep.kv(f"  batched at batch size {sizes.alt_batch}",
           f"{same_count(s_new.replies, alt.replies)} of {n_replies}")
    rep.kv("  one at a time", f"{same_count(s_new.replies, s_old.replies)} of {n_replies}")
    rep.note("Across batch sizes each reply has the same random draws, so a")
    rep.note("difference there comes from rounding in the forward pass. The")
    rep.note("old engine drew from torch's generator, so its replies are a")
    rep.note("different sample: compare its counts, not its replies.")

    # -- twice -----------------------------------------------------------------------------
    rep.section("The batched engine twice (one process, same batch size)")
    rep.table(["pass", "replies", "identical"],
              [[label, sum(len(r) for r in new[n].replies),
                same_count(new[n].replies, again[n].replies)]
               for n, label in (("ours", "greedy, our tasks"), ("bfcl", "greedy, BFCL"),
                                ("sampled", f"{SAMPLES} samples, our tasks"))])

    # -- speed -----------------------------------------------------------------------------
    rep.section("Wall clock (varies run to run)")
    rows = []
    for n, label in (("ours", "greedy, ours"), ("bfcl", "greedy, BFCL"),
                     ("sampled", f"x{SAMPLES}, ours")):
        rows.append([label, f"{old[n].seconds:.0f}", f"{new[n].seconds:.0f}",
                     f"{again[n].seconds:.0f}", speedup(old[n].seconds, new[n].seconds)])
    tot_old = sum(p.seconds for p in old.values())
    tot_new = sum(p.seconds for p in new.values())
    rows.append(["total", f"{tot_old:.0f}", f"{tot_new:.0f}",
                 f"{sum(p.seconds for p in again.values()):.0f}", speedup(tot_old, tot_new)])
    rep.table(["pass", "one at a time s", f"batched {batch} s", "again s", "speed-up"], rows)
    rep.note("Generation only (no loading or checking); speed-up: one at a")
    rep.note("time / batched (first run).")
    rep.blank()
    rep.table(["batched pass", "batches", "decode steps", "s/step", "new tokens", "row use"],
              [_steps_row(label, new[n]) for n, label in
               (("ours", "greedy, ours"), ("bfcl", "greedy, BFCL"),
                ("sampled", f"x{SAMPLES}, ours"))])
    rep.note("decode steps: forward passes after each batch's prefill, summed;")
    rep.note("s/step includes the prefill. row use: new tokens / (rows x steps),")
    rep.note("the share of row-steps that produced a counted token.")
    rep.close()


def _clip(text: str) -> str:
    """One line of a reply for the report: newlines shown as \\n, cut to fit 74 columns."""
    flat = text.replace("\n", "\\n")
    return flat[:52]


def _steps_row(label: str, p: Pass) -> list:
    steps = sum(b.steps for b in p.batches)
    row_steps = sum(b.steps * b.rows for b in p.batches)
    return [label, len(p.batches), steps, f"{p.seconds / max(1, steps):.3f}", p.new_tokens,
            f"{100 * p.new_tokens / max(1, row_steps):.0f}%"]
