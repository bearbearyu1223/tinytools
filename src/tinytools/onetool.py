"""The one-tool rerun's report (plan 3g): ``sft-report --profile onetool``.

The onetool profile trains the Instruct arm's real set without its 514 "no call, 1 tool"
rows (``data/sft/real-onetool.jsonl``), seeds 1-3, settings exactly the main run's, and
scores each adapter as the main run scored its own. This report reads those runs
(``outputs/onetool/real-seed<N>/``) beside the main run's untrained and real arm (read
only, ``--main-from``) and prints, paired by task over seed means:

* the primary comparison, fixed before the run: right on BFCL's one-function call items
  (simple + parallel), onetool real - main real, bootstrap 95% interval; "confirmed" if
  the interval excludes 0 upward. The verdict is printed only when both sides have the
  rule's seeds (1-3) and every run trained on its pinned set (:func:`check_inputs`:
  otherwise the report stops before printing anything);
* beside it (descriptive): the no-call share there; right and no call on the
  multi-function items (multiple + parallel_multiple); right on our tasks that need a
  call; declining on our no-tool-fits tasks and BFCL irrelevance; BFCL's outcomes per
  category for the untrained model, the main run's real arm and onetool's;
* training: the settings (r4c) with the row count and each seed's losses beside the main
  run's real arm.

The strict reading only (a call counts inside ``<tool_call>`` tags): the report counts
the replies the tag-free reading would change. Model-free; the arithmetic is
:mod:`tinytools.effects`'. The pure parts are tested in ``tests/test_onetool.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import effects as E
from . import sftrun as R
from . import train as T

ONE_FUNCTION = ("simple", "parallel")  # BFCL categories whose items offer one function
MULTI_FUNCTION = ("multiple", "parallel_multiple")
RULE_SEEDS = [1, 2, 3]  # the rule's seeds, on both sides
MAIN_FROM = "runs/2026-11-23-main/outputs/main"  # the main run's arm tree (in git)
QUICK_MAIN_FROM = "outputs/main-quick"
# score.json fields two scorings must share to be compared
SCORED_AS = ("model", "engine", "batch_size", "dtype", "device", "seed", "samples",
             "temperature", "max_new_tokens", "limits")


# -- pure ---------------------------------------------------------------------------------


def arm_seeds(out: Path, arm: str = "real") -> tuple[list[int], list[str]]:
    """Seeds whose ``<arm>-seed<N>/`` is trained and scored (train.json and score.json),
    and the labels of runs there that are not (the repeat is not a seed's run)."""
    seeds, partial = [], []
    for d in sorted(out.iterdir()) if out.exists() else []:
        if not d.name.startswith(f"{arm}-seed") or d.name.endswith("-repeat"):
            continue
        if (d / "train.json").exists() and (d / "score.json").exists():
            seeds.append(int(d.name.split("seed")[1]))
        else:
            partial.append(d.name)
    return sorted(seeds), partial


def excludes_zero_upward(p: E.Paired) -> bool:
    """The rule's reading: the interval's lower end above 0."""
    return p.lo > 0


def verdict(p: E.Paired) -> str:
    return "confirmed" if excludes_zero_upward(p) else "not confirmed"


def compare(onetool: list[list[float]], main: list[list[float]]) -> E.Paired:
    """onetool real - main real on the same tasks: per task, each arm's seed mean of its
    0/1 values (``x[s][t]``: seed s, task t); a paired bootstrap over tasks."""
    return E.paired("onetool real", E.seed_mean(onetool), "main real", E.seed_mean(main),
                    len(onetool), len(main))


def primary_line(p: E.Paired, seeds: list[int] | None = None,
                 main_seeds: list[int] | None = None) -> str:
    """The primary line. The verdict needs the rule's seeds on both sides
    (:data:`RULE_SEEDS`); with fewer it is withheld and the seeds are said."""
    head = (f"primary: right on BFCL's one-function call items (simple + "
            f"parallel, {p.n}), onetool real - main real: {E.points(p.diff)} points "
            f"{E.interval(p.lo, p.hi)}, ")
    if seeds is None or (seeds == RULE_SEEDS and main_seeds == RULE_SEEDS):
        return (head + f"excludes 0 upward: {'yes' if excludes_zero_upward(p) else 'no'} "
                f"-> {verdict(p)}")
    return (head + "verdict withheld: the rule needs seeds 1-3 on both sides, and "
            f"onetool has {_seed_list(seeds)}, main real {_seed_list(main_seeds or [])}")


def _seed_list(seeds: list[int]) -> str:
    return ", ".join(map(str, seeds)) or "none"


def check_inputs(one_shas: dict[str, str], main_shas: dict[str, str], one_pin: str,
                 main_pin: str) -> list[str]:
    """Why the report must not print a verdict: each run (label -> the sha256 its
    train.json records) not trained on its side's pinned set. Empty: all as pinned. Pure."""
    bad = [f"onetool {k}: trained on {(v or '?')[:16]}, not the variant's {one_pin[:16]}"
           for k, v in one_shas.items() if v != one_pin]
    bad += [f"main {k}: trained on {(v or '?')[:16]}, not real.jsonl's {main_pin[:16]}"
            for k, v in main_shas.items() if v != main_pin]
    return bad


def mean_count(per_seed: list[list[str]], outcome: str) -> str:
    """How many tasks had ``outcome``, averaged over seeds, one decimal (an int for one
    run)."""
    counts = [sum(o == outcome for o in run) for run in per_seed]
    if len(counts) == 1:
        return str(counts[0])
    return f"{sum(counts) / len(counts):.1f}"


def differing(a: dict, b: dict, keys) -> list[str]:
    """The ``keys`` whose values differ between two records. Pure."""
    return [k for k in keys if a.get(k) != b.get(k)]


# -- the report -----------------------------------------------------------------------------


def run_report(root: Path, quick: bool, reading: str = "strict", source: Path | None = None,
               main_from: Path | None = None, profile: str = "onetool") -> int:
    from dataclasses import replace

    from . import bfcl, ours, tasks
    from .cli import _load_or_make_tasks, _quiet_hf
    from .report import Report
    from .tagfree import STRICT, TAGFREE

    if reading != STRICT:
        sys.exit(f"--profile {profile} reports on the strict reading only (as fixed before the run)")
    _quiet_hf()
    m = R.mode(quick, root, profile=profile)
    if source is not None:
        m = replace(m, out=source)
    main_dir = main_from if main_from is not None else root / (QUICK_MAIN_FROM if quick
                                                               else MAIN_FROM)
    seeds, partial = arm_seeds(m.out)
    if not seeds:
        sys.exit(f"{R._shown(m.out, root)} has no trained and scored real-seed<N>/; "
                 "run sft-pair --profile onetool first")
    main_seeds, _ = arm_seeds(main_dir)
    if not (main_dir / R.UNTRAINED / "score.json").exists() or not main_seeds:
        sys.exit(f"{R._shown(main_dir, root)} has no scored untrained/ and real-seed<N>/")
    from . import sft

    trains = {d.name: json.loads((d / "train.json").read_text())
              for d in sorted(m.out.iterdir()) if (d / "train.json").exists()}
    main_trains = {s: json.loads((main_dir / R.run_label("real", s) / "train.json").read_text())
                   for s in main_seeds}
    bad = check_inputs({R.run_label("real", s): trains[R.run_label("real", s)].get("data_sha256")
                        for s in seeds},
                       {R.run_label("real", s): t.get("data_sha256")
                        for s, t in main_trains.items()},
                       sft.VARIANT_SHA256[T.PROFILES[profile].real_set],
                       sft.SET_SHA256["real.jsonl"])
    if bad:
        sys.exit("not reporting: a run did not train on its pinned set (check --from and "
                 "--main-from)\n  " + "\n  ".join(bad))
    un_info = json.loads((main_dir / R.UNTRAINED / "score.json").read_text())
    if un_info["model"] != m.settings.model:
        sys.exit(f"{R._shown(main_dir, root)} scored {un_info['model']}, not "
                 f"{m.settings.model}")

    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    test = ours.limit_per_kind(splits["test"], m.ours_limit)
    by_id = {t["id"]: t for t in splits["test"] + splits["train-sample"]}
    items = {cat: bfcl.load_items(cat, m.bfcl_limit, "eval") for cat in bfcl.CATEGORIES}

    def load(d: Path, rd: str) -> R.Scores:
        sc = R.load_scores(d, by_id, items, rd, samples=False)
        if list(sc.ours) != [t["id"] for t in test]:
            raise ValueError(f"{d}: its tasks are not the report's")
        return sc

    def arms(rd: str):
        return (load(main_dir / R.UNTRAINED, rd),
                [load(main_dir / R.run_label("real", s), rd) for s in main_seeds],
                [load(m.out / R.run_label("real", s), rd) for s in seeds])

    un, main, one = arms(STRICT)
    un_t, main_t, one_t = arms(TAGFREE)
    checks = json.loads((m.out / "checks.json").read_text()) if (
        m.out / "checks.json").exists() else {}

    rep = Report("tinytools sft-report" + m.title
                 + (" --quick (not for publication)" if quick else ""),
                 out=root / "outputs" / f"sft{m.suffix}.txt")
    rep.header()
    _setup(rep, m, root, main_dir, seeds, partial, main_seeds, un, main, one, test, items)
    _reading(rep, (un, main, one), (un_t, main_t, one_t), test, items)
    sets = _sets(test, items)
    _primary(rep, un, main, one, sets, seeds, main_seeds)
    _beside(rep, main, one, sets)
    _bfcl_outcomes(rep, un, main, one, items)
    R._r4c(rep, m, trains, seeds, checks)
    _training(rep, trains, main_trains, seeds, main_seeds)
    rep.close()
    return 0


def _sets(test: list[dict], items: dict) -> dict[str, tuple[str, list[str]]]:
    """The report's task sets: (source, ids) by name."""
    from . import tasks

    return {
        "one-function": ("bfcl", [i.id for c in ONE_FUNCTION for i in items[c]]),
        "multi-function": ("bfcl", [i.id for c in MULTI_FUNCTION for i in items[c]]),
        "ours, call tasks": ("ours", [t["id"] for t in test if t["kind"] != tasks.NO_FIT]),
        "ours, no fit": ("ours", [t["id"] for t in test if t["kind"] == tasks.NO_FIT]),
        "BFCL irrelevance": ("bfcl", [i.id for i in items["irrelevance"]]),
    }


def _setup(rep, m, root, main_dir, seeds, partial, main_seeds, un, main, one, test,
           items) -> None:
    from . import tasks

    rep.section("Setup")
    rep.kv("model", m.settings.model)
    R._profile_lines(rep, m)
    rep.kv("onetool real, seeds", ", ".join(map(str, seeds)))
    if partial:
        rep.kv("  not trained and scored", ", ".join(partial))
    rep.kv("the main run, read from", R._shown(main_dir, root), wrap=True)
    rep.kv("  its arms read", "untrained, real")
    rep.kv("  real, seeds", ", ".join(map(str, main_seeds)))
    rep.note("The untrained scores are the main run's: the same model,")
    rep.note("tasks, prompts and decoding, so it was not scored again.")
    diff = sorted({k for x in one for k in differing(x.info, main[0].info, SCORED_AS)}
                  | {k for x in main for k in differing(x.info, un.info, SCORED_AS)})
    rep.kv("scored as the main run's arms", not diff)
    rep.note("(model, engine, batch, dtype, device, generation seed, samples,")
    rep.note("temperature, max new tokens, task limits"
             + (f"; differ: {', '.join(diff)})" if diff else ")"), wrap=True)
    info = one[0].info
    rep.kv("scoring", f"{info['engine']} engine, batch {info['batch_size']}")
    rep.kv("  dtype, device", f"{info['dtype']}, {info['device']}")
    rep.kv("decoding", "greedy")
    rep.kv("max new tokens", info["max_new_tokens"])
    n = {k: sum(t["kind"] == k for t in test) for k in tasks.KINDS}
    rep.kv("our test tasks", f"{len(test)} ({' / '.join(str(n[k]) for k in tasks.KINDS)})")
    rep.kv("BFCL eval items", f"{sum(len(v) for v in items.values())} ("
           + " / ".join(str(len(v)) for v in items.values()) + ")")
    rep.kv("checked by", "sft-report, from the saved replies")
    rep.kv("intervals", "paired bootstrap over tasks")
    rep.note(f"({R.RESAMPLES_TEXT}.)")
    rep.note("An arm's per-task value is the mean of its seeds' 0/1 values.")


def _reading(rep, strict, tagfree, test, items) -> None:
    """Strict is the reading; how many replies the tag-free one would change."""
    rep.section("The reading: strict")
    rep.note("A call counts only inside <tool_call> tags. The Instruct model")
    rep.note("writes its calls in them; replies whose outcome the tag-free")
    rep.note("reading would change, ours and BFCL, greedy:")
    sets = _sets(test, items)
    ids = {src: [i for s_, ids_ in sets.values() if s_ == src for i in ids_]
           for src in ("ours", "bfcl")}
    ids = {k: list(dict.fromkeys(v)) for k, v in ids.items()}
    rows = []
    for label, a, b in (("untrained", [strict[0]], [tagfree[0]]),
                        ("main real", strict[1], tagfree[1]),
                        ("onetool real", strict[2], tagfree[2])):
        n = changed = 0
        for x, y in zip(a, b, strict=True):
            for src, ii in ids.items():
                n += len(ii)
                changed += sum(p != q for p, q in zip(R._outcomes(x, src, ii),
                                                      R._outcomes(y, src, ii), strict=True))
        rows.append([label, len(a), n, changed])
    rep.table(["arm", "runs", "replies", "changed"], rows)


def _primary(rep, un, main, one, sets, seeds, main_seeds) -> None:
    src, ids = sets["one-function"]
    p = compare([R._right(x, src, ids) for x in one], [R._right(x, src, ids) for x in main])
    u = E.exact_mean(R._right(un, src, ids), 1)
    rep.section("The prediction (fixed before the run)")
    rep.note("Leaving out the one-tool no-call rows raises right on BFCL's")
    rep.note("one-function call items; confirmed if the interval excludes 0")
    rep.note("upward. Paired by task, seed means.")
    rep.blank()
    rep.note(primary_line(p, seeds, main_seeds), wrap=True)
    rep.note(f"(untrained {E.percent(u)}, main real {E.percent(p.share_b)}, onetool real "
             f"{E.percent(p.share_a)}; {R.RESAMPLES_TEXT}, over {p.n} items.)", wrap=True)


def _beside(rep, main, one, sets) -> None:
    from .classify import DECLINED, NO_CALL, RIGHT

    rep.section("Beside it (descriptive)")
    rows = []
    spec = [("one-function", "right", {RIGHT}, True),
            ("one-function", "no call", {NO_CALL}, False),
            ("multi-function", "right", {RIGHT}, False),
            ("multi-function", "no call", {NO_CALL}, False),
            ("ours, call tasks", "right", {RIGHT}, False),
            ("ours, no fit", "declined", {DECLINED}, False),
            ("BFCL irrelevance", "declined", {DECLINED}, False)]
    for name, what, wanted, primary in spec:
        src, ids = sets[name]
        if not ids:
            continue
        p = compare([E.indicator(R._outcomes(x, src, ids), wanted) for x in one],
                    [E.indicator(R._outcomes(x, src, ids), wanted) for x in main])
        rows.append([f"{name}, {what}" + (" *" if primary else ""), len(ids),
                     E.percent(p.share_b), E.percent(p.share_a), E.points(p.diff),
                     E.interval(p.lo, p.hi)])
    rep.table(["set, share", "n", "main", "onetool", "points", "95% interval"], rows)
    rep.note("* the primary comparison, above. points: onetool - main real.")
    rep.note("one-function: BFCL simple + parallel; multi-function: multiple +")
    rep.note("parallel_multiple; ours, call tasks: the kinds that need a call;")
    rep.note("no fit / irrelevance: declined is right. Predicted before the run:")
    rep.note("multi-function and our tasks barely move (our tasks never offer")
    rep.note("one tool); declining: no prediction.")


def _bfcl_outcomes(rep, un, main, one, items) -> None:
    from .classify import NO_CALL, RIGHT, WRONG_ARGUMENTS, WRONG_COUNT, WRONG_FUNCTION, short_label

    cols = (RIGHT, NO_CALL, WRONG_ARGUMENTS, WRONG_FUNCTION, WRONG_COUNT)
    rep.section("BFCL call items: outcomes per category")
    rows = []
    for cat in (*ONE_FUNCTION, *MULTI_FUNCTION):
        ids = [i.id for i in items[cat]]
        if not ids:
            continue
        for j, (label, runs) in enumerate((("untrained", [un]), ("main real", main),
                                           ("onetool", one))):
            per = [R._outcomes(x, "bfcl", ids) for x in runs]
            rows.append([R._short(cat) if j == 0 else "", label, len(ids) if j == 0 else "",
                         *[mean_count(per, c) for c in cols]])
    rep.table(["category", "arm", "n", *[short_label(c) for c in cols]], rows)
    rep.note("Counts, greedy; main real and onetool: the mean over their seeds.")
    rep.note("The rest of n is unparseable replies. simple and parallel offer")
    rep.note("one function; multiple and par_multi several.")


def _training(rep, trains: dict, main_trains: dict, seeds, main_seeds) -> None:
    rep.section("Training runs (r4d), beside the main run's real arm")
    rows = []
    for arm, ss, get in (("onetool", seeds, lambda s: trains[R.run_label("real", s)]),
                         ("main real", main_seeds, lambda s: main_trains[s])):
        for s in ss:
            t = get(s)
            rows.append([arm, s, t["rows"], t["steps"], f"{t['losses'][0]:.4f}",
                         f"{t['losses'][-1]:.4f}", f"{t['seconds'] / 60:.1f}"])
    rep.table(["arm", "seed", "rows", "steps", "loss, first", "loss, last", "minutes"], rows)
    rep.note("loss: the step's mean over its scored tokens. minutes: wall clock,")
    rep.note("loading included (varies run to run).")
    notes = {trains[R.run_label("real", s)].get("note") for s in seeds} - {"", None}
    for n in notes:
        rep.note(n, wrap=True)
    rep.blank()
    first = trains[R.run_label("real", seeds[0])]["settings"]
    ref = main_trains[main_seeds[0]]["settings"]
    diff = differing(first, ref, sorted(set(first) | set(ref)))
    rep.kv("settings as the main run's", "yes" if not diff else "no: " + ", ".join(diff),
           wrap=True)
    rep.note("(every training setting; the set is the only change)")
    rep.note("Seed N sets LoRA A's init and the shuffle order, as in the main")
    rep.note("run. The shuffle permutes a different set (its rows, not main's),")
    rep.note("so the same seed need not give main's order:")
    both = [s for s in seeds if s in main_trains]
    rep.table(["seed", "same init (A) as main", "same first-step rows"],
              [[s, trains[R.run_label("real", s)]["init_sha256"] == main_trains[s]["init_sha256"],
                trains[R.run_label("real", s)]["first_step_rows"] ==
                main_trains[s]["first_step_rows"]] for s in both])
    rep.note("first-step rows: the first step's row ids, in order.")
    from . import sft

    pinned = sft.VARIANT_SHA256.get(T.PROFILES["onetool"].real_set)
    rep.kv("trained on the pinned set", all(trains[R.run_label("real", s)]["data_sha256"] == pinned
                                            for s in seeds))
    rep.note("(each run's data sha256 against sft.py's pin for the variant)")
