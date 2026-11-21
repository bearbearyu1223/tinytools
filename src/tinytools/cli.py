"""``tinytools``: data and evaluation for the tool-calling posts (9-12).

Commands (``tinytools <command> --help`` for each one's flags):

    info            the device, the library versions, the pinned bfcl-eval
    toolace-stats   count ToolACE's rows, turns and calls; check the conversion
    toolace-convert write the English rows in Qwen2.5's tool format to data/
    bfcl-score      score a model on BFCL, split five ways; or sample it 8 times
    tasks           generate our tool tasks (test / dev / train) and count them
    check-demo      three calls checked by exact match and by execution match
    score           score a model on our tool tasks, five ways; or sample it 8 times
    sft-data        the two SFT sets: ToolACE truncated (real) and random-call (control)

Every command prints a report on stdout (progress goes to stderr) and saves the
same report under outputs/, so ``tinytools bfcl-score > x.txt`` captures only
the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import __version__, progress
from .report import Report

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"


def _quiet_hf() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()


def _versions() -> list[tuple[str, str]]:
    from importlib.metadata import version

    return [(p, version(p)) for p in ("torch", "transformers", "bfcl-eval", "numpy")]


# -- info ------------------------------------------------------------------------


def cmd_info(args) -> None:
    from .device import get_device

    rep = Report("tinytools info", out=OUTPUTS / "info.txt")
    rep.header()
    rep.section("Versions")
    rep.kv("tinytools", __version__)
    for name, v in _versions():
        rep.kv(name, v)
    rep.kv("python", sys.version.split()[0])
    rep.kv("device chosen", str(get_device()))
    from importlib.metadata import version

    from .bfcl import LEADERBOARD_BFCL_EVAL

    rep.kv("leaderboard's bfcl-eval", LEADERBOARD_BFCL_EVAL)
    rep.kv("installed matches it", version("bfcl-eval") == LEADERBOARD_BFCL_EVAL)
    rep.close()


# -- ToolACE ---------------------------------------------------------------------


def _toolace(tok_id: str):
    """The census, the converted rows rendered to records, and the file's hash."""
    from transformers import AutoTokenizer

    from . import toolace

    path = toolace.download()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = toolace.load_rows(path)
    census, converted = toolace.census_and_convert(rows)
    tok = AutoTokenizer.from_pretrained(tok_id)
    on_item = progress.items("rendering converted rows with the chat template", len(converted))
    records = []
    for k, conv in enumerate(converted, start=1):
        records.append(toolace.render(conv, tok, census))
        on_item(k)
    return census, records, sha


def _print_census(rep: Report, c, sha: str, tok_id: str) -> None:
    from . import toolace

    rep.section("The file")
    rep.kv("dataset", f"{toolace.REPO_ID} @ {toolace.REVISION[:12]}")
    rep.kv("data.json sha256", sha[:16])
    rep.kv("rows", c.total)

    rep.section("Language")
    rep.note(f"Rule: a row is Chinese if the first {toolace.ZH_HEAD} characters of its")
    rep.note("system prompt contain a CJK ideograph (U+4E00-U+9FFF).")
    rep.kv("Chinese rows (dropped)", c.chinese)
    rep.kv("English rows", c.english)
    rep.kv("contrast: rows with any CJK", c.any_cjk)

    rep.section("English rows that do not convert, by reason")
    if c.failures:
        rep.table(["reason", "rows"], sorted(c.failures.items(), key=lambda kv: (-kv[1], kv[0])))
    rep.kv("rows converted", c.converted)
    rep.kv("  with an empty tool list", c.empty_tool_list)
    rep.kv("  with tool results", c.tool_turn_rows)
    rep.kv("  with a date/time line kept", c.context_kept)

    rep.section("Converted rows by number of assistant turns")
    rep.table(["assistant turns", "rows"], sorted(c.by_assistant_turns.items()))

    rep.section("Assistant turns: calls and not-calls")
    rep.note("A turn is a call if it starts with '[' and ends with ']' (and parses).")
    rep.note("Not-calls are split by what precedes them: a tool result (answering")
    rep.note("from it) or a user turn (asking for a missing argument, saying no")
    rep.note("function fits, or chatting).")
    rep.kv("assistant turns", c.assistant_turns)
    rep.kv("  calls", c.call_turns)
    rep.kv("    with 2+ calls in one turn", c.multi_call_turns)
    rep.kv("  not-calls after a tool result", c.text_after_tool)
    rep.kv("  not-calls after a user turn", c.text_after_user)
    rep.kv("calls in all call turns", c.calls)
    rep.kv("calls passing an unlisted argument", c.unknown_arg_calls)

    rep.section("Rows the random-call control can use")
    rep.kv("rows with 2+ tools in the list", c.two_plus_tools)
    rep.kv("  with at least one call", c.two_plus_tools_with_call)

    rep.section(f"Masking, tokenized with {tok_id}")
    rep.kv("tokens, all converted rows", c.tokens_total)
    rep.kv("tokens trained (assistant turns)", c.tokens_trainable)
    rep.kv("share trained", round(c.tokens_trainable / max(1, c.tokens_total), 4))
    rep.kv("tokens straddling a mask edge", c.tokens_straddling)


def cmd_toolace_stats(args) -> None:
    _quiet_hf()
    census, _, sha = _toolace(args.tokenizer)
    rep = Report("tinytools toolace-stats", out=OUTPUTS / "toolace-stats.txt")
    rep.header()
    _print_census(rep, census, sha, args.tokenizer)
    rep.close()


def cmd_toolace_convert(args) -> None:
    _quiet_hf()
    census, records, sha = _toolace(args.tokenizer)
    out = DATA / "toolace-en-qwen.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = Report("tinytools toolace-convert", out=OUTPUTS / "toolace-convert.txt")
    rep.header()
    _print_census(rep, census, sha, args.tokenizer)
    rep.section("Written")
    rep.kv("file", str(out.relative_to(ROOT)))
    rep.kv("records", len(records))
    rep.kv("sha256", hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    rep.note("Each record: id, tools, messages, text (the chat template's")
    rep.note("rendering) and train_spans (character spans of assistant turns).")
    rep.close()


# -- BFCL ------------------------------------------------------------------------


def _short_cat(cat: str) -> str:
    return {"parallel_multiple": "par_multiple"}.get(cat, cat)


def cmd_bfcl_score(args) -> None:
    _quiet_hf()
    from . import bfcl
    from .classify import OUTCOMES, WRONG_FUNCTION, mixed_share, short_label
    from .device import get_device

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in cats if c not in bfcl.CATEGORIES]
    if unknown:
        sys.exit(f"unknown categories {unknown}; choose from {list(bfcl.CATEGORIES)}")
    sampling = args.samples > 1
    if sampling and args.temperature <= 0:
        sys.exit("--samples > 1 needs --temperature > 0")
    mode = f"samples{args.samples}-t{args.temperature:g}" if sampling else "greedy"
    tag = f"{args.model.split('/')[-1]}-{mode}-limit{args.limit}"
    rep = Report(f"tinytools bfcl-score ({mode})", out=OUTPUTS / f"bfcl-{tag}.txt")
    rep.header()
    rep.section("Setup")
    rep.kv("model", args.model)
    rep.kv("bfcl-eval", bfcl.bfcl_version())
    rep.kv("items per category", f"first {args.limit} in bfcl's order")
    rep.kv("decoding", "greedy" if not sampling else
           f"{args.samples} samples, temperature {args.temperature:g}")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("seed", f"{args.seed} (+ crc32 of each item id)")

    device = get_device()
    progress.status(f"loading {args.model} on {device}")
    tok, model = bfcl.load_model(args.model, device)
    results, records = [], []
    for cat in cats:
        items = bfcl.load_items(cat, args.limit)
        on_item = progress.items(f"{cat}, {mode}", len(items))
        results.append(bfcl.score(tok, model, items, samples=args.samples,
                                  temperature=args.temperature, seed=args.seed,
                                  on_item=on_item, records=records))

    heads = ["category", "n", *[short_label(o) for o in OUTCOMES], "bfcl"]
    rows = [[_short_cat(r.category), r.n, *[r.outcomes[o] for o in OUTCOMES], r.bfcl_right]
            for r in results]
    rows.append(["total", sum(r.n for r in results),
                 *[sum(r.outcomes[o] for r in results) for o in OUTCOMES],
                 sum(r.bfcl_right for r in results)])
    rep.section("Every reply, split five ways" + (" (all samples)" if sampling else ""))
    rep.table(heads, rows)
    rep.note("bfcl = BFCL's own pass count. It equals 'right' on the AST")
    rep.note("categories; on irrelevance BFCL also passes replies whose")
    rep.note("<tool_call> fails to decode, which this split calls unparseable.")

    rep.section("Controls")
    rep.table(["category", "items", "reference passes", "replies hit token cap"],
              [[_short_cat(r.category), r.oracle_n, r.oracle_right, r.hit_cap] for r in results])
    rep.note("reference passes: BFCL's own answer, written in Qwen's format and")
    rep.note("scored by the same checker (the harness can give full marks).")
    rep.note("hit token cap: replies cut off mid-generation.")

    rep.section("'wrong fn' that is only a wrong number of calls")
    rep.table(["category", "wrong fn", "right names, wrong count"],
              [[_short_cat(r.category), r.outcomes[WRONG_FUNCTION], r.wrong_count_only]
               for r in results])
    rep.note("e.g. one spotify.play call where the reference has two.")

    if sampling:
        rep.section(f"Groups of {args.samples}: how many samples are right per task")
        dist = [[_short_cat(r.category),
                 *[sum(1 for x in r.right_per_task if x == k) for k in range(args.samples + 1)]]
                for r in results]
        rep.table(["category", *[str(k) for k in range(args.samples + 1)]], dist)
        rep.section("Groups with signal (some right, some wrong)")
        mix = []
        for r in results:
            none, mixed, every = mixed_share(r.right_per_task, args.samples)
            mix.append([_short_cat(r.category), len(r.right_per_task), none, mixed, every,
                        f"{mixed / max(1, len(r.right_per_task)):.2f}"])
        allc = [x for r in results for x in r.right_per_task]
        none, mixed, every = mixed_share(allc, args.samples)
        mix.append(["total", len(allc), none, mixed, every, f"{mixed / max(1, len(allc)):.2f}"])
        rep.table(["category", "tasks", "all wrong", "mixed", "all right", "mixed share"], mix)

    rep.section("Wall clock (varies run to run)")
    tsec = sum(r.seconds for r in results)
    titems = sum(len(r.right_per_task) for r in results)
    rep.table(["category", "tasks", "s/task", "new tokens/task"],
              [[_short_cat(r.category), len(r.right_per_task),
                f"{r.seconds / max(1, len(r.right_per_task)):.2f}",
                f"{r.new_tokens / max(1, len(r.right_per_task)):.0f}"] for r in results]
              + [["total", titems, f"{tsec / max(1, titems):.2f}",
                  f"{sum(r.new_tokens for r in results) / max(1, titems):.0f}"]])
    rep.close()

    items_path = OUTPUTS / f"bfcl-{tag}-replies.jsonl"
    with open(items_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    progress.status(f"replies written to {items_path.relative_to(ROOT)}")


# -- our tool tasks --------------------------------------------------------------

TASKS_DIR = DATA / "tasks"


def _block(rep: Report, text: str, indent: int = 4) -> None:
    """Print text inside the report, long lines broken with a trailing backslash."""
    from .ours import wrap_lines
    from .report import WIDTH

    for line in wrap_lines(text, WIDTH - indent):
        rep.print((" " * indent + line).rstrip())


def _load_or_make_tasks(seed: int) -> dict[str, list[dict]]:
    """The splits from data/tasks/, regenerated (and checked against the files) if missing."""
    from . import tasks

    splits = tasks.generate(seed)
    if seed == tasks.DEFAULT_SEED:
        for name, records in splits.items():
            p = TASKS_DIR / f"{name}.jsonl"
            if p.exists() and p.read_bytes() != tasks.jsonl_bytes(records):
                sys.exit(f"{p} differs from what seed {seed} generates; rerun `tinytools tasks`")
    return splits


def cmd_tasks(args) -> None:
    from . import tasks, tools

    splits = tasks.generate(args.seed)
    directory = TASKS_DIR if args.seed == tasks.DEFAULT_SEED else TASKS_DIR / f"seed{args.seed}"
    paths = tasks.write(splits, directory)
    suffix = "" if args.seed == tasks.DEFAULT_SEED else f"-seed{args.seed}"
    rep = Report("tinytools tasks", out=OUTPUTS / f"tasks{suffix}.txt")
    rep.header()

    rep.section("The frozen clock")
    c = tools.clock()
    rep.kv("date", f"{c['date']} ({c['weekday']})")
    rep.kv("time", c["time"])
    rep.kv("time zone", c["timezone"])
    rep.note("The system prompt states it as:")
    _block(rep, tools.context_line())

    rep.section("The data the tools run on")
    people = tools.contacts()
    cal = tools.calendar()
    rep.kv("contacts", len(people))
    rep.kv("  with a tricky spelling", sum(p["tricky_spelling"] for p in people))
    rep.kv("calendar events", len(cal))
    rep.kv("  first and last day", f"{cal[0]['date']} to {cal[-1]['date']}")
    rep.kv("documents", len(tools.documents()))
    rep.kv("tools that run", len(tools.TOOL_NAMES))
    for name in tools.TOOL_NAMES:
        rep.note(f"  {name}")
    rep.kv("distractors (schema only)", len(tools.DISTRACTOR_NAMES))
    for name in tools.DISTRACTOR_NAMES:
        rep.note(f"  {name}")
    rep.note("A distractor is offered like any tool but is never the answer.")

    names = list(splits)
    rep.section("Tasks by kind")
    rows = [[tasks.SHORT_KIND[k], *[sum(t["kind"] == k for t in splits[s]) for s in names]]
            for k in tasks.KINDS]
    rows.append(["total", *[len(splits[s]) for s in names]])
    rep.table(["kind", *names], rows)
    rep.note(f"seed {args.seed}; train-r6a is a seeded sample of train (post 10's")
    rep.note("ready-for-RL measurement), the same 60/20/20 mix.")

    rep.section("Tasks by tool")
    rows = []
    for k in tasks.KINDS:
        for tool in tasks.KIND_TOOLS[k]:
            label = tool if k != tasks.NO_FIT else f"needs {tool}"
            rows.append([label[:27], tasks.SHORT_KIND[k],
                         *[sum(t["kind"] == k and t["tool"] == tool for t in splits[s])
                           for s in names]])
    rep.table(["tool", "kind", *names], rows)
    rep.note("no tool fits: the tool the request needs, which is never offered.")

    rep.section("Tools offered per task")
    rows = []
    for s in names:
        n = [len(t["tools"]) for t in splits[s]]
        d = sum(any(x in tools.DISTRACTOR_NAMES for x in t["tools"]) for t in splits[s])
        rows.append([s, min(n), max(n), f"{sum(n) / len(n):.2f}", d])
    rep.table(["split", "min", "max", "mean", "with a distractor"], rows)

    rep.section("Unique requests, and overlap between splits")
    rows = [[s, len(splits[s]), len({t["request"] for t in splits[s]})] for s in names]
    rep.table(["split", "tasks", "unique request texts"], rows)
    main = list(tasks.SPLITS)

    def keyset(s, by):
        return {t["request"] if by == "text" else (t["template"], json.dumps(t["slots"],
                                                                                sort_keys=True))
                for t in splits[s]}
    pairs = [(a, b) for i, a in enumerate(main) for b in main[i + 1:]]
    rep.table(["splits", "same text", "same template+slots"],
              [[f"{a} / {b}", len(keyset(a, "text") & keyset(b, "text")),
                len(keyset(a, "slots") & keyset(b, "slots"))] for a, b in pairs])

    rep.section("Files (regenerated byte for byte from the seed)")
    rep.table(["split", "tasks", "sha256 (first 16)"],
              [[s, len(splits[s]), tasks.sha256(splits[s])[:16]] for s in names])
    rep.note(f"files: {paths['test'].parent.relative_to(ROOT)}/<split>.jsonl")
    rep.close()


# Three hand-written calls, each checked against a reference by the same functions
# `score` uses: one passes both checks, one passes execution only, one fails both.
CHECK_DEMO = [
    ("passes both: argument order, 90 vs 90.0, spacing",
     'Put "Budget sync" on my calendar for 2026-03-17 at 15:00 for 90 minutes.',
     {"name": "create_calendar_event", "arguments": {
         "title": "Budget sync", "date": "2026-03-17", "start_time": "15:00",
         "duration_minutes": 90}},
     {"name": "create_calendar_event", "arguments": {
         "duration_minutes": 90.0, "start_time": "15:00", "date": "2026-03-17",
         "title": "Budget  sync"}}),
    ("passes execution only: operands swapped",
     "What is 348 times 27?",
     {"name": "calculator", "arguments": {"expression": "348 * 27"}},
     {"name": "calculator", "arguments": {"expression": "27 * 348"}}),
    ("fails both: a broader search that includes the answer",
     "Find the Q3 sales report.",
     {"name": "search_documents", "arguments": {"query": "Q3 sales report"}},
     {"name": "search_documents", "arguments": {"query": "sales report"}}),
]


def _brief(result: dict) -> str:
    """A tool result on one line, record lists cut down to id and title/name."""
    if "error" in result or "result" in result:
        return json.dumps(result)
    (key, value), = result.items()
    if key == "created":
        return json.dumps(result)
    short = [{k: r[k] for k in ("id", "title", "name") if k in r} for r in value]
    return json.dumps({key: short})


def cmd_check_demo(args) -> None:
    from .checkers import ABS_TOL, REL_TOL, call_text, check_reply

    rep = Report("tinytools check-demo", out=OUTPUTS / "check-demo.txt")
    rep.header()
    rep.section("How a call is checked")
    rep.note("exact match: one call, the reference's tool, the same arguments")
    rep.note("  after normalizing argument order, number format (90 = 90.0) and")
    rep.note("  whitespace; an expression compares as a syntax tree.")
    rep.note("execution match: one call, the reference's tool, run on fresh data;")
    rep.note("  no error, and the same result: searches return the same set,")
    rep.note("  a created event leaves the same calendar, numbers agree within")
    rep.note(f"  {REL_TOL:g} relative (or {ABS_TOL:g} absolute).")
    for i, (label, request, ref, call) in enumerate(CHECK_DEMO, start=1):
        raw = f"<tool_call>\n{call_text(call['name'], call['arguments'])}\n</tool_call>"
        c = check_reply(raw, ref)
        rep.section(f"Call {i}: {label}")
        rep.note("request:")
        _block(rep, request)
        rep.note("reference call:")
        _block(rep, call_text(ref["name"], ref["arguments"]))
        rep.note("the call checked:")
        _block(rep, call_text(call["name"], call["arguments"]))
        rep.kv("exact match", c.exact)
        rep.kv("execution match", c.execution)
        rep.note("reference's result:")
        _block(rep, _brief(c.reference_run.result))
        rep.note("this call's result:")
        _block(rep, _brief(c.reply_run.result))
        rep.kv("execution match compared", _compared(ref["name"], c))
    rep.close()


def _compared(name: str, c) -> str:
    if name == "calculator":
        return f"{c.reference_run.compared} vs {c.reply_run.compared}"
    if name == "create_calendar_event":
        same = c.reference_run.compared == c.reply_run.compared
        return f"calendar after the call: {'same' if same else 'different'}"
    same = c.reference_run.compared == c.reply_run.compared
    return (f"id sets: {len(c.reference_run.compared)} vs {len(c.reply_run.compared)}, "
            f"{'same' if same else 'different'}")


def _tagname(model: str, split: str, mode: str, limit) -> str:
    return f"{model.split('/')[-1]}-{split}-{mode}-limit{limit if limit is not None else 'all'}"


def cmd_score(args) -> None:
    _quiet_hf()
    from . import bfcl, ours, tasks
    from .checkers import call_text, check_reply
    from .classify import OUTCOMES, mixed_share, short_label
    from .device import get_device

    sampling = args.samples > 1
    if sampling and args.temperature <= 0:
        sys.exit("--samples > 1 needs --temperature > 0")
    mode = f"samples{args.samples}-t{args.temperature:g}" if sampling else "greedy"
    tag = _tagname(args.model, args.split, mode, args.limit)
    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    items = ours.limit_per_kind(splits[args.split], args.limit)

    rep = Report(f"tinytools score ({args.split}, {mode})", out=OUTPUTS / f"score-{tag}.txt")
    rep.header()
    rep.section("Setup")
    rep.kv("model", args.model)
    rep.kv("tasks", f"{args.split} split, seed {tasks.DEFAULT_SEED}")
    rep.kv("  sha256 (first 16)", tasks.sha256(splits[args.split])[:16])
    rep.kv("tasks scored", "all" if args.limit is None else f"first {args.limit} of each kind, by id")
    rep.kv("decoding", "greedy (argmax)" if not sampling else
           f"{args.samples} samples, temperature {args.temperature:g}")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("seed", f"{args.seed} (+ crc32 of each task id)")
    rep.kv("right means", "execution match (exact beside it)")

    device = get_device()
    progress.status(f"loading {args.model} on {device}")
    tok, model = bfcl.load_model(args.model, device)

    if not sampling:
        _r1(rep, tok, model, ours.r1_task(splits["test"]), args.seed)

    greedy = ours.score(tok, model, items, samples=1, temperature=0.0, seed=args.seed,
                        on_item=progress.items(f"{args.split}, greedy", len(items)))
    sampled = None
    if sampling:
        sampled = ours.score(tok, model, items, samples=args.samples,
                             temperature=args.temperature, seed=args.seed,
                             on_item=progress.items(f"{args.split}, {mode}", len(items)))
    main = sampled if sampling else greedy

    heads = ["kind", "n", *[short_label(o) for o in OUTCOMES], "exact"]
    rep.section("Every reply, split five ways" + (" (all samples)" if sampling else ""))
    rep.table(heads, [[tasks.SHORT_KIND.get(r.kind, r.kind), r.n,
                       *[r.outcomes.get(o, 0) for o in OUTCOMES], r.exact]
                      for r in ours.outcome_rows(main)])
    rep.note("right = execution match; exact = exact match (the same replies).")
    rep.note("On no-tool-fits tasks right is no call; a call is 'wrong fn'.")
    rep.blank()
    rep.table(["kind", "right by execution", "right by exact match"],
              [[tasks.SHORT_KIND.get(r.kind, r.kind), ours.pct(r.execution, r.n),
                ours.pct(r.exact, r.n)] for r in ours.outcome_rows(main)])

    rep.section("Controls, scored by the same checks")
    ref_right = never_right = 0
    for s in main:
        ref = s.task["reference"]
        raw = ("I can't do that with these tools." if ref is None else
               f"<tool_call>\n{call_text(ref['name'], ref['arguments'])}\n</tool_call>")
        c = check_reply(raw, ref)
        ref_right += c.execution and c.exact
        never_right += check_reply("I can't do that with these tools.", ref).execution
    rep.kv("reference answers right (both)", ours.pct(ref_right, len(main)))
    rep.kv("never calling right", ours.pct(never_right, len(main)))
    rep.note("reference answers: the harness can give full marks. never calling:")
    rep.note("what declining every task scores (right on no-tool-fits only).")

    exec_only, exact_only = ours.disagreements(main)
    rep.section("Execution match but not exact match")
    rep.kv("replies", len(exec_only))
    for s, i in exec_only:
        call = s.checks[i].call
        rep.print(f"  {s.task['id']}" + (f" sample {i + 1}" if sampling else "")
                  + f"  {s.task['tool']}")
        _block(rep, "ref: " + call_text(s.task["reference"]["name"],
                                        s.task["reference"]["arguments"]), indent=6)
        _block(rep, "got: " + call_text(*call), indent=6)
    rep.section("Exact match but not execution match (a check: should be 0)")
    rep.kv("replies", len(exact_only))
    for s, _ in exact_only:
        rep.print(f"  {s.task['id']}  {s.task['tool']}")

    if sampling:
        g = args.samples
        right = {k: [s.right for s in sampled if s.task["kind"] == k] for k in tasks.KINDS}
        allr = [s.right for s in sampled]
        rep.section(f"Groups of {g}: how many samples are right per task")
        rep.table(["kind", *[str(k) for k in range(g + 1)]],
                  [[tasks.SHORT_KIND[k], *ours.histogram(right[k], g)] for k in tasks.KINDS
                   if right[k]] + [["total", *ours.histogram(allr, g)]])
        rep.section("Groups with signal (some right, some wrong)")
        rows = []
        for label, rc in [*[(tasks.SHORT_KIND[k], right[k]) for k in tasks.KINDS if right[k]],
                          ("total", allr)]:
            none, mixed, every = mixed_share(rc, g)
            rows.append([label, len(rc), none, mixed, every,
                         f"{100 * mixed / len(rc):.1f}%", f"{ours.per_thousand(mixed, len(rc), g):.0f}"])
        rep.table(["kind", "tasks", "all wrong", "mixed", "all right", "share", "per 1k"], rows)
        rep.note("share: mixed groups / tasks.")
        rep.note(f"per 1k: mixed groups per 1,000 answers = (1000 / {g}) x share.")
        rep.section(f"Greedy right vs pass@{g}")
        rows = []
        for k in [*tasks.KINDS, "total"]:
            gs = [s for s in greedy if k == "total" or s.task["kind"] == k]
            ss = [s for s in sampled if k == "total" or s.task["kind"] == k]
            if not gs:
                continue
            rows.append([tasks.SHORT_KIND.get(k, k), len(gs), ours.pct(sum(s.right for s in gs), len(gs)),
                         ours.pct(ours.pass_at_k([s.right for s in ss]), len(ss))])
        rep.table(["kind", "tasks", "greedy right", f"pass@{g}"], rows)
        rep.note(f"pass@{g}: tasks with at least one right answer in {g} samples.")

    rep.section("Wall clock (varies run to run)")
    rows = []
    for label, res in [("greedy", greedy)] + ([(mode, sampled)] if sampling else []):
        secs = sum(s.seconds for s in res)
        rows.append([label, len(res), f"{secs / max(1, len(res)):.2f}",
                     f"{sum(s.new_tokens for s in res) / max(1, len(res)):.0f}",
                     sum(sum(s.capped) for s in res)])
    rep.table(["pass", "tasks", "s/task", "new tokens/task", "hit token cap"], rows)
    rep.close()

    path = OUTPUTS / f"score-{tag}-replies.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for label, res in [("greedy", greedy)] + ([(mode, sampled)] if sampling else []):
            for s in res:
                for i, (text, c, capped) in enumerate(zip(s.replies, s.checks, s.capped,
                                                          strict=True)):
                    f.write(json.dumps({
                        "id": s.task["id"], "pass": label, "sample": i, "kind": s.task["kind"],
                        "tool": s.task["tool"], "request": s.task["request"], "reply": text,
                        "outcome": c.outcome, "execution": c.execution, "exact": c.exact,
                        "hit_cap": capped,
                        "result": c.reply_run.result if c.reply_run else None,
                    }, ensure_ascii=False) + "\n")
    progress.status(f"replies written to {path.relative_to(ROOT)}")


def _r1(rep: Report, tok, model, task: dict, seed: int) -> None:
    """r1a and r1b: one test task's whole prompt, and the model's reply run through the tool."""
    from . import bfcl, ours, tasks
    from .checkers import call_text, check_reply

    text = ours.prompt(tok, task)
    census = ours.prompt_census(tok, text)
    (OUTPUTS / "r1-prompt.txt").write_text(text)
    rep.section(f"One task, whole: {task['id']} ({task['kind']}, {task['tool']})")
    rep.note("The templated prompt, exactly as the model reads it (Qwen2.5's")
    rep.note("chat template; long lines broken with a trailing \\; unbroken copy")
    rep.note("in outputs/r1-prompt.txt):")
    rep.blank()
    _block(rep, text)
    rep.blank()
    rep.kv("prompt tokens", census.tokens)
    rep.kv("  the tool list (<tools> block)", census.tool_list_tokens)
    rep.kv("  the whole tools section", census.tools_section_tokens)
    rep.kv("  tokens straddling those edges", census.straddling)

    cfg = bfcl.generation_config(tok, model, 1, 0.0)
    g = bfcl.generate_text(tok, model, text, task["id"], cfg, seed)
    raw = g.texts[0]
    c = check_reply(raw, task["reference"])
    rep.section("The model's reply, parsed and run")
    rep.note("raw reply (up to the end-of-turn token):")
    _block(rep, raw)
    rep.note("parsed call:")
    _block(rep, call_text(*c.call) if c.call else "(none: no single decodable call)")
    rep.note("the tool's result, fed back as the next turn:")
    if c.call and c.reply_run is not None:
        msgs = [*tasks.messages(task),
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": c.call[0], "arguments": c.call[1]}}]},
                {"role": "tool", "content": json.dumps(c.reply_run.result)}]
        full = tok.apply_chat_template(msgs, tools=tasks.tool_list(task), tokenize=False)
        start = full.index("<|im_start|>user\n<tool_response>")
        _block(rep, full[start:].rstrip("\n"))
    else:
        _block(rep, "(nothing to run)")
    rep.note("reference call:")
    _block(rep, call_text(task["reference"]["name"], task["reference"]["arguments"]))
    rep.kv("outcome", c.outcome)
    rep.kv("exact match / execution match", f"{'yes' if c.exact else 'no'} / "
           f"{'yes' if c.execution else 'no'}")


# -- the SFT training sets -----------------------------------------------------------

SFT_DIR = DATA / "sft"


def _of(n: int, total: int) -> str:
    return f"{n} of {total} ({100 * n / max(1, total):.1f}%)"


def cmd_sft_data(args) -> None:
    _quiet_hf()
    from transformers import AutoTokenizer

    from . import sft

    src = ROOT / sft.CONVERTED
    if not src.exists():
        sys.exit(f"{sft.CONVERTED} is missing; run `tinytools toolace-convert` first")
    raw = src.read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()
    converted = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    on_item = progress.items("truncating, drawing random calls, rendering", len(converted))
    s = sft.build_sets(converted, args.seed, tok, on_item)

    default = args.seed == sft.DEFAULT_SEED
    directory = SFT_DIR if default else SFT_DIR / f"seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, records in (("real", s.real), ("control", s.control)):
        data = sft.jsonl_bytes(records)
        (directory / f"{name}.jsonl").write_bytes(data)
        files[name] = (len(records), hashlib.sha256(data).hexdigest())

    suffix = "" if default else f"-seed{args.seed}"
    rep = Report("tinytools sft-data", out=OUTPUTS / f"sft-data{suffix}.txt")
    rep.header()

    rep.section("Input")
    rep.kv("converted ToolACE", sft.CONVERTED)
    rep.kv("  sha256 (first 16)", src_sha[:16])
    rep.kv("  is toolace-convert's output", src_sha == sft.CONVERTED_SHA256)
    rep.kv("rows", len(converted))
    rep.kv("tokenizer (chat template)", args.tokenizer)
    rep.kv("seed", f"{args.seed} (each row: Random(\"seed:row id\"))")

    rep.section("Truncation: each row cut after its first call turn")
    rep.note("Both arms. Not-calling assistant turns before the first call stay;")
    rep.note("a row with no call is kept whole. Turns after the cut (tool")
    rep.note("results, answers, later requests) are removed.")
    sub_label = f"{sft.MIN_TOOLS}+ tools"
    cut_rows = [
        ["rows", s.cut_all.rows, s.cut_subset.rows],
        ["rows cut", s.cut_all.rows_cut, s.cut_subset.rows_cut],
        ["turns removed", s.cut_all.turns_removed, s.cut_subset.turns_removed],
        ["  of them assistant turns", s.cut_all.assistant_turns_removed,
         s.cut_subset.assistant_turns_removed],
        ["tokens removed", s.cut_all.tokens_removed, s.cut_subset.tokens_removed],
        ["  of them trained", s.cut_all.trained_removed, s.cut_subset.trained_removed],
    ]
    rep.table(["", "all converted", sub_label], cut_rows)

    rep.section(f"The training rows ({sub_label}) beside the full set")
    cols = [s.shape_converted, s.shape_truncated, s.shape_real, s.shape_control]
    fields = [
        ("rows", "rows"),
        ("assistant turns", "assistant_turns"),
        ("  call turns", "call_turns"),
        ("    parallel (2+ calls)", "parallel_turns"),
        ("  not-calls after user", "not_calls_after_user"),
        ("  not-calls after tool", "not_calls_after_tool"),
        ("calls", "calls"),
        ("tool results", "tool_results"),
        ("tokens", "tokens"),
        ("tokens trained", "trained"),
    ]
    rows = [[label, *[getattr(c, f) for c in cols]] for label, f in fields]
    rows.append(["share trained", *[f"{c.trained / max(1, c.tokens):.4f}" for c in cols]])
    rep.table(["", "converted", "truncated", "real arm", "control"], rows)
    rep.note(f"converted: all {s.shape_converted.rows} rows as toolace-convert wrote them;")
    rep.note("truncated: the same rows cut; real arm: the truncated rows with")
    rep.note(f"{sub_label}; control: their random-call twins. Tokens are the chat")
    rep.note("template's rendering; trained = inside assistant turns.")

    rep.section("Tokens trained, real vs control")
    rep.kv("real arm", s.shape_real.trained)
    rep.kv("control", s.shape_control.trained)
    rep.kv("control / real", f"{s.shape_control.trained / max(1, s.shape_real.trained):.4f}")

    cs = s.control_stats
    rep.section("Control: does the drawn tool match the call it replaced?")
    rep.note("Each call's tool is drawn uniformly from the row's k tools, so it")
    rep.note("matches the replaced call's tool with chance 1/k.")
    rep.kv("control calls", cs.calls)
    rep.kv("drew the replaced call's tool", _of(cs.hits, cs.calls))
    rep.kv("expected (sum of 1/k)", f"{cs.expected:.1f} (sd {cs.variance ** 0.5:.1f})")
    rep.blank()
    krows = [[k, cs.rows_by_k[k], cs.calls_by_k[k], cs.hits_by_k[k],
              f"{cs.calls_by_k[k] / k:.1f}"] for k in sorted(cs.rows_by_k)]
    rep.table(["tools offered (k)", "rows", "calls", "drew it", "expected"], krows)

    rep.section("Control: where each call's arguments came from")
    rep.note("1 another row: a call to the drawn tool in another row that fits")
    rep.note("  this row's schema; 2 the same, schema differs; 3 this row: the")
    rep.note("  drawn tool is the replaced call's and no other row calls it, so")
    rep.note("  the real arguments are kept; 4 built: no other row calls the")
    rep.note("  tool, required parameters filled one by one.")
    src_rows = [[name, cs.source_hit[name], cs.source_miss[name],
                 cs.source_hit[name] + cs.source_miss[name],
                 f"{cs.mean_chars_change(name):+.1f}"] for name in sft.SOURCES]
    src_rows.append(["total", cs.hits, cs.calls - cs.hits, cs.calls,
                     f"{sum(cs.chars_change.values()) / max(1, cs.calls):+.1f}"])
    rep.table(["source", "drew it", "drew another", "calls", "chars"], src_rows)
    rep.note("chars: mean change in the arguments' JSON length against the")
    rep.note("call replaced (built calls leave optional parameters out).")
    rep.kv("calls equal to the real call", _of(cs.identical, cs.calls))
    rep.kv("built calls with a placeholder", _of(cs.with_placeholder,
                                                  cs.source_miss[sft.BUILT]
                                                  + cs.source_hit[sft.BUILT]))
    rep.blank()
    rep.table(["built calls' parameters, filled from", "parameters"],
              [[p, cs.param_sources[p]] for p in sft.PARAM_SOURCES])

    _r4b(rep, s, tok, sft)

    rep.section("Files (regenerated byte for byte from the input and seed)")
    rep.table(["file", "records", "sha256 (first 16)"],
              [[str((directory / f"{n}.jsonl").relative_to(ROOT)), k, h[:16]]
               for n, (k, h) in files.items()])
    rep.close()


def _r4b(rep: Report, s, tok, sft) -> None:
    real = next(r for r in s.real if r["id"] == sft.EXAMPLE_ID)
    twin = next(r for r in s.control if r["id"] == sft.EXAMPLE_ID)
    rep.section(f"One training row and its control twin ({sft.EXAMPLE_ID})")
    rep.kv("tools offered", ", ".join(t["function"]["name"] for t in real["tools"]))
    for label, rec in (("real arm", real), ("control", twin)):
        enc = tok(rec["text"], return_offsets_mapping=True, add_special_tokens=False)
        turns = sft.turn_tokens(rec["text"], enc["offset_mapping"],
                                [tuple(x) for x in rec["train_spans"]])
        rep.blank()
        rep.note(f"{label}, turn by turn:")
        rows = [[role, n, t, n - t] for role, n, t in turns]
        rows.append(["total", rec["n_tokens"], rec["n_trained"],
                     rec["n_tokens"] - rec["n_trained"]])
        rep.table(["turn", "tokens", "scored", "masked"], rows)
    rep.note("The system turn holds the date line and the tool list. An")
    rep.note("assistant turn's '<|im_start|>assistant' header is masked.")
    rep.blank()
    user = next(m["content"] for m in real["messages"] if m["role"] == "user")
    rep.note("user:")
    _block(rep, user, indent=4)
    for label, rec in (("real call:", real), ("control call:", twin)):
        rep.note(label)
        for m in rec["messages"]:
            for tc in m.get("tool_calls") or []:
                _block(rep, json.dumps(tc["function"], ensure_ascii=False), indent=4)
    e = twin["random_calls"][0]
    rep.kv("control's tool", f"{e['drawn']} (replaced {e['original']})")
    rep.kv("arguments", f"{e['source']}: copied from {e['copied_from']}")


# -- entry point -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tinytools", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="device and versions").set_defaults(fn=cmd_info)

    for name, fn, text in [
        ("toolace-stats", cmd_toolace_stats, "count ToolACE and check the conversion"),
        ("toolace-convert", cmd_toolace_convert, "write the converted English rows"),
    ]:
        s = sub.add_parser(name, help=text)
        s.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="whose chat template to render with")
        s.set_defaults(fn=fn)

    s = sub.add_parser("bfcl-score", help="score a model on BFCL, five-way split")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--categories", default="simple,multiple,parallel,parallel_multiple,irrelevance")
    s.add_argument("--limit", type=int, default=20, help="first N items per category")
    s.add_argument("--samples", type=int, default=1, help="1 = greedy; >1 = sampled group")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_bfcl_score)

    s = sub.add_parser("tasks", help="generate our tool tasks (test, dev, train) and count them")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_tasks)

    sub.add_parser("check-demo", help="three calls checked by exact and execution match"
                   ).set_defaults(fn=cmd_check_demo)

    s = sub.add_parser("score", help="score a model on our tool tasks, five-way split")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--split", choices=["test", "dev", "train-r6a"], default="test")
    s.add_argument("--limit", type=int, default=None, help="first N tasks of each kind, by id")
    s.add_argument("--samples", type=int, default=1, help="1 = greedy; >1 = sampled group")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("sft-data", help="the two SFT sets: real (truncated) and random-call")
    s.add_argument("--seed", type=int, default=0, help="the control's draws")
    s.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="whose chat template to render with")
    s.set_defaults(fn=cmd_sft_data)

    args = p.parse_args(argv)
    t0 = time.perf_counter()
    args.fn(args)
    progress.status(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
