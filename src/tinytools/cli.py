"""``tinytools``: data and evaluation for the tool-calling posts (9-12).

Commands (``tinytools <command> --help`` for each one's flags):

    info            the device, the library versions, the pinned bfcl-eval
    toolace-stats   count ToolACE's rows, turns and calls; check the conversion
    toolace-convert write the English rows in Qwen2.5's tool format to data/
    bfcl-score      score a model on BFCL, split five ways; or sample it 8 times

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
    from .classify import OUTCOMES, WRONG_FUNCTION, mixed_share
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

    heads = ["category", "n", "no call", "unparse", "wrong fn", "wrong arg", "right", "bfcl"]
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

    args = p.parse_args(argv)
    t0 = time.perf_counter()
    args.fn(args)
    progress.status(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
