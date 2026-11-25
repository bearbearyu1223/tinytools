"""``tinytools``: data and evaluation for the tool-calling posts (9-12).

Commands (``tinytools <command> --help`` for each one's flags):

    info            the device, the library versions, the pinned bfcl-eval
    toolace-stats   count ToolACE's rows, turns and calls; check the conversion
    toolace-convert write the English rows in Qwen2.5's tool format to data/
    bfcl-score      score a model on BFCL, outcome by outcome; or sample it 8 times
    tasks           generate our tool tasks (test / dev / train) and count them
                    (--onetool: post 10's one-tool kinds, their sub-kinds, the step mix)
    bfcl-functions  functions offered per BFCL item, every item and after the dev slice
    check-demo      three calls checked by exact match and by execution match
    score           score a model on our tool tasks, outcome by outcome; or sample 8
    rescore         re-check a score replies file with the current checkers (no model)
    bfcl-rescore    re-judge a bfcl-score replies file with bfcl's checker (no model)
    prompt-view     r1's prompt laid out part by part, its tokens by part, its tool
    sft-data        the SFT sets: ToolACE truncated (real), random-call per seed (control)
    figures         the post's figures, parsed from the captures, into figures/
    milestone0      six models on the dev split, the model rule's verdict, LoRA vs full timing
    engine-check    the batched engine against the one-at-a-time engine it replaced
    sft-check       before training: template, set sha256, lengths, memory, merge checks
    sft-score       score the untrained model, or a saved adapter merged into it
                    (--noise-floor: the untrained train-sample samples again, seed 1)
    sft-train       train one LoRA adapter (--repeat: seed 1 real again, compared)
    sft-pair        one seed's pair: train real, score, train control, score
                    (--profile onetool: the real arm only, train and score)
    sft-report      r4c, r4d, r5a-d, r6a from the saved runs and replies (no model);
                    --reading tagfree: on the tag-free reading
    sft-followup    section 5's follow-ups: ToolACE's no-call wording, BFCL no-call
                    replies and the parameters they ask for, right pooled (no model)

Every command prints a report on stdout (progress goes to stderr) and saves the
same report under outputs/, so ``tinytools bfcl-score > x.txt`` captures only
the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

from . import __version__, progress
from .report import WIDTH, Report

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
PROFILE_NAMES = ("main", "base", "onetool")  # train.PROFILES (not imported here: it pulls in lora/torch)


def _profile_tokenizer(profile: str, model: str) -> str | None:
    """The tokenizer a profile scores ``model`` with; ``None``: the model's own (main)."""
    from .train import PROFILES

    p = PROFILES[profile]
    if p.tokenizer is None:
        return None
    try:
        return p.tokenizer_for(model)
    except ValueError as e:
        sys.exit(str(e))


def _r1_path(profile: str, model: str) -> Path:
    """Where r1's unbroken prompt goes: outputs/r1-prompt.txt (main, as it was), else the
    profile's output directory (outputs/base/ or outputs/base-quick/ by the model)."""
    if profile == "main":
        return OUTPUTS / "r1-prompt.txt"
    from .train import PROFILES

    p = PROFILES[profile]
    return OUTPUTS / p.out_name(model == p.quick.model) / "r1-prompt.txt"


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
    if c.failures[toolace.NOT_JSON]:
        rep.blank()
        rep.note(f"The {c.failures[toolace.NOT_JSON]} rows whose tool list is not JSON, by how "
                 "the list is")
        rep.note("written (detected from its first characters):")
        rep.table(["format", "rows"], [[f, c.not_json_formats[f]] for f in toolace.LIST_FORMATS])
        rep.note("JSON with other keys: \"tool_name\", \"definition\", \"arguments\"")
        rep.note("in place of the name / description / parameters schema.")
        rep.blank()
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
    rep.kv("tokens scored (assistant turns)", c.tokens_trainable)
    rep.kv("share scored", round(c.tokens_trainable / max(1, c.tokens_total), 4))
    rep.note("scored: the tokens the loss is computed on, each assistant turn")
    rep.note("from after its '<|im_start|>assistant' header through <|im_end|>.")
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
    return {"parallel_multiple": "par_multi"}.get(cat, cat)


def _mixed_table(rep: Report, first: str, groups: list[tuple[str, list[int]]],
                 samples: int) -> None:
    """Groups with signal: one table, one format, in ``score`` and ``bfcl-score``."""
    from .ours import MIXED_HEADERS, mixed_rows

    rep.section("Groups with signal (some right, some wrong)")
    rep.table([first, *MIXED_HEADERS], mixed_rows(groups, samples))
    rep.note("share: mixed groups / tasks.")
    rep.note(f"per 1k: mixed groups per 1,000 answers = (1000 / {samples}) x share.")
    rep.note("Both rounded half up.")


def _bfcl_tag(model: str, split: str, mode: str, limit: int | None) -> str:
    """The capture's name; split "all" keeps the name captures had before splits existed."""
    part = "" if split == "all" else f"{split}-"
    return f"{model.split('/')[-1]}-{part}{mode}-limit{limit if limit is not None else 'all'}"


BFCL_REPLIES = re.compile(r"bfcl-(?P<model>.+?)-(?:(?P<split>dev|eval)-)?"
                          r"(?P<mode>greedy|samples(?P<g>\d+)-t(?P<t>[0-9.]+))"
                          r"-limit(?P<limit>\d+|all)-replies\.jsonl")


def parse_bfcl_replies_name(name: str) -> dict | None:
    """What a ``bfcl-score`` replies file's name says: model, split, mode, samples,
    temperature, limit; ``None`` if it is not such a name."""
    m = BFCL_REPLIES.fullmatch(name)
    if m is None:
        return None
    sampling = m.group("g") is not None
    return {"model": m.group("model"), "split": m.group("split") or "all",
            "mode": m.group("mode"), "samples": int(m.group("g")) if sampling else 1,
            "temperature": float(m.group("t")) if sampling else 0.0,
            "limit": None if m.group("limit") == "all" else int(m.group("limit"))}


def _bfcl_setup(rep: Report, *, model: str, split: str, limit: int | None, sampling: bool,
                samples: int, temperature: float, seed: int, dtype: str,
                determinism: str, engine: tuple[str, int] | None) -> None:
    from . import bfcl

    rep.header()
    rep.section("Setup")
    rep.kv("model", model)
    rep.kv("bfcl-eval", bfcl.bfcl_version())
    rep.kv("split", {"eval": "eval: every item minus the dev slice",
                     "dev": "dev: the dev slice only",
                     "all": "all: every item, dev slice included"}[split])
    rep.kv("  dev slice", f"{bfcl.DEV_SHARE}% of each category, seed {bfcl.DEV_SEED}")
    if split == "all":
        rep.note("WARNING: this split includes the dev slice, which is held out of")
        rep.note("every published BFCL number; publish from --split eval.")
    rep.kv("items per category", "all in the split" if limit is None
           else f"first {limit} in bfcl's order")
    rep.kv("decoding", "greedy" if not sampling else
           f"{samples} samples, temperature {temperature:g}")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("dtype", dtype)
    rep.kv("determinism", determinism, wrap=True)
    _seed_and_engine(rep, seed, engine, samples, "item")


def _seed_and_engine(rep: Report, seed: int, engine: tuple[str, int] | None, samples: int,
                     noun: str) -> None:
    """The seed line, and the engine lines of a batched capture. ``engine`` is
    (name, batch size), or ``None`` for a capture of the one-at-a-time engine, whose
    Setup is printed as it always was."""
    from .engine import print_setup

    if engine is None:
        rep.kv("seed", f"{seed} (+ crc32 of each {noun} id)")
        return
    rep.kv("seed", f"{seed} (+ {noun} id and sample number)")
    if samples > 1:
        rep.note("Each reply draws from its own generator, seeded with the seed,")
        rep.note(f"its {noun} id and sample number: its draws don't depend on its batch.")
    print_setup(rep, engine[1], (samples,))


def _batch_size(args, device) -> int:
    from .engine import default_batch_size

    if args.batch_size is not None and args.batch_size < 1:
        sys.exit("--batch-size must be at least 1")
    return args.batch_size or default_batch_size(device.type)


def _check_sampling(samples: int, temperature: float) -> None:
    """Greedy is ``--samples 1`` at temperature 0; sampling needs more than one sample."""
    if samples > 1 and temperature <= 0:
        sys.exit("--samples > 1 needs --temperature > 0")
    if samples == 1 and temperature > 0:
        sys.exit("--samples 1 with --temperature > 0 would sample one reply but be labelled "
                 "greedy; use --temperature 0 for greedy, or --samples 2 or more")


def _cap_table(rep: Report, first: str, rows: list[tuple[str, int, Counter]]) -> None:
    """Under a call-needing outcome table: per category, the model's replies cut off at
    the token cap, and the outcome each was counted as in the table above."""
    from .classify import OUTCOMES, short_label

    def counted(c: Counter) -> str:
        return ", ".join(f"{short_label(o)} {c[o]}" for o in OUTCOMES if c[o]) or "-"

    total = Counter()
    for _, _, c in rows:
        total.update(c)
    rep.blank()
    rep.table([first, "hit token cap", "counted above as"],
              [[label, n, counted(c)] for label, n, c in rows]
              + [["total", sum(n for _, n, _ in rows), counted(total)]])
    rep.note("hit token cap: the model's reply ran to max new tokens (Setup)")
    rep.note("and was cut off there; it is counted in the table above.")


def _no_fit_cap_lines(rep: Report, first: str, rows: list[tuple[str, dict, Counter]]) -> None:
    """Under a no-tool-fits table: of the declined and called-anyway replies, how many
    hit the token cap. ``rows``: (label, outcome counts, capped replies' outcomes)."""
    from .classify import NO_FIT_OUTCOMES

    rep.blank()
    rep.table([first, "outcome", "replies", "of which hit token cap"],
              [[label, o, counts.get(o, 0), c[o]] for label, counts, c in rows
               for o in NO_FIT_OUTCOMES])
    rep.note("hit token cap: the model's reply ran to max new tokens (Setup)")
    rep.note("and was cut off there. Cut off before any <tool_call> tag, it")
    rep.note("counts as declined.")


def _bfcl_body(rep: Report, results: list, *, sampling: bool, samples: int,
               wall: list[str] | None) -> None:
    """Everything ``bfcl-score`` prints after Setup, from judged replies (no model needed).

    ``wall`` replaces the wall-clock table with an earlier capture's lines when the
    replies carry no timings (``bfcl-rescore`` of an older file).
    """
    from .classify import CALL_OUTCOMES, NO_FIT_OUTCOMES, short_label

    calls = [r for r in results if r.category != "irrelevance"]
    no_fit = [r for r in results if r.category == "irrelevance"]
    tail = " (all samples)" if sampling else ""
    if calls:
        rep.section("Tasks that need a call: every reply" + tail)
        rows = [[_short_cat(r.category), r.n, *[r.outcomes[o] for o in CALL_OUTCOMES]]
                for r in calls]
        rows.append(["total", sum(r.n for r in calls),
                     *[sum(r.outcomes[o] for r in calls) for o in CALL_OUTCOMES]])
        rep.table(["category", "n", *[short_label(o) for o in CALL_OUTCOMES]], rows)
        rep.note("wrong fn: names differ as a set; wrong n: the right names, too")
        rep.note("many or too few calls; bad args: the right calls, arguments the")
        rep.note("checker rejects; right: BFCL's checker accepts the reply (BFCL's")
        rep.note("accuracy on these categories).")
        _cap_table(rep, "category", [(_short_cat(r.category), r.hit_cap, r.capped_outcomes)
                                     for r in calls])
    if no_fit:
        rep.section("Tasks where no tool fits: every reply" + tail)
        rep.table(["category", "n", *[short_label(o) for o in NO_FIT_OUTCOMES], "bfcl right"],
                  [[_short_cat(r.category), r.n, *[r.outcomes[o] for o in NO_FIT_OUTCOMES],
                    r.bfcl_right] for r in no_fit])
        rep.note("declined: no <tool_call> tag (right); called anyway: any tag.")
        rep.note("bfcl right: BFCL's own verdict, which also passes a reply whose")
        rep.note("<tool_call> fails to decode (here it called anyway).")
        _no_fit_cap_lines(rep, "category", [(_short_cat(r.category), r.outcomes,
                                             r.capped_outcomes) for r in no_fit])

    rep.section("Controls")
    rep.table(["category", "items", "reference passes"],
              [[_short_cat(r.category), r.oracle_n, r.oracle_right] for r in results])
    rep.note("reference passes: BFCL's own answer, written in Qwen's format and")
    rep.note("scored by the same checker (the harness can give full marks).")

    if sampling:
        rep.section(f"Groups of {samples}: how many samples are right per task")
        dist = [[_short_cat(r.category),
                 *[sum(1 for x in r.right_per_task if x == k) for k in range(samples + 1)]]
                for r in results]
        rep.table(["category", *[str(k) for k in range(samples + 1)]], dist)
        _mixed_table(rep, "category",
                     [*[(_short_cat(r.category), r.right_per_task) for r in results],
                      ("total", [x for r in results for x in r.right_per_task])], samples)

    rep.section(WALL_CLOCK)
    if any(r.seconds is None for r in results):
        if wall:
            for line in wall:
                rep.print(line)
        else:
            rep.note("(not recorded: the replies file has no timings and there is no")
            rep.note("earlier capture to carry them over from)")
        return
    tsec = sum(r.seconds for r in results)
    titems = sum(len(r.right_per_task) for r in results)
    rep.table(["category", "tasks", "s/task", "new tokens/task"],
              [[_short_cat(r.category), len(r.right_per_task),
                f"{r.seconds / max(1, len(r.right_per_task)):.2f}",
                f"{r.new_tokens / max(1, len(r.right_per_task)):.0f}"] for r in results]
              + [["total", titems, f"{tsec / max(1, titems):.2f}",
                  f"{sum(r.new_tokens for r in results) / max(1, titems):.0f}"]])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_bfcl_score(args) -> None:
    _quiet_hf()
    from . import bfcl
    from .device import dtype_name, get_device, preferred_dtype, set_determinism

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in cats if c not in bfcl.CATEGORIES]
    if unknown:
        sys.exit(f"unknown categories {unknown}; choose from {list(bfcl.CATEGORIES)}")
    _check_sampling(args.samples, args.temperature)
    sampling = args.samples > 1
    mode = f"samples{args.samples}-t{args.temperature:g}" if sampling else "greedy"
    tag = _bfcl_tag(args.model, args.split, mode, args.limit)
    device = get_device()
    determinism = set_determinism(device)
    dtype = dtype_name(preferred_dtype(device))
    batch = _batch_size(args, device)
    from .engine import ENGINE

    rep = Report(f"tinytools bfcl-score ({mode})", out=OUTPUTS / f"bfcl-{tag}.txt")
    _bfcl_setup(rep, model=args.model, split=args.split, limit=args.limit, sampling=sampling,
                samples=args.samples, temperature=args.temperature, seed=args.seed,
                dtype=dtype, determinism=determinism, engine=(ENGINE, batch))

    progress.status(f"loading {args.model} on {device}")
    tok, model = bfcl.load_model(args.model, device)
    results, records = [], []
    for cat in cats:
        items = bfcl.load_items(cat, args.limit, args.split)
        on_item = progress.items(f"{cat}, {mode}", len(items))
        results.append(bfcl.score(tok, model, items, samples=args.samples,
                                  temperature=args.temperature, seed=args.seed,
                                  batch_size=batch, on_item=on_item, records=records,
                                  setup={"dtype": dtype, "device": device.type,
                                         "engine": ENGINE, "batch_size": batch}))
    _bfcl_body(rep, results, sampling=sampling, samples=args.samples, wall=None)
    rep.close()
    items_path = OUTPUTS / f"bfcl-{tag}-replies.jsonl"
    _write_jsonl(items_path, records)
    progress.status(f"replies written to {items_path.relative_to(ROOT)}")


def recorded_setup(rows: list[dict], earlier: str | None) -> tuple[str, str]:
    """(device type, dtype) a replies file was generated with: from its rows, else from
    the header of the capture it came with (files written before rows carried them).

    The dtype is the one :func:`device.preferred_dtype` picks: float32 on mps and
    cpu; a CUDA capture without the field can't say (bf16 or not), so that raises.
    """
    dev = {r.get("device") for r in rows}
    dt = {r.get("dtype") for r in rows}
    if None not in dev and None not in dt:
        if len(dev) != 1 or len(dt) != 1:
            raise ValueError(f"mixed setups in one replies file: {dev}, {dt}")
        return dev.pop(), dt.pop()
    m = re.search(r"^device {8}: (\w+)", earlier or "", re.MULTILINE)
    if m is None:
        raise ValueError("the replies don't record their device and dtype, and there is no "
                         "earlier capture whose header says the device")
    if m.group(1) == "cuda":
        raise ValueError("a CUDA capture without a recorded dtype (bf16 or fp32?)")
    return m.group(1), "float32"


def recorded_engine(rows: list[dict]) -> tuple[str, int] | None:
    """(engine, batch size) from a replies file's rows; ``None``: the one-at-a-time
    engine (rows written before the batched engine carry no such fields)."""
    from .engine import recorded

    return recorded(rows)


def cmd_bfcl_rescore(args) -> None:
    """Re-judge a ``bfcl-score`` replies file (no model); write ``bfcl-score``'s capture."""
    _quiet_hf()
    from . import bfcl
    from .device import determinism_label

    path = args.replies.resolve()
    name = parse_bfcl_replies_name(path.name)
    if name is None:
        sys.exit(f"{path.name}: not a bfcl-score replies file name "
                 "(bfcl-<model>-[dev-|eval-]<mode>-limit<N>-replies.jsonl)")
    if name["model"] != args.model.split("/")[-1]:
        sys.exit(f"{path.name} was scored with {name['model']}; pass --model for it")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    if not rows:
        sys.exit(f"{path}: no replies")
    sampling, samples = name["samples"] > 1, name["samples"]
    source = name["split"]
    if args.split is not None and args.split != source:
        if source != "all":
            sys.exit(f"{path.name} holds the {source} split; --split {args.split} filters "
                     "an all-split file")
        name["split"] = args.split
    tag = _bfcl_tag(args.model, name["split"], name["mode"], name["limit"])
    out = OUTPUTS / f"bfcl-{tag}.txt"
    # filtering writes a new replies file under the split's name; the source stays as it was
    replies_out = path if name["split"] == source else OUTPUTS / f"bfcl-{tag}-replies.jsonl"
    earlier = out.read_text(encoding="utf-8") if out.exists() else None
    try:
        device_type, dtype = recorded_setup(rows, earlier)
        engine = recorded_engine(rows)
    except ValueError as e:
        sys.exit(f"{path.name}: {e}")

    by_cat: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], {}).setdefault(r["id"], []).append(r)
    results, ordered = [], []
    for cat, group in by_cat.items():
        items = bfcl.load_items(cat, name["limit"], name["split"])
        if name["split"] != source:  # keep the split's items (all minus the dev slice)
            keep = set(bfcl.in_split(list(group), bfcl.dev_slice(cat), name["split"]))
            group = {i: rs for i, rs in group.items() if i in keep}
        if [it.id for it in items] != list(group):
            sys.exit(f"{path.name}: the {cat} replies are not the first "
                     f"{name['limit'] or 'all'} items of split {name['split']}, in order")
        if any(len(rs) != samples for rs in group.values()):
            sys.exit(f"{path.name}: expected {samples} replies per {cat} item")
        reps = [sorted(rs, key=lambda r: r.get("sample", 0)) for rs in group.values()]
        for rs in reps:
            for i, r in enumerate(rs):
                r.setdefault("sample", i)
                r.update(dtype=dtype, device=device_type)
                if engine is not None:
                    r.update(engine=engine[0], batch_size=engine[1])
        ordered += [r for rs in reps for r in rs]
        results.append(bfcl.rejudge(items, reps, progress.items(f"rescore, {cat}", len(items))))
    wall = section_lines(earlier, WALL_CLOCK) if earlier else None

    rep = Report(f"tinytools bfcl-score ({name['mode']})", out=out)
    _bfcl_setup(rep, model=args.model, split=name["split"], limit=name["limit"],
                sampling=sampling, samples=samples, temperature=name["temperature"],
                seed=args.seed, dtype=dtype, determinism=determinism_label(device_type),
                engine=engine)
    rep.kv("checked by", "bfcl-rescore of the saved replies")
    profile = getattr(args, "profile", "main")  # main when args are built by hand
    tok_id = _profile_tokenizer(profile, args.model)
    if tok_id is not None:
        rep.kv("chat template and tokenizer", tok_id)
        rep.note(f"(--profile {profile}: the template the replies were scored with)")
    if name["split"] != source:
        rep.note(f"(the {name['split']} items of a --split {source} replies file)")
    _bfcl_body(rep, results, sampling=sampling, samples=samples, wall=wall)
    rep.close()
    _write_jsonl(replies_out, ordered)
    if replies_out == path:
        progress.status(f"verdicts in {path.relative_to(ROOT)} updated (replies unchanged)")
    else:
        progress.status(f"{name['split']} replies written to {replies_out.relative_to(ROOT)} "
                        f"({path.name} unchanged)")


# -- our tool tasks --------------------------------------------------------------

TASKS_DIR = DATA / "tasks"


def _block(rep: Report, text: str, indent: int = 4) -> None:
    """Print text inside the report, long lines broken with a trailing backslash."""
    from .ours import wrap_lines

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
    rep.note(f"seed {args.seed}; {tasks.SAMPLE_SPLIT} is a seeded sample of train (post 10's")
    rep.note("ready-for-RL measurement), the same 60/20/20 mix.")

    rep.section("Tasks by tool")
    rows = []
    for k in tasks.KINDS:
        for tool in tasks.KIND_TOOLS[k]:
            rows.append([tool, tasks.SHORT_KIND[k],
                         *[sum(t["kind"] == k and t["tool"] == tool for t in splits[s])
                           for s in names]])
    rep.table(["tool needed", "kind", *names], rows)
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


def cmd_tasks_onetool(args) -> None:
    """r2a: post 10's one-tool kinds by sub-kind and split, the operation tools, the
    near-miss rule, the shortcut controls (D, D_R, D_near), the phrasing check, the
    entity overlap with both train splits, post 9 unchanged."""
    from . import onetool_tasks as ot
    from . import shortcuts as sc
    from . import tasks, tools
    from .bfclfuncs import count_pct

    post9 = tasks.generate(args.seed)
    splits = ot.generate(args.seed, post9=post9)
    directory = TASKS_DIR if args.seed == tasks.DEFAULT_SEED else TASKS_DIR / f"seed{args.seed}"
    paths = ot.write(splits, directory)
    suffix = "" if args.seed == tasks.DEFAULT_SEED else f"-seed{args.seed}"
    rep = Report("tinytools tasks --onetool", out=OUTPUTS / f"tasks-onetool{suffix}.txt")
    rep.header()
    names = list(splits)
    test = splits["test"]

    rep.section("One tool offered: tasks by kind and sub-kind")
    per = {s: ot.counts(splits[s]) for s in names}
    sub = {s: ot.subkind_counts(splits[s]) for s in names}
    rows = []
    for kind in ot.KINDS:
        rows.append([kind, *[sum(n for (k, _), n in sub[s].items() if k == kind)
                             for s in names]])
        for sk in ot.SUBKINDS[kind]:
            rows.append([f"  {sk}", *[sub[s][(kind, sk)] for s in names]])
            parts = [st for st in ot.STRATA if st.subkind == sk and st.name != sk]
            for st in parts:
                label = st.phrasing if sk == ot.PLAIN else f"{st.job_class} job"
                rows.append([f"    {label}", *[per[s][st.name] for s in names]])
    rows.append(["total", *[len(splits[s]) for s in names]])
    rows.append(["reading-required", *[sum(t["reading_required"] for t in splits[s])
                                       for s in names]])
    rep.table(["kind / sub-kind", *names], rows)
    rep.note(f"seed {args.seed}. {ot.SAMPLE_SPLIT} is a seeded sample of train, dealt")
    rep.note("over each stratum's jobs as the splits are.")
    rep.note("fits: right = execution match; doesn't fit: right = declined.")
    rep.note("reading-required: a doesn't-fit item whose job is a tool that runs")
    rep.note("(not the one offered); only reading the tool tells it from a fits")
    rep.note("item of the same job.")

    rep.section("The tools: post 9's five, and twelve that run now")
    by_domain: dict[str, list[str]] = {}
    for name in tools.OPERATION_NAMES:
        by_domain.setdefault(ot.JOBS[name].domain, []).append(name)
    rep.note("operation tools (their own registry; post 9 never offers them):")
    for d, ns in by_domain.items():
        rep.kv(f"  {d}", ", ".join(ns), width=12, wrap=True)
    rep.kv("post 9's tools that run", len(tools.TOOL_NAMES))
    rep.kv("distractors (schema only)", len(tools.DISTRACTOR_NAMES))
    rep.kv("tools that run now", len(tools.EXECUTABLE_NAMES))
    schema_sha = hashlib.sha256(ot.post9_schema_bytes()).hexdigest()
    rep.kv("post 9's schemas byte-identical",
           f"{'yes' if schema_sha == ot.POST9_SCHEMAS_SHA256 else 'NO'} ({schema_sha[:16]})")
    in_post9 = sum(n in t["tools"] for s in tasks.SPLITS for t in post9[s]
                   for n in tools.OPERATION_NAMES)
    rep.kv("post 9 tasks offering a new tool", in_post9)
    rep.note("checked by execution: state reset before every call; calendar,")
    rep.note("contacts and documents compared whole after the call; math by")
    rep.note("the result's canonical polynomial. delete_calendar_event stays a")
    rep.note("distractor (it takes an event id).")

    rep.section("Near miss: which pairs count")
    rep.note(f"dropped: {ot.STEP_RULE}.", wrap=True)
    rows = [[o, job] for o, job in ot.dropped_near_pairs()]
    rep.table(["offered", "needs (pair dropped)"], rows)
    rep.note(f"{len(rows)} pairs dropped. Kept: cancel or delete offered for a change")
    rep.note("(one call can't do the edit, so declining is right).")
    short = ot.near_shortfall()
    rep.kv("families with no near-miss pair", ", ".join(short) if short else "none",
           wrap=True)
    rep.blank()
    rows = []
    for fam in ot.FAMILIES:
        def fn(s, fam=fam):
            f = sum(t["tool"] == fam and t["subkind"] == ot.VERB for t in splits[s])
            n = sum(t["tool"] == fam and t["subkind"] == ot.NEAR for t in splits[s])
            return f"{f}/{n}"
        rows.append([fam, len(ot.near_offers(fam)), *[fn(s) for s in names]])
    rep.table(["family (fits / near miss)", "pairs", *names], rows)
    worst = max(abs(sum(t["tool"] == f and t["subkind"] == ot.VERB for t in splits[s])
                    - sum(t["tool"] == f and t["subkind"] == ot.NEAR for t in splits[s]))
                for f in ot.FAMILIES for s in names)
    rep.note("fits: operation verb (the family's own tool offered); near miss:")
    rep.note("another tool of the domain offered. pairs: tools that can be offered.")
    rep.kv("largest fits - near-miss gap", f"{worst} (any family, any split)")

    rep.section("Cross-domain: what the request needs")
    rows = [[job, ot.JOBS[job].domain, *[sum(t["tool"] == job and t["subkind"] == ot.CROSS
                                             for t in splits[s]) for s in names]]
            for job in (*ot.EXECUTABLE_JOBS, *ot.DISTRACTOR_JOBS)]
    rep.table(["needs", "domain", *names], rows)
    rep.note("the first 17 run (reading-required); the last 5 are distractors.")
    dev = splits["dev"]
    dev_jobs = {t["tool"] for t in dev}
    dev_cross_domains = {t["job_domain"] for t in dev if t["subkind"] == ot.CROSS}
    rep.kv("dev: every distractor job",
           all(any(t["tool"] == j and t["subkind"] == ot.CROSS for t in dev)
               for j in ot.DISTRACTOR_JOBS))
    rep.kv("dev: every domain as cross-domain",
           all(ot.JOBS[j].domain in dev_cross_domains for j in ot.EXECUTABLE_JOBS))
    rep.kv("dev: every operation family", all(f in dev_jobs for f in ot.FAMILIES))

    rep.section("Examples from test")
    by_id = sorted(test, key=lambda t: t["id"])
    for st in ot.STRATA:
        t = next(x for x in by_id if ot.stratum_of(x) == st.name)
        rep.note(f"{st.name}:")
        rep.note(f"  offered {t['tools'][0]}, needs {t['tool']}:")
        rep.note(f"    {t['request']}", wrap=True)

    rep.section("Shortcut controls: rules that never read the operation")
    rows = []
    for group in (sc.RULES, sc.DESCRIPTIVE, sc.CHECKS):
        for label, rule in group.items():
            x = sc.rule_scores(test, rule)
            rows.append([label, f"{100 * x.whole.c_plus:.1f}", f"{100 * x.whole.c_minus:.1f}",
                         f"{100 * x.whole.d:+.1f}", f"{100 * x.reading.d:+.1f}",
                         f"{100 * x.near.d:+.1f}"])
    rep.table(["rule (test set)", "c+", "c-", "D", "D_R", "D_near"], rows)
    x = sc.rule_scores(test, sc.rule_a_calls)
    rep.note("controls (the first three): Rule A declines on a delete, change,")
    rep.note("share or symbolic word; job identity declines unless the job is a")
    rep.note("tool that runs; domain identity calls iff the tool offered is in")
    rep.note("the request's domain (it reads the tool's app, not its operation).")
    rep.note("tool-name overlap (descriptive): calls iff a word of the tool's")
    rep.note("name, or its plural, is in the request. call iff tricky (a check):")
    rep.note("calls iff the values are written the tricky way.")
    rep.note("c+, c-, D: call rate (%) where a call is / isn't needed, whole")
    rep.note("test; D = c+ - c-.")
    rep.note(f"D_R: on R, {x.reading.n_minus} reading-required items and the "
             f"{x.reading.n_plus} fits items", wrap=True)
    rep.note(f"of the same {len(x.reading.jobs)} jobs; means over the jobs.")
    rep.note(f"D_near: operation-verb fits vs near miss (same domain), "
             f"{x.near.n_plus} and {x.near.n_minus}", wrap=True)
    rep.note(f"items; means over the {len(x.near.jobs)} operation families.")
    r = sc.reading_subset(test)
    rows = []
    for j in sc.reading_jobs(r):
        if j not in ot.TRICKY_TOOLS:
            continue
        side = {k: [t for t in r if t["tool"] == j and t["kind"] == k] for k in ot.KINDS}
        rows.append([j, *[count_pct(sum(t["phrasing"] == ot.TRICKY for t in side[k]),
                                    len(side[k])) for k in ot.KINDS]])
    rep.blank()
    rep.table(["R: tricky values, by job", "fits", "doesn't fit"], rows)
    rep.note("cross-domain items of these jobs are dealt 3 : 1 explicit : tricky,")
    rep.note("as plain fits are; near misses and operation-verb fits are explicit.")

    rep.section("Entity overlap with post 9's train split and one-tool train")
    p9_train = {ot.dated_record_entity(t) for t in post9["train"]}
    one_train = {ot.dated_record_entity(t) for t in splits["train"]}
    rows = []
    for st in [*ot.STRATA, None]:
        mine = [t for t in test if st is None or ot.stratum_of(t) == st.name]
        hits = [sum(ot.dated_record_entity(t) in keys for t in mine)
                for keys in (p9_train, one_train)]
        rows.append(["all" if st is None else st.name,
                     *[count_pct(h, len(mine)) for h in hits]])
    rep.table(["test: stratum", "post 9 train", "one-tool train"], rows)
    rep.note("an item whose job and slot values (contact, title, numbers, a date")
    rep.note("counted once however written) occur in a train item: post 9's, or")
    rep.note("one-tool's (RL trains on both). Distractor jobs are held out from")
    rep.note("all of post 9's splits; cancel counts post 9's delete-event items.")
    agree = ot.answer_agreement(test, splits["train"])
    rep.kv("right answer in one-tool train", "same {}, both {}, opposite {}".format(
        agree[ot.SAME_ANSWER], agree[ot.BOTH_ANSWERS], agree[ot.OPPOSITE_ANSWER]))
    rep.note("per test item above: its one-tool train items all have its right")
    rep.note("answer (call or decline), some each way, or all the opposite one.")
    keys = {s: {ot.dated_record_key(t) for t in splits[s]} for s in ("dev", "train")}
    rep.kv("test items whose task is in", "dev {}, train {}".format(
        *[sum(ot.dated_record_key(t) in keys[s] for t in test) for s in ("dev", "train")]))
    rep.kv("dev items whose task is in", "train {}".format(
        sum(ot.dated_record_key(t) in keys["train"] for t in splits["dev"])))
    rep.note("a task: the job, its values (a date counted once however written)")
    rep.note("and the tool offered. Dev and train skip a task an earlier split")
    rep.note("has only if the date is written the same way, so these count the")
    rep.note("calendar tasks that recur with the date written another way.")
    rep.note(f"Calendar dates repeat across splits: the pool has only "
             f"{len(tasks._event_days())} event days.", wrap=True)

    rep.section("Pools: big enough for test + dev + train, all disjoint")
    pool_rows = sorted(ot.pool_capacity(args.seed, post9=post9),
                       key=lambda r: (r.headroom, r.job, r.phrasing))
    rep.table(["pool (the job; phrasing)", "candidates", "drawn", "ratio"],
              [[f"{r.job}{'; tricky' if r.phrasing == ot.TRICKY else ''}", r.candidates,
                r.drawn, f"{r.headroom:.2f}"] for r in pool_rows])
    rep.note("candidates: distinct requests, post 9's request texts (and, for")
    rep.note("distractor jobs, entities) removed; drawn: test + dev + train.")
    rep.kv("every pool covers its draw", all(r.candidates >= r.drawn for r in pool_rows))
    main = list(ot.SPLITS)

    def keyset(recs, by):
        return {t["request"] if by == "text" else (t["template"], json.dumps(t["slots"],
                                                                                sort_keys=True))
                for t in recs}
    pairs = [(a, b) for i, a in enumerate(main) for b in main[i + 1:]]
    rows = [[f"{a} / {b}", len(keyset(splits[a], "text") & keyset(splits[b], "text")),
             len(keyset(splits[a], "slots") & keyset(splits[b], "slots"))] for a, b in pairs]
    p9_text = ot.post9_texts(post9)
    rows += [[f"{s} / post 9 (any)", len(keyset(splits[s], "text") & p9_text), "-"]
             for s in main]
    rep.blank()
    rep.table(["splits", "same text", "same template+slots"], rows)
    rep.note("post 9's template ids never occur here (ours start '1t-').")

    rep.section("The step mix: kinds per RL step (fixed weights)")
    main_mix, fallback = ot.mix_shares(ot.STEP_MIX), ot.mix_shares(ot.FALLBACK_MIX)
    rows = [[k, ot.STEP_MIX[k], f"{main_mix[k]:.1f}%", ot.FALLBACK_MIX[k], f"{fallback[k]:.1f}%"]
            for k in ot.STEP_MIX]
    rows.append(["total", sum(ot.STEP_MIX.values()), f"{sum(main_mix.values()):.1f}%",
                 sum(ot.FALLBACK_MIX.values()), f"{sum(fallback.values()):.1f}%"])
    rep.table(["kind", "weight", "share", "fallback", "share"], rows)
    rep.note("fallback: the gate's, if too few one-tool-fits groups are mixed.")
    rep.note("within one tool, fits: plain and operation verb 1 : 1 (the splits).")

    rep.section("Post 9's splits, unchanged")
    rows, same = [], 0
    for s, pinned in ot.POST9_SHA256.items():
        h = tasks.sha256(post9[s])
        f = TASKS_DIR / f"{s}.jsonl"
        on_disk = "missing" if not f.exists() else (
            "same" if hashlib.sha256(f.read_bytes()).hexdigest() == pinned else "differs")
        same += h == pinned
        rows.append([s, len(post9[s]), h[:16], h == pinned, on_disk])
    rep.table(["split", "tasks", "sha256 (first 16)", "pinned", "file"], rows)
    rep.kv("post 9's split hashes unchanged", f"{'yes' if same == len(rows) else 'NO'} "
           f"({same} of {len(rows)} match)")

    rep.section("Files (regenerated byte for byte from the seed)")
    rep.table(["split", "tasks", "sha256 (first 16)"],
              [[s, len(splits[s]), ot.sha256(splits[s])[:16]] for s in names])
    rep.note(f"files: {paths[ot.FILE_PREFIX + 'test'].parent.relative_to(ROOT)}/"
             f"{ot.FILE_PREFIX}<split>.jsonl")
    rep.close()


def cmd_bfcl_functions(args) -> None:
    """r2b's first part: functions offered per BFCL item, before and after the dev slice."""
    from . import bfclfuncs as bf
    from .bfcl import DEV_SEED, DEV_SHARE, LEADERBOARD_BFCL_EVAL, bfcl_version

    rep = Report("tinytools bfcl-functions", out=OUTPUTS / "bfcl-functions.txt")
    rep.header()
    counts = [bf.count(c) for c in bf.ORDER]

    rep.section("The data: bfcl-eval's own files")
    version = bfcl_version()
    rep.kv("bfcl-eval", version)
    rep.kv("the leaderboard's version", f"{LEADERBOARD_BFCL_EVAL} (same: "
           f"{'yes' if version == LEADERBOARD_BFCL_EVAL else 'NO'})")
    rep.note("sha256 of each file read, and whether it is the pinned one:")
    for c in counts:
        rep.note(f"  {c.path.name:<44} pinned {'yes' if c.pinned else 'NO'}")
        rep.note(f"    {c.sha256}")
    rep.kv("every file the pinned one", f"{'yes' if all(c.pinned for c in counts) else 'NO'} "
           f"({sum(c.pinned for c in counts)} of {len(counts)})")

    sizes = sorted({k for c in counts for k in c.every})
    heads = ["category", "items", *[f"{k} fn{'s' if k > 1 else ''}" for k in sizes]]
    for title, attr, total in [("Functions offered per item, every item in the file", "every", "n"),
                               ("After the dev slice (the items post 10 scores)", "after_dev",
                                "n_after")]:
        rep.section(title)
        rows = []
        for c in counts:
            got, n = getattr(c, attr), getattr(c, total)
            rows.append([c.category, n, *[bf.count_pct(got[k], n) if got[k] else "-"
                                          for k in sizes]])
        rep.table(heads, rows)
        if attr == "every":
            rep.note("(%): share of the category's items.")
        else:
            rep.note(f"dev slice: {DEV_SHARE}% of each category, seed {DEV_SEED}, held out of")
            rep.note("every published BFCL number.")

    rep.section("One-function prompts")
    by = {c.category: c for c in counts}
    rows = []
    for label, cats in [("simple + parallel", ("simple", "parallel")),
                        ("irrelevance", ("irrelevance",))]:
        one = sum(by[c].every[1] for c in cats)
        one_after = sum(by[c].after_dev[1] for c in cats)
        n = sum(by[c].n for c in cats)
        n_after = sum(by[c].n_after for c in cats)
        rows.append([label, bf.count_pct(one, n), sum(by[c].dev for c in cats),
                     bf.count_pct(one_after, n_after)])
    rep.table(["set", "1 function, all", "in dev", "1 function, after"], rows)
    rep.note("simple + parallel need a call; irrelevance needs none.")
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
    noun = {"search_documents": "documents", "search_contacts": "contacts",
            "get_calendar_event": "events"}[name]
    return (f"{noun} found: {len(c.reference_run.compared)} vs {len(c.reply_run.compared)} "
            f"({'the same' if same else 'different'})")


def _tagname(model: str, split: str, mode: str, limit) -> str:
    return f"{model.split('/')[-1]}-{split}-{mode}-limit{limit if limit is not None else 'all'}"


def _score_setup(rep: Report, *, model: str, split: str, splits: dict, limit, sampling: bool,
                 samples: int, temperature: float, seed: int, dtype: str,
                 determinism: str, engine: tuple[str, int] | None) -> None:
    from . import bfcl, tasks

    rep.header()
    rep.section("Setup")
    rep.kv("model", model)
    rep.kv("tasks", f"{split} split, seed {tasks.DEFAULT_SEED}")
    rep.kv("  sha256 (first 16)", tasks.sha256(splits[split])[:16])
    rep.kv("tasks scored", "all" if limit is None else f"first {limit} of each kind, by id")
    rep.kv("decoding", "greedy (argmax)" if not sampling else
           f"{samples} samples, temperature {temperature:g}")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("dtype", dtype)
    rep.kv("determinism", determinism, wrap=True)
    _seed_and_engine(rep, seed, engine, samples, "task")
    rep.kv("right means", "execution match (exact beside it)")


def cmd_score(args) -> None:
    _quiet_hf()
    from . import bfcl, ours, tasks
    from .device import dtype_name, get_device, preferred_dtype, set_determinism

    _check_sampling(args.samples, args.temperature)
    sampling = args.samples > 1
    mode = f"samples{args.samples}-t{args.temperature:g}" if sampling else "greedy"
    tag = _tagname(args.model, args.split, mode, args.limit)
    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    items = ours.limit_per_kind(splits[args.split], args.limit)

    device = get_device()
    determinism = set_determinism(device)
    dtype = dtype_name(preferred_dtype(device))
    batch = _batch_size(args, device)
    from .engine import ENGINE

    rep = Report(f"tinytools score ({args.split}, {mode})", out=OUTPUTS / f"score-{tag}.txt")
    _score_setup(rep, model=args.model, split=args.split, splits=splits, limit=args.limit,
                 sampling=sampling, samples=args.samples, temperature=args.temperature,
                 seed=args.seed, dtype=dtype, determinism=determinism, engine=(ENGINE, batch))

    progress.status(f"loading {args.model} on {device}")
    tok, model = bfcl.load_model(args.model, device)

    greedy = ours.score(tok, model, items, samples=1, temperature=0.0, seed=args.seed,
                        batch_size=batch,
                        on_item=progress.items(f"{args.split}, greedy", len(items)))
    if not sampling:
        _r1_from(rep, tok, splits, args.split, greedy)
    sampled = None
    if sampling:
        sampled = ours.score(tok, model, items, samples=args.samples,
                             temperature=args.temperature, seed=args.seed, batch_size=batch,
                             on_item=progress.items(f"{args.split}, {mode}", len(items)))
    _score_body(rep, greedy=greedy, sampled=sampled, mode=mode, samples=args.samples,
                split=args.split, wall=_wall_rows(greedy, sampled, mode))
    rep.close()
    path = OUTPUTS / f"score-{tag}-replies.jsonl"
    _write_replies(path, greedy, sampled, mode, {"dtype": dtype, "device": device.type,
                                                 "engine": ENGINE, "batch_size": batch})
    progress.status(f"replies written to {path.relative_to(ROOT)}")


def _wall_rows(greedy, sampled, mode: str) -> list[list] | None:
    """The wall-clock table's rows; ``None`` if any task's time or token count is unknown."""
    rows = []
    for label, res in [("greedy", greedy)] + ([(mode, sampled)] if sampled is not None else []):
        if any(s.seconds is None or s.new_tokens is None for s in res):
            return None
        secs = sum(s.seconds for s in res)
        rows.append([label, len(res), f"{secs / max(1, len(res)):.2f}",
                     f"{sum(s.new_tokens for s in res) / max(1, len(res)):.0f}",
                     sum(sum(s.capped) for s in res)])
    return rows


def _write_replies(path: Path, greedy, sampled, mode: str, setup: dict) -> None:
    """Every reply with its checks, one JSON line each; ``rescore`` reads this back."""
    with open(path, "w", encoding="utf-8") as f:
        for label, res in [("greedy", greedy)] + ([(mode, sampled)] if sampled is not None else []):
            for s in res:
                for i, (text, c, capped) in enumerate(zip(s.replies, s.checks, s.capped,
                                                          strict=True)):
                    row = {
                        "id": s.task["id"], "pass": label, "sample": i, "kind": s.task["kind"],
                        "tool": s.task["tool"], "request": s.task["request"], "reply": text,
                        "outcome": c.outcome, "execution": c.execution, "exact": c.exact,
                        "hit_cap": capped,
                        "result": c.reply_run.result if c.reply_run else None,
                        **setup,
                    }
                    if s.seconds is not None and s.new_tokens is not None:
                        row["task_seconds"] = s.seconds  # the task's whole group, all samples
                        row["task_new_tokens"] = s.new_tokens
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_body(rep: Report, *, greedy, sampled, mode: str, samples: int, split: str,
                wall: list[list] | list[str] | None) -> None:
    """Everything ``score`` prints after the r1 block, from scored replies (no model needed).

    ``wall`` is the wall-clock table's rows, or the lines of an earlier capture's
    wall-clock section (``rescore`` of a replies file with no timings), or ``None``.
    """
    from . import ours, tasks
    from .checkers import call_text, check_reply
    from .classify import CALL_OUTCOMES, NO_FIT_OUTCOMES, short_label

    sampling = sampled is not None
    main = sampled if sampling else greedy
    rows = ours.outcome_rows(main)
    capped = ours.capped_by_kind(main)
    calls = [r for r in rows if r.kind != tasks.NO_FIT and r.kind != "total"]
    no_fit = [r for r in rows if r.kind == tasks.NO_FIT]
    tail = " (all samples)" if sampling else ""
    if calls:
        rep.section("Tasks that need a call: every reply" + tail)
        total = ours.total_row(calls)
        rep.table(["kind", "n", *[short_label(o) for o in CALL_OUTCOMES]],
                  [[tasks.SHORT_KIND.get(r.kind, r.kind), r.n,
                    *[r.outcomes.get(o, 0) for o in CALL_OUTCOMES]] for r in [*calls, total]])
        rep.note("wrong fn: another tool; wrong n: the right tool, more than one")
        rep.note("call; bad args: the right tool, arguments that fail execution")
        rep.note("match; right: execution match.")
        _cap_table(rep, "kind", [(tasks.SHORT_KIND.get(r.kind, r.kind), *capped[r.kind])
                                 for r in calls])
    if no_fit:
        rep.section("Tasks where no tool fits: every reply" + tail)
        rep.table(["kind", "n", *[short_label(o) for o in NO_FIT_OUTCOMES]],
                  [[tasks.SHORT_KIND.get(r.kind, r.kind), r.n,
                    *[r.outcomes.get(o, 0) for o in NO_FIT_OUTCOMES]] for r in no_fit])
        rep.note("declined: no <tool_call> tag (right); called anyway: any tag.")
        _no_fit_cap_lines(rep, "kind", [(tasks.SHORT_KIND.get(r.kind, r.kind), r.outcomes,
                                         capped[r.kind][1]) for r in no_fit])
    rep.section("Right by execution and by exact match" + tail)
    rep.table(["kind", "right by execution", "right by exact match"],
              [[tasks.SHORT_KIND.get(r.kind, r.kind), ours.pct(r.execution, r.n),
                ours.pct(r.exact, r.n)] for r in rows])
    rep.note("The same replies, two checks. On no-tool-fits tasks both mean")
    rep.note("declined.")

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
        g = samples
        right = {k: [s.right for s in sampled if s.task["kind"] == k] for k in tasks.KINDS}
        allr = [s.right for s in sampled]
        rep.section(f"Groups of {g}: how many samples are right per task")
        rep.table(["kind", *[str(k) for k in range(g + 1)]],
                  [[tasks.SHORT_KIND[k], *ours.histogram(right[k], g)] for k in tasks.KINDS
                   if right[k]] + [["total", *ours.histogram(allr, g)]])
        _mixed_table(rep, "kind", [*[(tasks.SHORT_KIND[k], right[k]) for k in tasks.KINDS
                                     if right[k]], ("total", allr)], g)
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

    rep.section(WALL_CLOCK)
    if wall and isinstance(wall[0], str):
        for line in wall:
            rep.print(line)
    elif wall:
        rep.table(["pass", "tasks", "s/task", "new tokens/task", "hit token cap"], wall)
    else:
        rep.note("(not recorded: the replies file has no timings and there is no")
        rep.note("earlier capture to carry them over from)")


WALL_CLOCK = "Wall clock (varies run to run)"


def section_lines(text: str, title: str) -> list[str] | None:
    """The body of one section of a capture (the lines after its title's rule, up to the next
    section's rule or the end, trailing blanks dropped); ``None`` if the section is absent."""
    lines = text.split("\n")
    rule = "-" * 74
    for i in range(1, len(lines) - 1):
        if lines[i] == title and lines[i - 1] == rule and lines[i + 1] == rule:
            body = []
            for line in lines[i + 2:]:
                if line == rule or line == "=" * 74:
                    break
                body.append(line)
            while body and not body[-1].strip():
                body.pop()
            return body
    return None


def cmd_rescore(args) -> None:
    """Re-check a ``score`` replies file with the current checkers; write ``score``'s capture."""
    _quiet_hf()
    from . import ours, tasks
    from .checkers import check_reply

    path = args.replies.resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    if not rows:
        sys.exit(f"{path}: no replies")
    passes = list(dict.fromkeys(r["pass"] for r in rows))
    sampled_passes = [p for p in passes if p != "greedy"]
    if "greedy" not in passes or len(sampled_passes) > 1:
        sys.exit(f"{path}: expected a greedy pass and at most one sampled pass, got {passes}")
    mode = sampled_passes[0] if sampled_passes else "greedy"
    sampling = mode != "greedy"
    samples = 1 + max(r["sample"] for r in rows if r["pass"] == mode) if sampling else 1
    temperature = float(mode.rsplit("-t", 1)[1]) if sampling else 0.0

    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    by_id = {t["id"]: t for t in splits[args.split]}
    ids = list(dict.fromkeys(r["id"] for r in rows if r["pass"] == "greedy"))
    missing = [i for i in ids if i not in by_id]
    if missing:
        sys.exit(f"{path}: {len(missing)} task ids not in the {args.split} split (e.g. {missing[0]})")
    per_kind: dict[str, int] = {}
    for i in ids:
        per_kind[by_id[i]["kind"]] = per_kind.get(by_id[i]["kind"], 0) + 1
    limit = None
    if sorted(ids) != [t["id"] for t in ours.limit_per_kind(splits[args.split], None)]:
        limit = max(per_kind.values())
        if sorted(ids) != [t["id"] for t in ours.limit_per_kind(splits[args.split], limit)]:
            sys.exit(f"{path}: its tasks are not the first N of each kind of the {args.split} split")
    tag = _tagname(args.model, args.split, mode, limit)
    if path.name != f"score-{tag}-replies.jsonl":
        sys.exit(f"{path.name}: expected score-{tag}-replies.jsonl for --model {args.model} "
                 f"--split {args.split}; pass the model and split it was scored with")

    def rebuild(label: str) -> list:
        group: dict[str, list[dict]] = {}
        for r in rows:
            if r["pass"] == label:
                group.setdefault(r["id"], []).append(r)
        out = []
        on_item = progress.items(f"rescore, {label}", len(group))
        for k, (tid, rs) in enumerate(group.items(), start=1):
            rs.sort(key=lambda r: r["sample"])
            task = by_id[tid]
            out.append(ours.Scored(
                task, [r["reply"] for r in rs], [check_reply(r["reply"], task["reference"]) for r in rs],
                [r["hit_cap"] for r in rs], rs[0].get("task_new_tokens"), rs[0].get("task_seconds")))
            on_item(k)
        return out

    greedy = rebuild("greedy")
    sampled = rebuild(mode) if sampling else None
    wall = _wall_rows(greedy, sampled, mode)
    out = OUTPUTS / f"score-{tag}.txt"
    if wall is None and out.exists():  # timings live only in the capture these replies came from
        wall = section_lines(out.read_text(encoding="utf-8"), WALL_CLOCK)

    try:
        device_type, dtype = recorded_setup(rows, out.read_text(encoding="utf-8")
                                            if out.exists() else None)
        engine = recorded_engine(rows)
    except ValueError as e:
        sys.exit(f"{path.name}: {e}")
    from .device import determinism_label

    rep = Report(f"tinytools score ({args.split}, {mode})", out=out)
    _score_setup(rep, model=args.model, split=args.split, splits=splits, limit=limit,
                 sampling=sampling, samples=samples, temperature=temperature, seed=args.seed,
                 dtype=dtype, determinism=determinism_label(device_type), engine=engine)
    rep.kv("checked by", "rescore of the saved replies")
    profile = getattr(args, "profile", "main")  # main when args are built by hand
    tok_id = _profile_tokenizer(profile, args.model)
    if tok_id is not None:
        rep.kv("chat template and tokenizer", tok_id)
        rep.note(f"(--profile {profile}: the template the replies were scored with)")
    if not sampling:
        from .train import load_tokenizer

        _r1_from(rep, load_tokenizer(args.model, tok_id), splits, args.split, greedy,
                 missing="(not in this replies file: run score to print it)",
                 prompt_path=_r1_path(profile, args.model))
    _score_body(rep, greedy=greedy, sampled=sampled, mode=mode, samples=samples,
                split=args.split, wall=wall)
    rep.close()
    setup = {"dtype": dtype, "device": device_type}
    if engine is not None:
        setup.update(engine=engine[0], batch_size=engine[1])
    _write_replies(path, greedy, sampled, mode, setup)
    progress.status(f"verdicts in {path.relative_to(ROOT)} updated (replies unchanged)")


def _r1_from(rep: Report, tok, splits: dict, split: str, greedy,
             missing: str = "(not among the tasks scored)",
             prompt_path: Path | None = None) -> None:
    """The r1 block from the greedy pass's own reply to r1's task (``score`` and
    ``rescore`` alike), so the block and the replies file can't disagree."""
    from . import ours

    task = ours.r1_task(splits["test"])
    raw = next((s.replies[0] for s in greedy if s.task["id"] == task["id"]), None)
    if split == "test" and raw is not None:
        _r1(rep, tok, task, raw, prompt_path)
    else:
        rep.section(f"One task, whole: {task['id']} ({task['kind']}, {task['tool']})")
        rep.note(missing)


def _r1(rep: Report, tok, task: dict, raw: str, prompt_path: Path | None = None) -> None:
    """r1a and r1b: one test task's whole prompt, and the model's reply run through the tool."""
    from . import ours, tasks
    from .checkers import call_text, check_reply

    prompt_path = prompt_path or OUTPUTS / "r1-prompt.txt"
    text = ours.prompt(tok, task)
    census = ours.prompt_census(tok, text)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(text)
    rep.section(f"One task, whole: {task['id']} ({task['kind']}, {task['tool']})")
    rep.note("The templated prompt, exactly as the model reads it (Qwen2.5's")
    rep.note("chat template; long lines broken with a trailing \\; unbroken copy")
    rep.note(f"in {prompt_path.relative_to(ROOT)}):")
    rep.blank()
    _block(rep, text)
    rep.blank()
    rep.kv("prompt tokens", census.tokens)
    rep.kv("  the tool list (<tools> block)", census.tool_list_tokens)
    rep.kv("  the whole tools section", census.tools_section_tokens)
    rep.kv("  tokens straddling those edges", census.straddling)

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


# -- prompt-view: r1's prompt, readable -----------------------------------------------


def _prompt_view_path(task_id: str, r1_id: str) -> Path:
    return OUTPUTS / ("r1-prompt-view.txt" if task_id == r1_id
                      else f"prompt-view-{task_id}.txt")


def _wrapped_note(rep: Report, text: str) -> None:
    from .promptview import wrap

    for line in wrap(text, WIDTH - 2):
        rep.note(line)


def cmd_prompt_view(args) -> None:
    """A readable layout of one task's templated prompt: its parts, their tokens, its tool."""
    _quiet_hf()
    from . import ours, promptview, tasks
    from .train import MODEL, load_tokenizer

    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    test = splits["test"]
    r1_id = ours.r1_task(test)["id"]
    task_id = args.task or r1_id
    task = next((t for t in test if t["id"] == task_id), None)
    if task is None:
        sys.exit(f"{task_id}: not a test task")
    tok = load_tokenizer(args.model)
    text = ours.prompt(tok, task)
    exact: Path | None = OUTPUTS / "r1-prompt.txt"
    if task_id == r1_id and args.model == MODEL:  # the view must be of that very text
        if not exact.exists() or exact.read_text() != text:
            sys.exit(f"{exact.relative_to(ROOT)} is missing or differs from the prompt "
                     f"rendered now; rerun rescore on the main run's replies first")
    else:
        exact = None

    segs = promptview.split_prompt(text)  # raises unless the parts join back to `text`
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    counts = promptview.part_tokens(text, offsets)  # raises unless the rows sum to the total
    census = ours.prompt_census(tok, text)  # r1a's numbers: the view must agree with them
    if ((counts.total, counts.tool_list, counts.parts["tools"] + counts.parts["format"],
         counts.across_list_end + counts.across_parts)
            != (census.tokens, census.tool_list_tokens, census.tools_section_tokens,
                census.straddling)):
        sys.exit("the part counts disagree with r1a's census")
    found = promptview.tools_in(text)
    names = [t["function"]["name"] for t in found]
    if task["tool"] not in names:
        sys.exit(f"{task['tool']} is not in the prompt's tool list")
    la, lb = promptview.tool_list_span(text)

    rep = Report(f"tinytools prompt-view ({task_id})", out=_prompt_view_path(task_id, r1_id))
    # No hardware block: nothing here runs a model, so the capture is the same anywhere.
    rep.print("=" * WIDTH)
    rep.print(rep.title)
    rep.print("=" * WIDTH)
    where = exact.relative_to(ROOT) if exact else "the score capture's r1 block"
    _wrapped_note(rep, f"A readable layout of the exact prompt, which is in {where}.")
    rep.blank()
    rep.kv("task", f"{task_id} ({task['kind']}, {task['tool']})", wrap=True)
    rep.kv("tokenizer and chat template", args.model)
    if exact is not None:
        rep.kv(f"rendered now = {exact.name}", True)
    rep.kv("parts joined back = the prompt", True)

    ind = " " * 6
    rep.section("The prompt, part by part")
    _wrapped_note(rep, "Each part under a label. [system], [user] and [assistant] stand in "
                       "for the turn markers (<|im_start|> and the role; <|im_end|>); "
                       "[tools] and [format] are the template's text inside the system "
                       "turn. Long lines are wrapped at spaces.")
    for s in segs:
        if s.kind != "part":
            continue
        rep.blank()
        rep.note(f"[{s.name}]")
        if s.name == "tools":
            for line in promptview.wrap(text[s.start:la].rstrip("\n"), WIDTH - len(ind)):
                rep.print((ind + line).rstrip())
            lines = text[la:lb].split("\n")
            rep.print(ind + lines[0])
            for k, name in enumerate(names, start=1):
                tail = " (in full below)" if name == task["tool"] else ""
                rep.print(f"{ind}  line {k} of {len(names)}: the JSON of {name}{tail}")
            rep.print(ind + lines[-1])
        elif s.text == "":
            rep.print(f"{ind}(empty: the model writes its reply from here)")
        else:
            for line in promptview.wrap(s.text.rstrip("\n"), WIDTH - len(ind)):
                rep.print((ind + line).rstrip())

    rep.section("The prompt's tokens, part by part")
    rows: list[tuple[str, object]] = [("turn markers", counts.markers)]
    for name in promptview.PARTS:
        rows.append((f"[{name}]", counts.parts[name]))
        if name == "tools":
            rows += [("  the preamble before the list", counts.preamble),
                     ("  the tool list itself", counts.tool_list),
                     ("  a token across the list's end", counts.across_list_end),
                     ("  the blank line after the list", counts.after_list)]
    rows.append(("tokens across a part's edge", counts.across_parts))
    rows.append(("prompt tokens (indented rows aside)", counts.rows_sum()))
    rep.table(["part", "tokens"], rows)
    rep.blank()
    across = [text[a:b] for a, b in offsets if a < lb < b]
    joined = (f" The token across the list's end is {across[0]!r}: the closing '>' of "
              "</tools> joined to the blank line after it." if across else "")
    _wrapped_note(rep, "Qwen2.5's tokenizer on the whole prompt; each token is counted in the "
                       "part it lies in. A part runs up to the next marker or part, so the "
                       "newlines the template puts between parts count with the part before "
                       "them. Turn markers: <|im_start|>, the role and its newline; "
                       "<|im_end|> and its newline. The indented rows split [tools]."
                       + joined)
    rep.blank()
    _wrapped_note(rep, f"In r1's terms: the tool list is the {census.tool_list_tokens}; "
                       f"[tools] + [format] = {census.tools_section_tokens}, the whole tools "
                       f"section; the token across the list's end is r1's "
                       f"{census.straddling} straddling token.")

    tool = found[names.index(task["tool"])]
    fn = tool["function"]
    rep.section(f"The task's tool, laid out: {fn['name']}")
    rep.kv("name", fn["name"], width=12)
    rep.kv("description", fn.get("description", ""), width=12, wrap=True)
    rep.blank()
    for line in promptview.param_table(promptview.params(tool), WIDTH - 2):
        rep.note(line)
    rep.blank()
    src = exact.relative_to(ROOT) if exact else "the prompt"
    _wrapped_note(rep, "Parsed from the tool's JSON inside the prompt. The model reads it as "
                       f"one line of JSON, exactly as in {src}.")
    rep.close()


# -- the SFT training sets -----------------------------------------------------------

SFT_DIR = DATA / "sft"


def _of(n: int, total: int) -> str:
    from .ours import pct

    return pct(n, total)


def parse_seeds(text: str) -> tuple[int, ...]:
    """``"1,2,3"`` -> ``(1, 2, 3)``: distinct, in the order given."""
    seeds: list[int] = []
    for part in text.split(","):
        if part.strip():
            n = int(part)
            if n not in seeds:
                seeds.append(n)
    if not seeds:
        raise ValueError("no seeds given")
    return tuple(seeds)


def cmd_sft_data(args) -> None:
    _quiet_hf()
    from transformers import AutoTokenizer

    from . import sft

    seeds = parse_seeds(args.seeds)
    src = ROOT / sft.CONVERTED
    if not src.exists():
        sys.exit(f"{sft.CONVERTED} is missing; run `tinytools toolace-convert` first")
    raw = src.read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()
    converted = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    variant = getattr(args, "variant", None)  # None when args are built by hand
    if variant is not None:
        return _sft_data_variant(args, variant, converted, src_sha, tok)
    on_item = progress.items(f"truncating, drawing random calls for seeds {args.seeds}, rendering",
                             len(converted))
    s = sft.build_sets(converted, seeds, tok, on_item)

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, records in [("real", s.real),
                          *[(f"control-seed{n}", s.controls[n].records) for n in seeds]]:
        data = sft.jsonl_bytes(records)
        (SFT_DIR / f"{name}.jsonl").write_bytes(data)
        files[name] = (len(records), hashlib.sha256(data).hexdigest())

    rep = Report("tinytools sft-data", out=OUTPUTS / "sft-data.txt")
    rep.header()
    cols = [f"seed {n}" for n in seeds]
    ctl = [s.controls[n] for n in seeds]

    rep.section("Input")
    rep.kv("converted ToolACE", sft.CONVERTED)
    rep.kv("  sha256 (first 16)", src_sha[:16])
    rep.kv("  is toolace-convert's output", src_sha == sft.CONVERTED_SHA256)
    rep.kv("rows", len(converted))
    rep.kv("tokenizer (chat template)", args.tokenizer)
    rep.kv("control seeds", f"{', '.join(map(str, seeds))} (one set per training seed)")
    rep.note("Each row's draws: Random(\"seed:row id\"). The real arm has no")
    rep.note("randomness.")

    rep.section("Which rows train")
    rep.note("Rows with 2+ tools and a call, and every row with no call (any")
    rep.note("tool list). No-call rows go into both arms unchanged; rows with")
    rep.note("one tool and a call are left out.")
    rep.table(["row type", "converted", "dropped", "each arm"],
              [[t, s.rows_converted[t], s.rows_dropped[t] if t != sft.R_OUT else "-",
                s.rows_trained[t]] for t in sft.ROW_TYPES]
              + [["total", sum(s.rows_converted.values()), sum(s.rows_dropped.values()),
                  sum(s.rows_trained.values())]])
    no_call = [t for t in sft.ROW_TYPES if t not in (sft.R_CALL, sft.R_OUT)]
    trains = [t for t in sft.ROW_TYPES if t != sft.R_OUT]
    rep.blank()  # ends the table above (a reader takes rows until a line splits differently)
    rep.table(["subtotal", "converted", "dropped", "each arm"], [
        [label, sum(s.rows_converted[t] for t in ts), sum(s.rows_dropped[t] for t in ts),
         sum(s.rows_trained[t] for t in ts)]
        for label, ts in (("no call (the three rows above)", no_call),
                          ("all but the left-out rows", trains))])
    rep.note("dropped: from both arms, by reason (checked after truncation):")
    rep.table(["reason", "rows"], [[r, len(s.dropped[r])] for r in sft.DROP_REASONS]
              + [["total", sum(len(v) for v in s.dropped.values())]])
    for r in sft.DROP_REASONS:
        if s.dropped[r]:
            rep.note(f"{r}:")
            for line in textwrap.wrap(", ".join(s.dropped[r]), WIDTH - 4):
                rep.print("    " + line)
    rep.note("An empty turn would teach the model to end its turn at once; a")
    rep.note("[Name(...)] call written in a text turn would train as plain text.")

    rep.section("Truncation: each row cut after its first call turn")
    rep.note("Both arms. Not-calling assistant turns before the first call stay;")
    rep.note("a row with no call is kept whole. Turns after the cut (tool")
    rep.note("results, answers, later requests) are removed.")
    rep.table(["", "all converted", "training rows"], [
        ["rows", s.cut_all.rows, s.cut_train.rows],
        ["rows cut", s.cut_all.rows_cut, s.cut_train.rows_cut],
        ["turns removed", s.cut_all.turns_removed, s.cut_train.turns_removed],
        ["  of them assistant turns", s.cut_all.assistant_turns_removed,
         s.cut_train.assistant_turns_removed],
        ["tokens removed", s.cut_all.tokens_removed, s.cut_train.tokens_removed],
        ["  of them scored", s.cut_all.trained_removed, s.cut_train.trained_removed],
    ])

    rep.section("Turns in each arm, beside the full set")
    shapes = [s.shape_converted, s.shape_truncated, s.shape_real, ctl[0].shape]
    fields = [
        ("rows", "rows"),
        ("assistant turns", "assistant_turns"),
        ("  call turns", "call_turns"),
        ("    parallel (2+ calls)", "parallel_turns"),
        ("  not-calls after user", "not_calls_after_user"),
        ("  not-calls after tool", "not_calls_after_tool"),
        ("calls", "calls"),
        ("tool results", "tool_results"),
    ]
    rep.table(["", "converted", "truncated", "real arm", "control"],
              [[label, *[getattr(c, f) for c in shapes]] for label, f in fields])
    rep.note(f"converted: all {s.shape_converted.rows} rows as toolace-convert wrote them;")
    rep.note("truncated: the same rows cut; real arm: the training rows;")
    rep.note("control: their random-call twins, the same counts for every seed")
    rep.note("(checked). Not-calls after user: asking for an argument, saying")
    rep.note("no function fits, chatting.")

    rep.section("Tokens, real vs control (the chat template's rendering)")
    trows = [["tokens", s.shape_real.tokens, *[c.shape.tokens for c in ctl]],
             ["tokens scored", s.shape_real.trained, *[c.shape.trained for c in ctl]],
             ["control / real, all", "", *[f"{c.shape.tokens / s.shape_real.tokens:.4f}"
                                           for c in ctl]],
             ["control / real, scored", "", *[f"{c.shape.trained / s.shape_real.trained:.4f}"
                                              for c in ctl]]]
    rep.table(["", "real arm", *cols], trows)
    rep.note("scored: the tokens the loss is computed on: each assistant turn")
    rep.note("from after its '<|im_start|>assistant' header through its")
    rep.note("<|im_end|>. The header is masked. The no-call rows are the same")
    rep.note("in both arms, so only call turns can move the ratio.")

    rep.section("Control: does the drawn tool match the call it replaced?")
    rep.note("Each call's tool is drawn uniformly from the row's k tools, so it")
    rep.note("matches the replaced call's tool with chance 1/k.")
    cs0 = ctl[0].stats
    rep.kv("control calls (every seed)", cs0.calls)
    rep.kv("expected to draw it (sum of 1/k)", f"{cs0.expected:.1f} (sd {cs0.variance ** 0.5:.1f})")
    rep.blank()
    rep.table(["control set", "drew the replaced call's tool"],
              [[col, _of(c.stats.hits, c.stats.calls)] for col, c in zip(cols, ctl, strict=True)])
    rep.blank()
    rep.table(["tools offered (k)", "rows", "calls", "expected", *cols],
              [[k, cs0.rows_by_k[k], cs0.calls_by_k[k], f"{cs0.calls_by_k[k] / k:.1f}",
                *[c.stats.hits_by_k[k] for c in ctl]] for k in sorted(cs0.rows_by_k)])
    rep.note("rows: training rows with a call (no-call rows have no draws).")

    rep.section("Control: where each call's arguments came from")
    rep.note("another row: a call to the drawn tool in another row that fits")
    rep.note("  this row's schema; other schema: the same, schema differs;")
    rep.note("built: no other row calls the tool (even when it is the replaced")
    rep.note("  call's tool), so its required parameters are filled one by one.")
    rep.note("No argument ever comes from the row itself.")
    src_rows = [[name, *[c.stats.sources(name) for c in ctl]] for name in sft.SOURCES]
    src_rows.append(["  built, drew the replaced tool", *[c.stats.source_hit[sft.BUILT]
                                                           for c in ctl]])
    src_rows.append(["total", *[c.stats.calls for c in ctl]])
    src_rows.append(["taken from the row itself", *[c.stats.from_own_row for c in ctl]])
    src_rows.append(["equal to the real call", *[c.stats.identical for c in ctl]])
    src_rows += [[f"  {name}", *[c.stats.identical_by_source[name] for c in ctl]]
                 for name in sft.SOURCES]
    src_rows.append(["  any source, tool takes no args",
                     *[c.stats.identical_no_args for c in ctl]])
    rep.table(["control calls", *cols], src_rows)
    rep.note("equal to the real call: the drawn tool is the right one and the")
    rep.note("arguments copied from other rows happen to equal this row's")
    rep.note("(coincidence only: another row asked the same, or a built call's")
    rep.note("copied values match). A tool that takes no arguments has only")
    rep.note("one possible call, so drawing it reproduces the real call.")
    rep.blank()
    rep.table(["built calls' parameters, from", *cols],
              [[p, *[c.stats.param_sources[p] for c in ctl]] for p in sft.PARAM_SOURCES]
              + [["empty values (placeholders)", *[c.stats.empty_built for c in ctl]]])
    rep.note("Tried in that order; the last copies a value of the declared")
    rep.note("type from any argument in another row (seeded choice).")
    rep.blank()
    rep.table(["mean chars change", *cols],
              [[name, *[f"{c.stats.mean_chars_change(name):+.1f}" for c in ctl]]
               for name in sft.SOURCES]
              + [["all calls", *[f"{c.stats.mean_chars_change():+.1f}" for c in ctl]]])
    rep.note("The arguments' JSON length against the call replaced (built")
    rep.note("calls leave optional parameters out).")

    _r4b(rep, s, tok, sft, seeds[0])

    rep.section("Files (regenerated byte for byte from the input and seeds)")
    rep.table(["file", "records", "sha256 (first 16)"],
              [[str((SFT_DIR / f"{n}.jsonl").relative_to(ROOT)), k, h[:16]]
               for n, (k, h) in files.items()])
    rep.close()


def _sft_data_variant(args, variant: str, converted: list[dict], src_sha: str, tok) -> None:
    """``sft-data --variant NAME``: the real set with a variant's row types left out
    (sft.VARIANTS), into data/sft/real-NAME.jsonl; the four sets are not touched. Rows,
    truncation, drops and rendering are the full set's (build_sets with ``leave_out``);
    checked against real.jsonl when it is here."""
    from . import sft

    leave = sft.VARIANTS[variant]
    on_item = progress.items(f"truncating, rendering (variant {variant})", len(converted))
    s = sft.build_sets(converted, (), tok, on_item, leave_out=leave)
    name = sft.variant_file(variant)
    data = sft.jsonl_bytes(s.real)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    (SFT_DIR / name).write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()

    rep = Report(f"tinytools sft-data --variant {variant}",
                 out=OUTPUTS / f"sft-data-{variant}.txt")
    rep.header()
    rep.section("Input")
    rep.kv("converted ToolACE", sft.CONVERTED)
    rep.kv("  sha256 (first 16)", src_sha[:16])
    rep.kv("  is toolace-convert's output", src_sha == sft.CONVERTED_SHA256)
    rep.kv("rows", len(converted))
    rep.kv("tokenizer (chat template)", args.tokenizer)
    rep.kv("variant", variant)
    rep.kv("  left out", ", ".join(leave))
    rep.note("The real arm's set with those training rows left out. Truncation,")
    rep.note("drops and rendering are the full set's; no control set is made.")

    note = sft.VARIANT_NOTE[variant]
    label = {t: (f"{t} ({note})" if t in leave else t) for t in sft.ROW_TYPES}
    rep.section("Which rows train")
    rep.table(["row type", "converted", "dropped", "trains"],
              [[label[t], s.rows_converted[t], s.rows_dropped[t] if t != sft.R_OUT else "-",
                s.rows_trained[t]] for t in sft.ROW_TYPES]
              + [["total", sum(s.rows_converted.values()), sum(s.rows_dropped.values()),
                  sum(s.rows_trained.values())]])
    full = sum(s.rows_trained.values()) + sum(s.rows_left_out.values())
    rep.blank()
    rep.kv("training rows, the full real set", full)
    rep.kv(f"  left out ({', '.join(leave)})", sum(s.rows_left_out.values()))
    rep.kv("  this variant", sum(s.rows_trained.values()))
    if sft.R_NO_CALL_1 in leave:
        rep.note("With the one-tool rows that call already out (the control can't")
        rep.note("randomise one tool), no training row offers a single tool.")
    rep.note("dropped: as in the full set, checked before rows are left out:")
    rep.table(["reason", "rows"], [[r, len(s.dropped[r])] for r in sft.DROP_REASONS]
              + [["total", sum(len(v) for v in s.dropped.values())]])

    rep.section("Turns and tokens, this variant")
    sh = s.shape_real
    rep.table(["", "this variant"], [
        ["rows", sh.rows], ["assistant turns", sh.assistant_turns],
        ["  call turns", sh.call_turns], ["    parallel (2+ calls)", sh.parallel_turns],
        ["  not-calls after user", sh.not_calls_after_user],
        ["  not-calls after tool", sh.not_calls_after_tool], ["calls", sh.calls],
        ["tool results", sh.tool_results], ["tokens", sh.tokens], ["tokens scored", sh.trained]])

    rep.section("Against data/sft/real.jsonl")
    real = SFT_DIR / "real.jsonl"
    if real.exists():
        raw_real = real.read_bytes()
        real_sha = hashlib.sha256(raw_real).hexdigest()
        cut, removed = sft.without_rows(raw_real, leave)
        rep.kv("real.jsonl as pinned (sft.py)", real_sha == sft.SET_SHA256["real.jsonl"])
        rep.kv("  its rows of the left-out types", removed)
        rep.kv("this file = real.jsonl less them", cut == data)
        rep.note("(byte for byte: every kept line of real.jsonl, in its order)")
    else:
        rep.note("real.jsonl is not here (sft-data --seeds makes it); not compared.")

    rep.section("File (regenerated byte for byte from the input)")
    rep.table(["file", "records", "sha256 (first 16)"],
              [[str((SFT_DIR / name).relative_to(ROOT)), len(s.real), sha[:16]]])
    rep.kv("as pinned (sft.py)", sha == sft.VARIANT_SHA256.get(name))
    rep.close()


def _r4b(rep: Report, s, tok, sft, seed: int) -> None:
    real = next(r for r in s.real if r["id"] == sft.EXAMPLE_ID)
    twin = next(r for r in s.controls[seed].records if r["id"] == sft.EXAMPLE_ID)
    rep.section(f"One training row and its control twin ({sft.EXAMPLE_ID}, seed {seed})")
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
    rep.note("The system turn holds the date line and the tool list. scored:")
    rep.note("the tokens the loss is computed on. In an assistant turn, the")
    rep.note("'<|im_start|>assistant' header line and the newline after")
    rep.note("<|im_end|> are masked; the rest, through <|im_end|>, is scored.")
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


# -- figures -------------------------------------------------------------------------


def cmd_figures(args) -> None:
    from .figures import make_figures

    make_figures(args.only, score=args.score, bfcl=args.bfcl, out=args.out)


# -- Milestone 0 -------------------------------------------------------------------------


def cmd_milestone0(args) -> None:
    _quiet_hf()
    from . import milestone0

    try:
        models = milestone0.parse_models(args.models) if args.models else None
    except ValueError as e:
        sys.exit(str(e))
    if args.batch_size is not None and args.batch_size < 1:
        sys.exit("--batch-size must be at least 1")
    code = milestone0.run(args.quick, models, ROOT, base_when_needed=args.base_when_needed,
                          deadline=args.deadline_epoch, gen_batch=args.batch_size)
    if code:
        progress.status("scoring stopped early: " + ("SIGTERM" if code > 128 else "the deadline,"
                                                     " before the rule could decide"))
        sys.exit(code)


def cmd_engine_check(args) -> None:
    _quiet_hf()
    from . import engine_check

    if args.batch_size is not None and args.batch_size < 1:
        sys.exit("--batch-size must be at least 1")
    for n in (args.ours, args.bfcl, args.sampled):
        if n is not None and n < 1:
            sys.exit("--ours, --bfcl and --sampled must be at least 1")
    engine_check.run(args.quick, args.batch_size, ROOT,
                     engine_check.sizes_for(args.quick, args.ours, args.bfcl, args.sampled))


# -- post 9's SFT runs (sftrun.py) ----------------------------------------------------


def _deadline(args) -> float | None:
    return getattr(args, "deadline_epoch", None)


def cmd_sft_check(args) -> None:
    from . import sftrun

    sys.exit(sftrun.run_check(ROOT, args.quick, args.model, args.dev_limit, args.batch_size,
                              args.profile))


def cmd_sft_score(args) -> None:
    from . import sftrun

    if args.untrained == (args.adapter is not None):
        sys.exit("give one of --untrained and --adapter")
    if args.noise_floor and not args.untrained:
        sys.exit("--noise-floor is the untrained model's (give --untrained)")
    sys.exit(sftrun.run_score(ROOT, args.quick, args.adapter.resolve() if args.adapter else None,
                              args.batch_size, floor=args.noise_floor, profile=args.profile))


def cmd_sft_train(args) -> None:
    from . import sftrun

    if args.repeat and (args.arm, args.seed) != ("real", 1):
        sys.exit("--repeat is seed 1 real (the training determinism check)")
    sys.exit(sftrun.run_train(ROOT, args.quick, args.arm, args.seed, args.repeat, _deadline(args),
                              args.token_budget, args.grad_checkpointing, args.profile))


def cmd_sft_pair(args) -> None:
    from . import sftrun

    sys.exit(sftrun.run_pair(ROOT, args.quick, args.seed, _deadline(args), args.batch_size,
                             args.token_budget, args.grad_checkpointing, args.profile))


def cmd_sft_report(args) -> None:
    from . import sftrun

    source = Path(args.source) if args.source else None
    main_from = getattr(args, "main_from", None)
    if main_from is not None and args.profile != "onetool":
        sys.exit("--main-from is the onetool report's (--profile onetool)")
    main_from = (ROOT / main_from) if main_from is not None else None
    sys.exit(sftrun.run_report(ROOT, args.quick, args.profile, args.reading, source, main_from))


def cmd_sft_followup(args) -> None:
    from . import followup

    sys.exit(followup.run(ROOT, ROOT / args.source))


# -- entry point -------------------------------------------------------------------

BATCH_HELP = ("sequences per forward pass (an item's samples stay in one batch); default "
              "by device: cuda 64, mps 8, cpu 4")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    args.fn(args)
    progress.status(f"done in {time.perf_counter() - t0:.0f} s")


def build_parser() -> argparse.ArgumentParser:
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

    s = sub.add_parser("bfcl-score", help="score a model on BFCL, split by outcome")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--categories", default="simple,multiple,parallel,parallel_multiple,irrelevance")
    s.add_argument("--split", choices=["eval", "dev", "all"], default="eval",
                   help="eval: all items minus the dev slice (published numbers); dev: the "
                        "slice (Milestone 0); all: both (prints a warning)")
    s.add_argument("--limit", type=int, default=None,
                   help="first N items of the split per category (default: all)")
    s.add_argument("--samples", type=int, default=1, help="1 = greedy; >1 = sampled group")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    s.set_defaults(fn=cmd_bfcl_score)

    s = sub.add_parser("bfcl-rescore", help="re-judge a bfcl-score replies file (no model)")
    s.add_argument("--replies", type=Path, required=True,
                   help="a bfcl-...-replies.jsonl; its verdict fields are rewritten")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="the model it was scored with")
    s.add_argument("--seed", type=int, default=0, help="the seed it was scored with (printed only)")
    s.add_argument("--split", choices=["eval"], default=None,
                   help="eval: keep only the eval items of an all-split file and write the "
                        "capture bfcl-score --split eval would (default: the file's own split)")
    s.add_argument("--profile", choices=sorted(PROFILE_NAMES), default="main",
                   help="base: the replies of post 9's base arm; Setup names the Instruct "
                        "chat template its prompts had")
    s.set_defaults(fn=cmd_bfcl_rescore)

    s = sub.add_parser("tasks", help="generate our tool tasks (test, dev, train) and count them")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--onetool", action="store_true",
                   help="post 10's one-tool kinds instead (data/tasks/onetool-*.jsonl, "
                        "outputs/tasks-onetool.txt); post 9's files are not rewritten")
    s.set_defaults(fn=lambda a: (cmd_tasks_onetool if a.onetool else cmd_tasks)(a))

    sub.add_parser("bfcl-functions", help="functions offered per BFCL item, before and after "
                   "the dev slice").set_defaults(fn=cmd_bfcl_functions)

    sub.add_parser("check-demo", help="three calls checked by exact and execution match"
                   ).set_defaults(fn=cmd_check_demo)

    s = sub.add_parser("score", help="score a model on our tool tasks, split by outcome")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--split", choices=["test", "dev", "train-sample"], default="test")
    s.add_argument("--limit", type=int, default=None, help="first N tasks of each kind, by id")
    s.add_argument("--samples", type=int, default=1, help="1 = greedy; >1 = sampled group")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("rescore", help="re-check a score replies file with the current checkers")
    s.add_argument("--replies", type=Path, required=True,
                   help="a score-...-replies.jsonl; its verdict fields are rewritten")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="the model it was scored with")
    s.add_argument("--split", choices=["test", "dev", "train-sample"], default="test")
    s.add_argument("--seed", type=int, default=0, help="the seed it was scored with (printed only)")
    s.add_argument("--profile", choices=sorted(PROFILE_NAMES), default="main",
                   help="base: the replies of post 9's base arm (sft-score --profile base); "
                        "r1 uses the Instruct chat template the arm ran with and writes "
                        "outputs/base/r1-prompt.txt")
    s.set_defaults(fn=cmd_rescore)

    s = sub.add_parser("prompt-view", help="a readable layout of r1's prompt: parts, tokens, "
                       "its tool (no model)")
    s.add_argument("--task", default=None, help="a test task id (default: r1's, test-0013)")
    s.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="whose tokenizer and chat template (default: the post's model)")
    s.set_defaults(fn=cmd_prompt_view)

    s = sub.add_parser("sft-data", help="the SFT sets: real (truncated), random-call per seed")
    s.add_argument("--seeds", default="1,2,3",
                   help="comma-separated: one control set per training seed")
    s.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="whose chat template to render with")
    s.add_argument("--variant", choices=["onetool"], default=None,
                   help="onetool: only the real set without its 'no call, 1 tool' rows, "
                        "into data/sft/real-onetool.jsonl, capture sft-data-onetool.txt; "
                        "the four sets are not touched and --seeds is not used")
    s.set_defaults(fn=cmd_sft_data)

    s = sub.add_parser("figures", help="render the post's figures from the captures")
    s.add_argument("--only", default=None, help="one figure by name (default: all)")
    s.add_argument("--score", type=Path, default=None,
                   help="greedy score capture (default: the whole-test-split one)")
    s.add_argument("--bfcl", type=Path, default=None,
                   help="greedy bfcl-score capture (default: the one under outputs/)")
    s.add_argument("--out", type=Path, default=None, help="directory (default: figures/)")
    s.set_defaults(fn=cmd_figures)

    s = sub.add_parser("milestone0", help="score six models on the dev split; the rule's verdict")
    s.add_argument("--quick", action="store_true",
                   help="0.5B only, 1 task per kind and category, a tiny timing probe (Mac check)")
    s.add_argument("--models", default=None,
                   help="comma-separated subset, e.g. 1.5B-instruct,3B-instruct (default: all six; "
                        "with --quick: 0.5B-instruct,0.5B-base)")
    s.add_argument("--base-when-needed", action="store_true",
                   help="score a base model only if the rule reaches it (no instruct "
                        "candidate qualifies), and 0.5B base not at all; base replies run to "
                        "the token cap, so this is most of the run's time")
    s.add_argument("--deadline-epoch", type=float, default=None,
                   help="Unix time by which scoring must end (the Lambda runner sets "
                        "$TINYTOOLS_DEADLINE_EPOCH to when its time limit fires): a model that "
                        "can't finish by then is not started, and the rule is printed from "
                        "what was measured")
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    s.set_defaults(fn=cmd_milestone0)

    s = sub.add_parser("engine-check", help="the batched engine vs the one-at-a-time engine")
    s.add_argument("--quick", action="store_true", help="a few items of each kind (a smoke test)")
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    s.add_argument("--ours", type=int, default=None,
                   help="greedy: first N test tasks per kind (default 20; --quick 2)")
    s.add_argument("--bfcl", type=int, default=None,
                   help="greedy: first N BFCL eval items per category (default 8; --quick 1)")
    s.add_argument("--sampled", type=int, default=None,
                   help="sampled: the first N test tasks by id (default 10; --quick 3)")
    s.set_defaults(fn=cmd_engine_check)

    quick_help = ("Qwen2.5-0.5B-Instruct (--profile base: Qwen2.5-0.5B), the first 32 SFT rows, "
                  "a few tasks; outputs/main-quick/ (base-quick/, onetool-quick/) (a Mac check; "
                  "its numbers are not results)")
    deadline_help = ("Unix time by which the work must end (the Lambda runner's "
                     "$TINYTOOLS_DEADLINE_EPOCH): work that can't finish by then is not started")

    def profile_flag(s) -> None:
        s.add_argument("--profile", choices=sorted(PROFILE_NAMES), default="main",
                       help="main: Qwen2.5-1.5B-Instruct, outputs/main/ (as the main run ran); "
                            "base: Qwen2.5-1.5B with the Instruct chat template and four "
                            "embedding rows trained, outputs/base/, captures *-base.txt; "
                            "onetool: main's settings, the real arm only, on "
                            "data/sft/real-onetool.jsonl, outputs/onetool/, captures "
                            "*-onetool.txt")

    def training_flags(s) -> None:
        s.add_argument("--token-budget", type=int, default=None,
                       help="micro-batch budget, rows x longest row (default 8192; --quick 4096)")
        s.add_argument("--grad-checkpointing", action="store_true",
                       help="recompute activations in the backward pass (off by default)")

    s = sub.add_parser("sft-check", help="template, set sha256, lengths, memory, merge checks")
    s.add_argument("--quick", action="store_true", help=quick_help)
    s.add_argument("--model", default=None,
                   help="the model for the merge check (default: 1.5B instruct; --quick 0.5B)")
    s.add_argument("--dev-limit", type=int, default=None,
                   help="merge check on the first N dev tasks of each kind (default: all 200; "
                        "--quick 2; 120 takes all 200)")
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    profile_flag(s)
    s.set_defaults(fn=cmd_sft_check)

    s = sub.add_parser("sft-score", help="score the untrained model or a merged adapter")
    s.add_argument("--quick", action="store_true", help=quick_help)
    s.add_argument("--untrained", action="store_true", help="the base model")
    s.add_argument("--adapter", type=Path, default=None,
                   help="outputs/main/<label>/adapter.safetensors (scores into its directory)")
    s.add_argument("--noise-floor", action="store_true",
                   help="with --untrained: only the train-sample tasks' 8 at T=1, generation "
                        "seed 1, into untrained-gen-seed1/ (r6a's noise floor)")
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    profile_flag(s)
    s.set_defaults(fn=cmd_sft_score)

    s = sub.add_parser("sft-train", help="train one LoRA adapter")
    s.add_argument("--quick", action="store_true", help=quick_help)
    s.add_argument("--arm", choices=["real", "control"], required=True)
    s.add_argument("--seed", type=int, required=True,
                   help="LoRA init and shuffle order; control N trains control-seedN.jsonl")
    s.add_argument("--repeat", action="store_true",
                   help="seed 1 real again, into real-seed1-repeat/, compared with the first")
    s.add_argument("--deadline-epoch", type=float, default=None, help=deadline_help)
    training_flags(s)
    profile_flag(s)
    s.set_defaults(fn=cmd_sft_train)

    s = sub.add_parser("sft-pair", help="one seed: train real, score, train control, score")
    s.add_argument("--quick", action="store_true", help=quick_help)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--deadline-epoch", type=float, default=None, help=deadline_help)
    s.add_argument("--batch-size", type=int, default=None, help=BATCH_HELP)
    training_flags(s)
    profile_flag(s)
    s.set_defaults(fn=cmd_sft_pair)

    s = sub.add_parser("sft-report", help="r4c, r4d, r5a-d, r6a from saved runs (no model)")
    s.add_argument("--quick", action="store_true", help=quick_help)
    profile_flag(s)
    s.add_argument("--reading", choices=("strict", "tagfree"), default="strict",
                   help="strict (default): calls only inside <tool_call> tags, sft*.txt; "
                        "tagfree: bare call JSON at a reply's start counts too, "
                        "r5a-d and r6a beside a strict vs tag-free table, sft-tagfree*.txt")
    s.add_argument("--from", dest="source", default=None,
                   help="the arm tree to read (e.g. runs/2026-11-23-main/outputs/main; "
                        "read-only); default outputs/<profile>[-quick]/")
    s.add_argument("--main-from", dest="main_from", default=None,
                   help="--profile onetool: the main run's arm tree its real arm is compared "
                        "with (read-only; default runs/2026-11-23-main/outputs/main, --quick "
                        "outputs/main-quick)")
    s.set_defaults(fn=cmd_sft_report)

    s = sub.add_parser("sft-followup", help="section 5's follow-ups: ToolACE's no-call wording, "
                       "BFCL no-call replies, pooled right (no model)")
    s.add_argument("--from", dest="source", default="runs/2026-11-23-main/outputs/main",
                   help="the arm tree to read (read-only; default: the main run's)")
    s.set_defaults(fn=cmd_sft_followup)
    return p


if __name__ == "__main__":
    main()
