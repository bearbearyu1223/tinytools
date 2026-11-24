"""sft-followup: three follow-up counts for post 9's section 5, from saved files (no model).

1. **ToolACE's no-call rows, by wording** (:func:`opening_kind`). Each of the real
   set's rows with no call (``data/sft/real.jsonl``, pinned by sha256) is classed by
   its first assistant turn, first match wins:

   * ``TEMPLATE``: the turn opens with "The question" / "The given question" /
     "The query" / "The given query", then "lacks" or "does not provide";
   * ``LACKS``: it contains "lacks the required" anywhere (the SFT replies' wording);
   * ``ASKS``: it asks outright: "please", "could you", "can you" or "would you",
     then provide / specify / share / tell / confirm / give / let me know;
   * ``NO_FIT``: it says no function fits (a fixed phrase list, :data:`NO_FIT_RE`);
   * ``OTHER``: none of these.

   The same classes are applied to the trained models' BFCL no-call replies, so the
   two can be compared. A keyword rule, not a reading: the classes are named by
   their wording, not by intent.

2. **BFCL items where a trained model makes no call.** Only replies in the form
   ``lacks the required "p" parameter`` (or ``"p" and "q" parameters``) are
   measured, because only there the asked-for parameters can be read off exactly
   (:func:`asked_parameters`). Each asked parameter is classed against the item's
   schema and BFCL's possible answer (:func:`parameter_kind`), and each reply by
   its first class in :data:`REPLY_ORDER`.

3. **Right on our tasks that need a call, pooled** (seed means), with the paired
   interval, so the post can quote one share per arm.

Pure helpers are unit-tested in ``tests/test_followup.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# -- 1: wording ------------------------------------------------------------------------

TEMPLATE = "template"
LACKS = "lacks the required"
ASKS = "asks outright"
NO_FIT = "no function fits"
OTHER = "other"
KINDS = (TEMPLATE, LACKS, ASKS, NO_FIT, OTHER)

TEMPLATE_RE = re.compile(r"The (?:given )?(?:question|query) (?:lacks|does not provide)\b")
LACKS_PHRASE = "lacks the required"
ASKS_RE = re.compile(r"\b(?:please|could you|can you|would you)(?: please)? "
                     r"(?:provide|specify|share|tell|confirm|give|let me know)\b", re.I)
NO_FIT_RE = re.compile(
    r"none of the|no (?:available |suitable |relevant |such )?functions?\b"
    r"|functions? (?:available|provided)? ?(?:do|does|can) not|do(?:es)? not (?:support|include)"
    r"|cannot be used|can't be used|not (?:covered|supported|available)", re.I)


def opening_kind(text: str) -> str:
    """The wording class of a no-call turn (first match in :data:`KINDS`). Pure."""
    t = text.strip()
    if TEMPLATE_RE.match(t):
        return TEMPLATE
    if LACKS_PHRASE in t:
        return LACKS
    if ASKS_RE.search(t):
        return ASKS
    if NO_FIT_RE.search(t):
        return NO_FIT
    return OTHER


def first_assistant_text(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "assistant":
            return m.get("content") or ""
    raise ValueError("a row with no assistant turn")


def has_call(messages: list[dict]) -> bool:
    return any(m["role"] == "assistant" and m.get("tool_calls") for m in messages)


def text_turns(messages: list[dict]) -> int:
    """Assistant turns that make no call (r4a-turns' "not-calls after user", once
    truncated rows have no tool results)."""
    return sum(m["role"] == "assistant" and not m.get("tool_calls") for m in messages)


# -- 2: asked parameters ---------------------------------------------------------------

ASKED_RE = re.compile(r'lacks the required ((?:"[^"\n]+"(?:, | and |, and )?)+) parameters?\b')

IN_QUESTION = "in the question"
OPTIONAL = "optional"
OMITTABLE = "may be omitted"
NOT_FOUND = "not found"
NOT_IN_SCHEMA = "not in the schema"
REPLY_ORDER = (IN_QUESTION, OPTIONAL, OMITTABLE, NOT_FOUND, NOT_IN_SCHEMA)


def asked_parameters(reply: str) -> list[str] | None:
    """The parameter names in the first ``lacks the required "p" parameter`` phrase,
    or None if the reply has no such phrase. Pure."""
    m = ASKED_RE.search(reply)
    if not m:
        return None
    return re.findall(r'"([^"\n]+)"', m.group(1))


def value_in_question(options: list, question: str) -> bool:
    """Whether one of BFCL's allowed values for a parameter appears in the question.

    Strings: case-insensitive, as a whole word or words, allowing a plural ending
    ("house price" in "house prices"), so "en" is not found in "sentiment" nor "GOOG"
    in "Google" (empty strings, BFCL's "may be omitted", skipped). Numbers: as written
    or, if whole, as an integer, not inside a longer number. Booleans, lists and
    dicts: never (not checkable by text). Pure.
    """
    q = question.lower()
    for v in options:
        if isinstance(v, bool) or isinstance(v, (list, dict)) or v is None:
            continue
        if isinstance(v, (int, float)):
            forms = {str(v)} | ({str(int(v))} if float(v).is_integer() else set())
            if any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\.?\d)", question)
                   for f in forms):
                return True
        elif isinstance(v, str) and v.strip() and re.search(
                r"(?<![a-z0-9])" + re.escape(v.lower()) + r"(?:e?s)?(?![a-z0-9])", q):
            return True
    return False


def parameter_kind(param: str, functions: list[dict], ground_truth: list[dict],
                   question: str) -> str:
    """One asked parameter's class, against every offered function that has it. Pure.

    ``functions``: BFCL's function docs; ``ground_truth``: BFCL's possible answer
    (a list of ``{name: {param: [allowed values]}}``).
    """
    owners = [f for f in functions if param in f["parameters"].get("properties", {})]
    if not owners:
        return NOT_IN_SCHEMA
    if not any(param in f["parameters"].get("required", []) for f in owners):
        return OPTIONAL
    names = {f["name"] for f in owners}
    answers = [call[n][param] for call in ground_truth for n in call
               if n in names and param in call[n]]
    if any(value_in_question(a, question) for a in answers):
        return IN_QUESTION
    if answers and all("" in a for a in answers):
        return OMITTABLE
    return NOT_FOUND


def reply_kind(kinds: list[str]) -> str:
    """A reply's class: the first of :data:`REPLY_ORDER` among its parameters'. Pure."""
    for k in REPLY_ORDER:
        if k in kinds:
            return k
    raise ValueError("a reply with no asked parameter")


# -- the command -----------------------------------------------------------------------

DEFAULT_SOURCE = "runs/2026-11-23-main/outputs/main"
EXAMPLE_FIRST = "simple_python_6"  # the post's example, printed first
EXAMPLES_PER_KIND = 2


def run(root: Path, source: Path) -> int:
    from . import bfcl, sftrun, tasks
    from . import effects as E
    from .classify import NO_CALL, is_right
    from .cli import _block, _load_or_make_tasks, _quiet_hf
    from .ours import pct
    from .report import Report

    _quiet_hf()
    if not (source / sftrun.UNTRAINED / "score.json").exists():
        sys.exit(f"{source} has no scored untrained model")
    seeds, _ = sftrun.complete_seeds(source)
    real_path = root / "data" / "sft" / "real.jsonl"
    pins = sftrun.set_pins(real_path.parent)
    have, want = pins["real.jsonl"]
    if have != want:
        sys.exit(f"{real_path} is not the pinned real set; run sft-data")
    real = [json.loads(x) for x in real_path.read_text(encoding="utf-8").split("\n") if x]

    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    by_id = {t["id"]: t for t in splits["test"] + splits["train-sample"]}
    items = {cat: bfcl.load_items(cat, None, "eval") for cat in bfcl.CATEGORIES}
    un = sftrun.load_scores(source / sftrun.UNTRAINED, by_id, items)
    sft = [sftrun.load_scores(source / sftrun.run_label("real", s), by_id, items) for s in seeds]
    ctl = [sftrun.load_scores(source / sftrun.run_label("control", s), by_id, items)
           for s in seeds]
    replies = {sc.label: {r["id"]: r["reply"] for r in sftrun._rows(
        source / sc.label / sftrun.BFCL_FILE)} for sc in [un, *sft, *ctl]}

    rep = Report("tinytools sft-followup", out=root / "outputs" / "sft-followup.txt")
    rep.header()
    rep.section("Setup")
    rep.kv("model", un.info.get("model", "?"))
    rep.kv("replies from", sftrun._shown(source, root))
    rep.kv("seeds", ", ".join(str(s) for s in seeds))
    rep.kv("real set", f"data/sft/real.jsonl, {len(real)} rows")
    rep.kv("  sha256 (first 16), as pinned", f"{have[:16]}, yes")
    rep.kv("reading", "strict (as sft-report)")
    rep.note("Outcomes re-checked here with the current checkers, as sft-report")
    rep.note("does, so every count below agrees with outputs/sft.txt.")

    # -- 1 ---------------------------------------------------------------------------
    rep.section("ToolACE's no-call rows, by wording")
    no_call = [r for r in real if not has_call(r["messages"])]
    call_rows = [r for r in real if has_call(r["messages"])]
    rep.kv("rows in each arm", len(real))
    rep.kv("  with no call", len(no_call))
    rep.kv("  with a call", len(call_rows))
    t_no, t_call = (sum(text_turns(r["messages"]) for r in rs) for rs in (no_call, call_rows))
    rep.kv("text turns in the no-call rows", t_no)
    asking_first = sum(text_turns(r["messages"]) > 0 for r in call_rows)
    rep.kv("text turns before a call", f"{t_call} (in {asking_first} rows)")
    rep.kv("  all text turns", f"{t_no + t_call}, as r4a-turns")
    rep.blank()
    rep.note("Each no-call row by its first assistant turn, first match wins:")
    rep.note("  1 template: the turn opens \"The [given] question|query\", then")
    rep.note("    \"lacks\" or \"does not provide\";")
    rep.note("  2 lacks the required: it contains that phrase (the SFT replies'")
    rep.note("    wording);")
    rep.note("  3 asks outright: please / could you / can you / would you, then")
    rep.note("    provide, specify, share, tell, confirm, give or let me know;")
    rep.note("  4 no function fits: a fixed phrase list (\"none of the\", \"no")
    rep.note("    functions\", \"do not support\", \"cannot be used\", ...);")
    rep.note("  5 other. A keyword rule: classes name the wording, not the intent.")
    rep.blank()

    bfcl_call = [i.id for c in sftrun.BFCL_CALL_CATS for i in items[c]]

    def nocall_texts(runs):
        return [replies[sc.label][i] for sc in runs for i in bfcl_call if sc.bfcl[i] == NO_CALL]

    cols = [("ToolACE", [first_assistant_text(r["messages"]) for r in no_call]),
            ("SFT", nocall_texts(sft)), ("control", nocall_texts(ctl)),
            ("untr", nocall_texts([un]))]
    counted = {name: [opening_kind(t) for t in texts] for name, texts in cols}
    rows = [[f"{i + 1} {k}", *[sum(x == k for x in counted[name]) for name, _ in cols]]
            for i, k in enumerate(KINDS)]
    rows.append(["total", *[len(texts) for _, texts in cols]])
    rep.table(["wording", *[name for name, _ in cols]], rows)
    rep.note("ToolACE: the no-call rows (both arms train on them). SFT, control:")
    rep.note("their no-call replies on BFCL's 900 call items, seeds pooled;")
    rep.note("untr: the untrained model's. No call: no <tool_call> tag (strict).")
    rep.blank()
    lacks_any = sum(any(LACKS_PHRASE in (m.get("content") or "") for m in r["messages"]
                        if m["role"] == "assistant") for r in real)
    lacks_first = sum(LACKS_PHRASE in first_assistant_text(r["messages"]) for r in no_call)
    rep.kv('rows with "lacks the required"', f"{lacks_any} of {len(real)}")
    rep.kv("  in a no-call row's first turn", lacks_first)
    rep.note('Any assistant turn containing the phrase, whatever its class above.')

    # -- 2 ---------------------------------------------------------------------------
    rep.section("BFCL call items where a trained model makes no call")
    by_item = {i.id: i for c in sftrun.BFCL_CALL_CATS for i in items[c]}

    def classify_reply(item, reply):
        asked = asked_parameters(reply)
        if asked is None:
            return None, []
        q = item.messages[-1]["content"]
        kinds = [parameter_kind(p, item.checker_entry["function"], item.ground_truth, q)
                 for p in asked]
        return reply_kind(kinds), list(zip(asked, kinds, strict=True))

    rep.note('Measured only on replies with the phrase lacks the required "p"')
    rep.note('parameter (or "p" and "q" parameters), where the asked parameters')
    rep.note("can be read off exactly. Each asked parameter, against the item's")
    rep.note("function docs and BFCL's possible answer, first match wins:")
    rep.note("  not in the schema: no offered function has the parameter;")
    rep.note("  optional: the schema doesn't list it as required;")
    rep.note("  in the question: required, and one of BFCL's allowed values for it")
    rep.note("    appears in the question (strings: case-insensitive, as whole")
    rep.note("    words, a plural ending allowed; numbers: as a number; booleans,")
    rep.note("    lists, dicts: never);")
    rep.note("  may be omitted: required, but BFCL's answer allows leaving it out;")
    rep.note("  not found: required, and its value is not in the question.")
    rep.note("A reply takes the first of its parameters' classes in the order")
    rep.note("of the rows below.")
    rep.note("Verbatim only, so \"in the question\" is a lower bound: a value the")
    rep.note("question words another way (a date in another format, \"two\" for")
    rep.note("2) counts as not found.")
    rep.blank()
    table_rows = {k: [] for k in ("no call", "  with the phrase", *[f"    {k}" for k in REPLY_ORDER])}
    arm_runs = [("SFT", sft), ("control", ctl), ("untrained", [un])]
    for _, runs in arm_runs:
        n = phr = 0
        kinds = {k: 0 for k in REPLY_ORDER}
        for sc in runs:
            for i in bfcl_call:
                if sc.bfcl[i] != NO_CALL:
                    continue
                n += 1
                k, _ = classify_reply(by_item[i], replies[sc.label][i])
                if k is not None:
                    phr += 1
                    kinds[k] += 1
        table_rows["no call"].append(n)
        table_rows["  with the phrase"].append(phr)
        for k in REPLY_ORDER:
            table_rows[f"    {k}"].append(kinds[k])
    rep.table(["replies", *[a for a, _ in arm_runs]],
              [[k, *v] for k, v in table_rows.items()])
    rep.note("On BFCL's 900 call items; SFT and control: 3 seeds pooled.")
    rep.blank()
    per_seed = []
    for arm, s, sc in [("SFT", s, x) for s, x in zip(seeds, sft, strict=True)] + [
            ("control", s, x) for s, x in zip(seeds, ctl, strict=True)]:
        ks = [classify_reply(by_item[i], replies[sc.label][i])[0]
              for i in bfcl_call if sc.bfcl[i] == NO_CALL]
        per_seed.append([f"{arm} {s}", len(ks), sum(k is not None for k in ks),
                         sum(k == IN_QUESTION for k in ks), sum(k == OPTIONAL for k in ks)])
    rep.table(["arm", "no call", "phrase", "in question", "optional"], per_seed)
    rep.note("phrase: with the phrase; in question, optional: two of its")
    rep.note("classes above. Per seed, of 900 call items.")

    # examples: the post's item first, then the first items of the two main classes
    s1 = sft[0]
    found = {}
    for i in bfcl_call:
        if s1.bfcl[i] == NO_CALL:
            k, _ = classify_reply(by_item[i], replies[s1.label][i])
            if k in (IN_QUESTION, OPTIONAL):
                found.setdefault(k, []).append(i)
    picks = [EXAMPLE_FIRST] + [i for k in (IN_QUESTION, OPTIONAL)
                               for i in [x for x in found.get(k, []) if x != EXAMPLE_FIRST]
                               [:EXAMPLES_PER_KIND]]
    for i in picks:
        item = by_item[i]
        k, asked = classify_reply(item, replies[s1.label][i])
        rep.section(f"Example: {i} (SFT seed {seeds[0]}: {k or 'no phrase'})")
        rep.note("question:")
        _block(rep, item.messages[-1]["content"])
        for f in item.checker_entry["function"]:
            props = list(f["parameters"].get("properties", {}))
            req = f["parameters"].get("required", [])
            rep.note(f"{f['name']}:")
            _block(rep, "required: " + (", ".join(req) or "none"))
            opt = [p for p in props if p not in req]
            _block(rep, "optional: " + (", ".join(opt) or "none"))
        rep.note("BFCL's possible answer:")
        _block(rep, json.dumps(item.ground_truth, ensure_ascii=False))
        rep.kv("untrained", un.bfcl[i])
        rep.note(f"SFT seed {seeds[0]}, {s1.bfcl[i]}:")
        _block(rep, replies[s1.label][i].strip())
        for p, pk in asked:
            rep.kv(f'  asks for "{p}"', pk)
        c1 = ctl[0]
        rep.kv(f"control seed {seeds[0]}", c1.bfcl[i])
        rep.kv("SFT, all seeds", ", ".join(sc.bfcl[i] for sc in sft))

    # -- 3 ---------------------------------------------------------------------------
    rep.section("Right on our tasks that need a call, pooled")
    ids = [t["id"] for t in splits["test"] if t["kind"] != tasks.NO_FIT]

    def rights(sc):
        return [1.0 if is_right(sc.ours[i]) else 0.0 for i in ids]

    u = rights(un)
    rep.kv("tasks", len(ids))
    rep.table(["arm", "right", "seeds"],
              [["untrained", pct(int(sum(u)), len(ids)), "-"]]
              + [[arm, f"{E.percent(E.exact_mean(E.seed_mean([rights(x) for x in runs]), len(runs)))}"
                  f" (seed mean)", ", ".join(str(int(sum(rights(x)))) for x in runs)]
                 for arm, runs in (("SFT", sft), ("control", ctl))])
    rows = []
    for arm, runs in (("SFT", sft), ("control", ctl)):
        p = E.paired(arm, E.seed_mean([rights(x) for x in runs]), "untrained", u, len(runs), 1)
        rows.append([f"{arm} - untrained", E.points(p.diff), E.interval(p.lo, p.hi)])
    rep.table(["difference", "points", "95% interval"], rows)
    rep.note(f"seeds: right of {len(ids)}, per seed. Intervals: paired by task,")
    rep.note(f"{sftrun.RESAMPLES_TEXT} (as r5a).")
    rep.close()
    return 0
