"""Our tool tasks: prompts, scoring with a model, and the pure summaries the reports print.

Generation is the batched engine (:mod:`tinytools.engine`, the path ``bfcl-score``
uses: the model's own chat template, greedy = argmax or plain temperature
sampling, each reply's draws seeded by the seed, its task id and sample number).
Each reply is checked both
ways and labelled by :func:`tinytools.checkers.check_reply`, whose outcome is
:func:`tinytools.classify.classify` with execution match as the verdict, so
the labels live in one place.

The summaries below are model-free and tested in ``tests/test_ours.py``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from . import tasks
from .checkers import Check, check_reply
from .classify import OUTCOMES, is_right

# r1a/r1b's task: the first test task (by id) of this kind and tool. A relative
# date, a 12-hour time and a worded duration, so the prompt's clock line matters.
R1_KIND = tasks.TRICKY
R1_TOOL = "create_calendar_event"


def r1_task(test: list[dict]) -> dict:
    return next(t for t in sorted(test, key=lambda t: t["id"])
                if t["kind"] == R1_KIND and t["tool"] == R1_TOOL)


def prompt(tok, task: dict) -> str:
    return tok.apply_chat_template(tasks.messages(task), tools=tasks.tool_list(task),
                                   tokenize=False, add_generation_prompt=True)


def limit_per_kind(items: list[dict], limit: int | None) -> list[dict]:
    """The first ``limit`` tasks of each kind, by id; all of them with no limit."""
    items = sorted(items, key=lambda t: t["id"])
    if limit is None:
        return items
    seen: Counter = Counter()
    out = []
    for t in items:
        if seen[t["kind"]] < limit:
            out.append(t)
            seen[t["kind"]] += 1
    return out


# -- token accounting for r1a ------------------------------------------------------------


def tokens_in_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """(tokens wholly inside [start, end), tokens straddling an edge), from character offsets."""
    inside = straddle = 0
    for a, b in offsets:
        if a >= start and b <= end:
            inside += 1
        elif a < end and b > start:
            straddle += 1
    return inside, straddle


@dataclass(frozen=True)
class PromptCensus:
    tokens: int
    tool_list_tokens: int  # the "<tools>" line through the "</tools>" line
    tools_section_tokens: int  # "# Tools" through the end of the system turn
    straddling: int


def prompt_census(tok, text: str) -> PromptCensus:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    # The block's own lines: "<tools></tools>" also appears in the sentence before it.
    a = text.index("\n<tools>\n") + 1
    b = text.index("\n</tools>\n", a) + len("\n</tools>")
    s = text.index("# Tools")
    e = text.index("<|im_end|>", s)
    inner, st1 = tokens_in_span(offsets, a, b)
    section, st2 = tokens_in_span(offsets, s, e)
    return PromptCensus(len(enc["input_ids"]), inner, section, st1 + st2)


def wrap_lines(text: str, width: int) -> list[str]:
    """Split ``text`` into lines no wider than ``width``; a broken line ends in ``\\``.

    For printing a prompt or a reply inside the 74-column report. A long line
    breaks after its last space that fits (mid-word only if there is none);
    dropping each trailing ``\\`` and joining the pieces gives the text back.
    """
    out = []
    for line in text.split("\n"):
        while len(line) > width:
            cut = line.rfind(" ", 0, width - 1) + 1 or width - 1
            out.append(line[:cut] + "\\")
            line = line[cut:]
        out.append(line)
    return out


# -- scoring -----------------------------------------------------------------------------


@dataclass
class Scored:
    task: dict
    replies: list[str]
    checks: list[Check]
    capped: list[bool]
    new_tokens: int
    seconds: float

    @property
    def right(self) -> int:
        return sum(is_right(c.outcome) for c in self.checks)


def score(tok, model, items: list[dict], *, samples: int, temperature: float, seed: int,
          batch_size: int, on_item=None, stats: list | None = None) -> list[Scored]:
    """Generate (the batched engine) and check every task. A task's ``seconds`` is its
    batch's wall seconds x its share of the batch's rows. ``stats`` collects one
    ``engine.BatchStat`` per batch."""
    from .bfcl import MAX_NEW_TOKENS
    from .engine import generate

    run = generate(tok, model, [t["id"] for t in items], [prompt(tok, t) for t in items],
                   samples=samples, temperature=temperature, seed=seed,
                   batch_size=batch_size, max_new_tokens=MAX_NEW_TOKENS, on_done=on_item)
    if stats is not None:
        stats.extend(run.batches)
    out = []
    for task, g in zip(items, run.items, strict=True):
        checks = [check_reply(text, task["reference"]) for text in g.texts]
        out.append(Scored(task, g.texts, checks, g.capped, sum(g.new_tokens), g.seconds))
    return out


# -- summaries (pure) --------------------------------------------------------------------


@dataclass
class KindRow:
    kind: str
    n: int = 0
    outcomes: dict = field(default_factory=lambda: {o: 0 for o in OUTCOMES})
    exact: int = 0
    execution: int = 0


def outcome_rows(results: list[Scored]) -> list[KindRow]:
    """Every reply's outcome, counted per kind (in KINDS order), then a total row."""
    rows = {k: KindRow(k) for k in tasks.KINDS}
    total = KindRow("total")
    for r in results:
        for c in r.checks:
            for row in (rows[r.task["kind"]], total):
                row.n += 1
                row.outcomes[c.outcome] = row.outcomes.get(c.outcome, 0) + 1
                row.exact += c.exact
                row.execution += is_right(c.outcome)
    return [rows[k] for k in tasks.KINDS if rows[k].n] + [total]


def capped_by_kind(results: list[Scored]) -> dict[str, tuple[int, Counter]]:
    """Per kind: (replies that hit the token cap, their outcomes)."""
    out: dict[str, tuple[int, Counter]] = {k: (0, Counter()) for k in tasks.KINDS}
    for r in results:
        n, c = out[r.task["kind"]]
        for check, cap in zip(r.checks, r.capped, strict=True):
            if cap:
                n += 1
                c[check.outcome] += 1
        out[r.task["kind"]] = (n, c)
    return out


def total_row(rows: list[KindRow], label: str = "total") -> KindRow:
    """The sum of some kinds' rows (e.g. the kinds that need a call)."""
    out = KindRow(label)
    for r in rows:
        out.n += r.n
        out.exact += r.exact
        out.execution += r.execution
        for o, k in r.outcomes.items():
            out.outcomes[o] = out.outcomes.get(o, 0) + k
    return out


def disagreements(results: list[Scored]) -> tuple[list[tuple[Scored, int]], list[tuple[Scored, int]]]:
    """(replies passing execution but failing exact, replies passing exact but failing execution)."""
    exec_only, exact_only = [], []
    for r in results:
        for i, c in enumerate(r.checks):
            if c.execution and not c.exact:
                exec_only.append((r, i))
            if c.exact and not c.execution:
                exact_only.append((r, i))
    return exec_only, exact_only


def histogram(right_counts: list[int], samples: int) -> list[int]:
    """How many tasks had 0, 1, ..., ``samples`` right answers."""
    c = Counter(right_counts)
    return [c.get(k, 0) for k in range(samples + 1)]


class HalfUp(Decimal):
    """An exact ratio whose ``f`` formats round half up (2.5 -> "3"), as a reader would.

    Python rounds floats (and plain Decimals, by default) half to even, so 2.5
    would print "2" and 0.25 would print "0.2".
    """

    def __format__(self, spec: str) -> str:
        m = re.fullmatch(r"([^.]*)\.(\d+)f", spec)
        if m is None:
            return super().__format__(spec)
        q = Decimal(1).scaleb(-int(m.group(2)))
        return format(Decimal(self).quantize(q, rounding=ROUND_HALF_UP), spec)


def ratio(k: int, n: int, scale: int = 1) -> HalfUp:
    """``scale * k / n`` as a :class:`HalfUp` (exact to 28 digits)."""
    return HalfUp(Decimal(scale * k) / Decimal(n)) if n else HalfUp(0)


def per_thousand(mixed: int, tasks_n: int, samples: int) -> HalfUp:
    """Post 8's unit: mixed groups per 1,000 answers generated = (1000 / G) x mixed share.

    Compares equal to the float (``per_thousand(2, 100, 8) == 2.5``); printed
    with one decimal, rounded half up.
    """
    return ratio(1000 * mixed, tasks_n * samples)


def pct(k: int, n: int) -> str:
    """``57 of 120 (47.5%)``, the plan's format for a share (half up: 1 of 400 is 0.3%)."""
    return f"{k} of {n} ({share(k, n)})" if n else f"{k} of 0"


def share(k: int, n: int) -> str:
    """``47.5%``: k / n as a percentage, one decimal, rounded half up."""
    return f"{ratio(100 * k, n):.1f}%"


MIXED_HEADERS = ("tasks", "all wrong", "mixed", "all right", "share", "per 1k")


def mixed_rows(groups: list[tuple[str, list[int]]], samples: int) -> list[list]:
    """The "groups with signal" table's rows (after a label column, :data:`MIXED_HEADERS`),
    in the one format ``score`` and ``bfcl-score`` both print.

    ``groups`` is ``(label, right count per task)``; each row is label, tasks,
    all wrong, mixed, all right, share (mixed / tasks) and per 1k (mixed groups
    per 1,000 answers), both rounded half up.
    """
    from .classify import mixed_share

    rows = []
    for label, rc in groups:
        none, mixed, every = mixed_share(rc, samples)
        rows.append([label, len(rc), none, mixed, every, share(mixed, len(rc)),
                     f"{per_thousand(mixed, len(rc), samples):.1f}"])
    return rows



def pass_at_k(right_counts: list[int]) -> int:
    return sum(1 for r in right_counts if r > 0)
