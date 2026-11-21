"""Our tool tasks: prompts, scoring with a model, and the pure summaries the reports print.

Generation is ``bfcl.generate_text`` (the path ``bfcl-score`` uses: the model's
own chat template, greedy = argmax or plain temperature sampling, each task's
generator seeded with ``seed + crc32(task id)``). Each reply is checked both
ways and labelled by :func:`tinytools.checkers.check_reply`, whose outcome is
:func:`tinytools.classify.classify` with execution match as the verdict, so
the labels live in one place.

The summaries below are model-free and tested in ``tests/test_ours.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import tasks
from .checkers import Check, check_reply
from .classify import OUTCOMES

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
        return sum(c.execution for c in self.checks)


def score(tok, model, items: list[dict], *, samples: int, temperature: float, seed: int,
          on_item=None) -> list[Scored]:
    from .bfcl import generate_text, generation_config

    cfg = generation_config(tok, model, samples, temperature)
    out = []
    for k, task in enumerate(items, start=1):
        g = generate_text(tok, model, prompt(tok, task), task["id"], cfg, seed)
        checks = [check_reply(text, task["reference"]) for text in g.texts]
        out.append(Scored(task, g.texts, checks, g.capped, sum(g.new_tokens), g.seconds))
        if on_item:
            on_item(k)
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
                row.execution += c.execution
    return [rows[k] for k in tasks.KINDS if rows[k].n] + [total]


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


def per_thousand(mixed: int, tasks_n: int, samples: int) -> float:
    """Post 8's unit: mixed groups per 1,000 answers generated = (1000 / G) x mixed share."""
    return 1000 * mixed / (tasks_n * samples) if tasks_n else 0.0


def pct(k: int, n: int) -> str:
    """``57 of 120 (47.5%)``, the plan's format for a share."""
    return f"{k} of {n} ({100 * k / n:.1f}%)" if n else f"{k} of 0"


def pass_at_k(right_counts: list[int]) -> int:
    return sum(1 for r in right_counts if r > 0)
