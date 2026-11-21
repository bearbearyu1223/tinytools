"""Figures for post 9 (teaching a small model to call tools).

Four figures, in the post's order:

* ``tool-call-round-trip`` (section 1, :func:`figure_round_trip`): one call's
  round trip. A program renders the tool list and the request into the
  prompt; the model writes text; a program parses it, runs the tool and feeds
  the result back as more text. From the r1 block of a greedy ``score``
  capture and ``outputs/r1-prompt.txt``.
* ``two-checks`` (section 2, :func:`figure_two_checks`): ``check-demo``'s
  three calls, each checked by exact match and by execution match.
* ``untrained-outcomes`` (section 3, :func:`figure_untrained_outcomes`): every
  greedy reply of the untrained model, split by outcome, one bar per BFCL
  category and per kind of our tasks. From the greedy ``score`` and
  ``bfcl-score`` captures; skipped with a message while either is missing or
  still being written.
* ``what-sft-trains`` (section 4, :func:`figure_what_sft_trains`): the
  ``sft-data`` example row, its turns as blocks sized by token count, scored
  vs masked, above its random-call control twin.

None of them runs a model. Every number and every name they draw is parsed
from a capture under ``outputs/``, the same text the post's receipts are pasted
from. The parsers are strict: a section is found by its exact title between
two rules, a ``label  value`` line by its exact label, a table by its exact
header, every row is checked (counts sum to n, totals equal the column sums),
and any mismatch raises :class:`CaptureError`. Nothing falls back to a typed
value. Sentences a figure states are built from parsed values (which
arguments differ, which check passed) and raise if the values don't support
them.

``tinytools figures [--only NAME]`` renders them to ``figures/<name>-light.png``
and ``figures/<name>-dark.png``.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from .classify import (
    CALL_OUTCOMES, CALLED_ANYWAY, DECLINED, NO_CALL, NO_FIT_OUTCOMES, RIGHT, UNPARSEABLE,
    WRONG_ARGUMENTS, WRONG_COUNT, WRONG_FUNCTION, short_label,
)
from .plotting import THEMES, Theme, ink_for, save_both, styled
from .report import WIDTH

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
MODEL = "Qwen2.5-0.5B-Instruct"
RULE = "-" * WIDTH
KV_WIDTH = 34  # Report.kv's label width


class CaptureError(ValueError):
    """The capture does not hold what a figure needs, in the shape it needs."""


# --------------------------------------------------------------------------
# Generic reading
# --------------------------------------------------------------------------


def sections(text: str) -> list[tuple[str, list[str]]]:
    """Every ``(title, body lines)`` in a report, in order.

    A section is a title line between two rules; its body runs to the next
    title's opening rule (or the end).
    """
    lines = text.splitlines()
    starts = [i for i in range(1, len(lines) - 1)
              if lines[i - 1] == RULE and lines[i + 1] == RULE and lines[i] != RULE]
    out = []
    for k, i in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else len(lines)
        out.append((lines[i], lines[i + 2:end]))
    return out


def section(text: str, title: str) -> list[str]:
    """The body of the one section titled exactly ``title``."""
    hits = [body for t, body in sections(text) if t == title]
    if len(hits) != 1:
        raise CaptureError(f"expected one section titled {title!r}, found {len(hits)}")
    return hits[0]


def section_matching(text: str, pattern: str) -> list[tuple[re.Match, list[str]]]:
    """Every section whose whole title matches ``pattern``."""
    return [(m, body) for t, body in sections(text) if (m := re.fullmatch(pattern, t))]


def has_section(text: str, title: str) -> bool:
    return any(t == title for t, _ in sections(text))


def kv(lines: list[str], label: str, where: str) -> str:
    """The value of the one ``Report.kv`` line labelled exactly ``label``.

    ``label`` may start with spaces (the reports indent sub-items that way).
    """
    start = 2 + KV_WIDTH + 1
    hits = [line[start:] for line in lines
            if len(line) > start and line.startswith("  ") and line[start - 1] == " "
            and line[2:2 + KV_WIDTH].rstrip() == label]
    if len(hits) != 1:
        raise CaptureError(f"expected one {label.strip()!r} line in {where}, found {len(hits)}")
    return hits[0]


def to_int(cell: str, what: str) -> int:
    if not re.fullmatch(r"\d+", cell):
        raise CaptureError(f"{what}: {cell!r} is not a count")
    return int(cell)


def yes_no(cell: str, what: str) -> bool:
    if cell not in ("yes", "no"):
        raise CaptureError(f"{what}: {cell!r} is not yes or no")
    return cell == "yes"


def match(pattern: str, value: str, what: str) -> re.Match:
    m = re.fullmatch(pattern, value)
    if not m:
        raise CaptureError(f"{what}: {value!r} is not in the printed format")
    return m


def note_index(lines: list[str], note: str, where: str, start: int = 0) -> int:
    """Index of the one line that is exactly ``"  " + note`` (at or after ``start``)."""
    hits = [i for i in range(start, len(lines)) if lines[i] == "  " + note]
    if len(hits) != 1:
        raise CaptureError(f"expected the note {note!r} once in {where}, found {len(hits)}")
    return hits[0]


def unwrap(lines: list[str], indent: int, where: str) -> str:
    """Undo ``ours.wrap_lines`` as ``cli._block`` printed it.

    Each line carries ``indent`` spaces (blank lines are empty); a line broken
    to fit ends in a backslash, and joining the pieces gives the text back.
    """
    out, buf = [], ""
    for line in lines:
        if line and not line.startswith(" " * indent):
            raise CaptureError(f"{where}: {line!r} is not indented {indent} spaces")
        piece = line[indent:]
        if piece.endswith("\\"):
            buf += piece[:-1]
        else:
            out.append(buf + piece)
            buf = ""
    if buf:
        raise CaptureError(f"{where}: the text ends on a broken line")
    return "\n".join(out)


def block_after(lines: list[str], note: str, where: str, indent: int = 4,
                start: int = 0) -> str:
    """The text printed (by ``cli._block``) under the note ``note``."""
    i = note_index(lines, note, where, start) + 1
    body = []
    while i < len(lines) and lines[i].startswith(" " * indent) and lines[i].strip():
        body.append(lines[i])
        i += 1
    if not body:
        raise CaptureError(f"{where}: nothing printed under {note!r}")
    return unwrap(body, indent, f"{where}, {note!r}")


def table(lines: list[str], headers: list[str], where: str, start: int = 0) -> list[list[str]]:
    """The cells of the first table at or after ``start`` with exactly these headers.

    ``Report.table`` puts two or more spaces between columns, so a line split
    on runs of 2+ spaces gives its cells (a label may hold single spaces). The
    header is followed by its dashed rule; rows run while a line splits into as
    many cells as there are headers (a note under the table has fewer).
    """
    hits = [i for i in range(start, len(lines))
            if lines[i].startswith("  ") and re.split(r"\s{2,}", lines[i].strip()) == headers]
    if not hits:
        raise CaptureError(f"no table headed {headers} in {where}")
    i = hits[0]
    if i + 1 >= len(lines) or not re.fullmatch(r"  -+", lines[i + 1]):
        raise CaptureError(f"no dashed rule under the table headed {headers} in {where}")
    rows = []
    for line in lines[i + 2:]:
        cells = re.split(r"\s{2,}", line.strip())
        if not line.startswith("  ") or len(cells) != len(headers):
            break
        rows.append(cells)
    if not rows:
        raise CaptureError(f"the table headed {headers} in {where} has no rows")
    return rows


def call_of(text: str, what: str) -> tuple[str, dict]:
    """``(name, arguments)`` of a printed call ``{"name": ..., "arguments": {...}}``."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise CaptureError(f"{what}: not JSON ({e})") from None
    if not (isinstance(d, dict) and set(d) == {"name", "arguments"}
            and isinstance(d["name"], str) and isinstance(d["arguments"], dict)):
        raise CaptureError(f"{what}: not a call")
    return d["name"], d["arguments"]


def argument_span(call_text: str, key: str, value) -> tuple[int, int]:
    """Where ``"key": value`` sits in a call printed by ``checkers.call_text``."""
    piece = f"{json.dumps(key)}: {json.dumps(value)}"
    i = call_text.find(piece)
    if i < 0 or call_text.find(piece, i + 1) >= 0:
        raise CaptureError(f"{piece!r} is not in {call_text!r} exactly once")
    return i, i + len(piece)


def differing_arguments(ref: dict, got: dict) -> list[str]:
    """Argument names whose printed values differ (or that one side lacks), in the
    reference's order, then the extra ones."""
    keys = list(ref) + [k for k in got if k not in ref]
    return [k for k in keys
            if k not in ref or k not in got or json.dumps(ref[k]) != json.dumps(got[k])]


def join_names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if not names:
        raise ValueError("no names to join")
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# --------------------------------------------------------------------------
# r1: one task's round trip (score capture + r1-prompt.txt)
# --------------------------------------------------------------------------

R1_TITLE = r"One task, whole: (\S+) \((.+), (\S+)\)"
R1_REPLY = "The model's reply, parsed and run"


@dataclass(frozen=True)
class RoundTrip:
    task_id: str
    kind: str
    tool: str
    prompt: str
    prompt_tokens: int
    tool_list_tokens: int
    reply: str
    call: tuple[str, dict]
    call_text: str
    fed_back: str
    reference: tuple[str, dict]
    outcome: str
    exact: bool
    execution: bool


def parse_round_trip(text: str, prompt_file: str) -> RoundTrip:
    """The r1 block of a greedy ``score`` capture, checked against ``r1-prompt.txt``."""
    found = section_matching(text, R1_TITLE)
    if len(found) != 1:
        raise CaptureError(f"expected one 'One task, whole' section, found {len(found)}")
    m, lines = found[0]
    where = "the r1 section"
    i = note_index(lines, "in outputs/r1-prompt.txt):", where)
    if i + 1 >= len(lines) or lines[i + 1] != "":
        raise CaptureError(f"{where}: no blank line before the prompt")
    end = next((j for j in range(i + 2, len(lines)) if lines[j].startswith("  prompt tokens")),
               None)
    if end is None:
        raise CaptureError(f"{where}: no 'prompt tokens' line after the prompt")
    body = lines[i + 2:end]
    while body and body[-1] == "":
        body.pop()
    printed = unwrap(body, 4, where)
    if printed != prompt_file.rstrip("\n"):
        raise CaptureError("the prompt printed in the capture is not r1-prompt.txt")
    tokens = to_int(kv(lines, "prompt tokens", where), "prompt tokens")
    tool_list = to_int(kv(lines, "  the tool list (<tools> block)", where), "tool list tokens")
    if not 0 < tool_list < tokens:
        raise CaptureError(f"tool list tokens {tool_list} not inside the prompt's {tokens}")

    lines = section(text, R1_REPLY)
    where = "the r1 reply section"
    reply = block_after(lines, "raw reply (up to the end-of-turn token):", where)
    call_text = block_after(lines, "parsed call:", where)
    fed = block_after(lines, "the tool's result, fed back as the next turn:", where)
    ref = block_after(lines, "reference call:", where)
    ex = match(r"(yes|no) / (yes|no)", kv(lines, "exact match / execution match", where),
               "exact match / execution match")
    call = call_of(call_text, "parsed call")
    if f"<tool_call>\n{call_text}\n</tool_call>" != reply.strip():
        raise CaptureError("the raw reply is not the parsed call in <tool_call> tags")
    if "<tool_response>" not in fed:
        raise CaptureError("the fed-back turn has no <tool_response>")
    return RoundTrip(task_id=m.group(1), kind=m.group(2), tool=m.group(3),
                     prompt=prompt_file.rstrip("\n"), prompt_tokens=tokens,
                     tool_list_tokens=tool_list, reply=reply, call=call, call_text=call_text,
                     fed_back=fed, reference=call_of(ref, "reference call"),
                     outcome=kv(lines, "outcome", where), exact=ex.group(1) == "yes",
                     execution=ex.group(2) == "yes")


def prompt_excerpt(prompt: str, width: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """The prompt shortened to fit a card ``width`` characters wide.

    Each tool's JSON line becomes ``{"type": "function", "function": {"name":
    "<name>", ...}}`` (its name parsed from the line); blank lines go; other
    long lines end in ``...``, except the user's request, which is kept whole
    (the caller wraps it). Returns the excerpt and the character spans of the
    tool list (``<tools>`` to ``</tools>``) and of the request.
    """
    lines = prompt.split("\n")
    try:
        a, b = lines.index("<tools>"), lines.index("</tools>")
        u = lines.index("<|im_start|>user")
    except ValueError as e:
        raise CaptureError(f"the prompt lacks a line it should have ({e})") from None
    if not (a < b < u):
        raise CaptureError("the prompt's <tools> block and user turn are out of order")
    out: list[str] = []
    tools_start = tools_end = req_start = req_end = -1
    pos = 0
    for k, line in enumerate(lines):
        if k == a:
            tools_start = pos
        if a < k < b:
            try:
                name = json.loads(line)["function"]["name"]
            except (json.JSONDecodeError, KeyError, TypeError):
                raise CaptureError(f"tool line {line[:40]!r}... is not a tool") from None
            line = f'{{"type": "function", "function": {{"name": "{name}", ...}}}}'
        elif k == u + 1:
            if not line.endswith("<|im_end|>"):
                raise CaptureError("the request line does not end the user turn")
            req_start, req_end = pos, pos + len(line) - len("<|im_end|>")
        elif not line:
            continue
        elif len(line) > width:
            line = line[:width - 3].rstrip() + "..."
        out.append(line)
        pos += len(line) + 1
        if k == b:
            tools_end = pos - 1
    return "\n".join(out), (tools_start, tools_end), (req_start, req_end)


# --------------------------------------------------------------------------
# r2b: three calls checked two ways (check-demo)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckedCall:
    number: int
    title: str
    request: str
    reference_text: str
    reference: tuple[str, dict]
    checked_text: str
    checked: tuple[str, dict]
    exact: bool
    execution: bool
    compared: str


def parse_check_demo(text: str) -> list[CheckedCall]:
    found = section_matching(text, r"Call (\d+): (.+)")
    if [int(m.group(1)) for m, _ in found] != [1, 2, 3]:
        raise CaptureError(f"expected sections Call 1..3, found {[m.group(0) for m, _ in found]}")
    out = []
    for m, lines in found:
        where = f"check-demo call {m.group(1)}"
        ref = block_after(lines, "reference call:", where)
        got = block_after(lines, "the call checked:", where)
        out.append(CheckedCall(
            number=int(m.group(1)), title=m.group(2),
            request=block_after(lines, "request:", where),
            reference_text=ref, reference=call_of(ref, f"{where} reference"),
            checked_text=got, checked=call_of(got, f"{where} checked call"),
            exact=yes_no(kv(lines, "exact match", where), "exact match"),
            execution=yes_no(kv(lines, "execution match", where), "execution match"),
            compared=kv(lines, "execution match compared", where)))
    return out


# --------------------------------------------------------------------------
# r4b: one SFT row and its control twin (sft-data)
# --------------------------------------------------------------------------

R4B_TITLE = r"One training row and its control twin \((.+), seed (\d+)\)"
TURN_HEADERS = ["turn", "tokens", "scored", "masked"]


@dataclass(frozen=True)
class Turn:
    role: str
    tokens: int
    scored: int
    masked: int


@dataclass(frozen=True)
class SftExample:
    row_id: str
    seed: int
    tools: tuple[str, ...]
    real: tuple[Turn, ...]
    control: tuple[Turn, ...]
    user: str
    real_text: str
    real_call: tuple[str, dict]
    control_text: str
    control_call: tuple[str, dict]
    drawn: str
    replaced: str
    source: str
    copied_from: str


def _turns(lines: list[str], note: str, where: str) -> tuple[Turn, ...]:
    i = note_index(lines, note, where)
    rows = table(lines, TURN_HEADERS, where, start=i)
    if rows[-1][0] != "total" or len(rows) < 2:
        raise CaptureError(f"{where}, {note!r}: no total row")
    turns = []
    for r in rows:
        t = Turn(r[0], *(to_int(c, f"{where}, {r[0]}") for c in r[1:]))
        if t.scored + t.masked != t.tokens:
            raise CaptureError(f"{where}, {t.role}: scored + masked != tokens")
        turns.append(t)
    total, body = turns[-1], turns[:-1]
    for field in ("tokens", "scored", "masked"):
        if sum(getattr(t, field) for t in body) != getattr(total, field):
            raise CaptureError(f"{where}, {note!r}: the {field} column does not sum to its total")
    return tuple(body)


def parse_sft_example(text: str) -> SftExample:
    found = section_matching(text, R4B_TITLE)
    if len(found) != 1:
        raise CaptureError(f"expected one 'One training row' section, found {len(found)}")
    m, lines = found[0]
    where = "the r4b section"
    real_text = block_after(lines, "real call:", where)
    control_text = block_after(lines, "control call:", where)
    drawn = match(r"(.+) \(replaced (.+)\)", kv(lines, "control's tool", where), "control's tool")
    args = match(r"(.+): copied from (\S+)", kv(lines, "arguments", where), "arguments")
    ex = SftExample(
        row_id=m.group(1), seed=int(m.group(2)),
        tools=tuple(kv(lines, "tools offered", where).split(", ")),
        real=_turns(lines, "real arm, turn by turn:", where),
        control=_turns(lines, "control, turn by turn:", where),
        user=block_after(lines, "user:", where),
        real_text=real_text, real_call=call_of(real_text, "real call"),
        control_text=control_text, control_call=call_of(control_text, "control call"),
        drawn=drawn.group(1), replaced=drawn.group(2),
        source=args.group(1), copied_from=args.group(2))
    if [t.role for t in ex.real] != [t.role for t in ex.control]:
        raise CaptureError("the real row and its twin have different turns")
    if ex.real_call[0] != ex.replaced or ex.control_call[0] != ex.drawn:
        raise CaptureError("the calls' names disagree with the 'control's tool' line")
    if ex.drawn not in ex.tools or ex.replaced not in ex.tools:
        raise CaptureError("the control's tool is not one the row offers")
    return ex


# --------------------------------------------------------------------------
# r3a / r3b: outcome counts per category (score and bfcl-score, greedy)
# --------------------------------------------------------------------------

CALL_SECTION = "Tasks that need a call: every reply"
NO_FIT_SECTION = "Tasks where no tool fits: every reply"
LAST_SECTION = "Wall clock (varies run to run)"
# The rows each capture must print, in order: (label in the capture, label drawn).
BFCL_CALL_ROWS = (("simple", "simple"), ("multiple", "multiple"), ("parallel", "parallel"),
                  ("par_multi", "parallel_multiple"))
BFCL_NO_FIT_ROWS = (("irrelevance", "irrelevance"),)
OUR_CALL_ROWS = (("choose+fill", "choose+fill"), ("tricky args", "tricky arguments"))
OUR_NO_FIT_ROWS = (("no tool fits", "no tool fits"),)


@dataclass(frozen=True)
class OutcomeRow:
    label: str
    needs_call: bool
    n: int
    counts: dict[str, int]  # outcome (classify's full name) -> count


@dataclass(frozen=True)
class Outcomes:
    model: str
    rows: tuple[OutcomeRow, ...]


def _outcome_rows(lines: list[str], first: str, outcomes: tuple[str, ...],
                  expected: tuple[tuple[str, str], ...], extra: list[str], where: str,
                  needs_call: bool, total: bool) -> list[OutcomeRow]:
    headers = [first, "n", *[short_label(o) for o in outcomes], *extra]
    rows = table(lines, headers, where)
    if total:
        if rows[-1][0] != "total":
            raise CaptureError(f"{where}: no total row")
        rows, total_row = rows[:-1], rows[-1]
    if [r[0] for r in rows] != [e[0] for e in expected]:
        raise CaptureError(f"{where}: rows {[r[0] for r in rows]}, expected "
                           f"{[e[0] for e in expected]}")
    out = []
    for r, (_, label) in zip(rows, expected, strict=True):
        n = to_int(r[1], f"{where}, {r[0]}, n")
        counts = {o: to_int(c, f"{where}, {r[0]}, {o}")
                  for o, c in zip(outcomes, r[2:2 + len(outcomes)], strict=True)}
        if sum(counts.values()) != n:
            raise CaptureError(f"{where}, {r[0]}: outcomes sum to {sum(counts.values())}, n = {n}")
        if n == 0:
            raise CaptureError(f"{where}, {r[0]}: no replies")
        out.append(OutcomeRow(label, needs_call, n, counts))
    if total:
        sums = [sum(x.n for x in out), *[sum(x.counts[o] for x in out) for o in outcomes]]
        if [to_int(c, f"{where}, total") for c in total_row[1:2 + len(outcomes)]] != sums:
            raise CaptureError(f"{where}: the total row is not the column sums")
    return out


def _greedy_model(text: str, decoding: str, what: str) -> str:
    setup = section(text, "Setup")
    if kv(setup, "decoding", f"{what} setup") != decoding:
        raise CaptureError(f"{what} is not a greedy capture")
    if not has_section(text, LAST_SECTION):
        raise CaptureError(f"{what} has no {LAST_SECTION!r} section: still being written?")
    return kv(setup, "model", f"{what} setup")


def parse_score_outcomes(text: str) -> Outcomes:
    """r3b: our three kinds, from a greedy ``tinytools score`` capture."""
    model = _greedy_model(text, "greedy (argmax)", "the score capture")
    calls = _outcome_rows(section(text, CALL_SECTION), "kind", CALL_OUTCOMES, OUR_CALL_ROWS, [],
                          "our call-needing tasks", True, True)
    no_fit = _outcome_rows(section(text, NO_FIT_SECTION), "kind", NO_FIT_OUTCOMES,
                           OUR_NO_FIT_ROWS, [], "our no-tool-fits tasks", False, False)
    return Outcomes(model, tuple(calls + no_fit))


def parse_bfcl_outcomes(text: str) -> Outcomes:
    """r3a: BFCL's five categories, from a greedy ``tinytools bfcl-score`` capture."""
    model = _greedy_model(text, "greedy", "the BFCL capture")
    calls = _outcome_rows(section(text, CALL_SECTION), "category", CALL_OUTCOMES,
                          BFCL_CALL_ROWS, [], "BFCL's call-needing categories", True, True)
    no_fit = _outcome_rows(section(text, NO_FIT_SECTION), "category", NO_FIT_OUTCOMES,
                           BFCL_NO_FIT_ROWS, ["bfcl right"], "BFCL irrelevance", False, False)
    return Outcomes(model, tuple(calls + no_fit))


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------

MONO = "DejaVu Sans Mono"  # ships with matplotlib, so a Mac and a Linux box draw alike
MONO_ADVANCE = 0.602       # DejaVu Sans Mono's advance width, in ems

# The outcome ramp, shared by both themes and by F3 and F5: right darkest,
# lighter the further a reply is from right. Warm neutrals, so they never read
# as an arm's colour (grey untrained, blue SFT, green control, orange post 10).
# On no-tool-fits tasks "declined" is right and takes right's colour, and
# "called anyway" (a function where none fits) takes wrong function's: the two
# never share a bar with the call-needing failures, and F3 keys them apart.
OUTCOME_COLOURS = {
    RIGHT: "#4a4038",
    DECLINED: "#4a4038",
    WRONG_ARGUMENTS: "#7a6e62",
    WRONG_COUNT: "#9c9184",
    WRONG_FUNCTION: "#bcb2a4",
    CALLED_ANYWAY: "#bcb2a4",
    UNPARSEABLE: "#d6cfc3",
    NO_CALL: "#ece8e0",
}
# Drawn left to right: right first, so every bar's right share starts at 0.
CALL_ORDER = (RIGHT, WRONG_ARGUMENTS, WRONG_COUNT, WRONG_FUNCTION, UNPARSEABLE, NO_CALL)
NO_FIT_ORDER = (DECLINED, CALLED_ANYWAY)


def tint(theme: Theme) -> str:
    """A quiet fill for highlighting text on the surface (ink from ``ink_for``)."""
    return "#e8e4da" if theme.name == "light" else "#45413a"


def canvas(w: float, h: float):
    """A figure whose one axes fills it, so data units are inches."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, edge, *, fill="none", lw=1.4, dashed=False, z=2, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                linestyle=(0, (3, 2)) if dashed else "solid", zorder=z))


def arrow(ax, x0, y0, x1, y1, color, *, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=13))


_WIDTHS: dict[float, float] = {}


def char_width(size: float) -> float:
    """One monospace character's width in inches at ``size`` points, as rendered.

    Measured, not taken from the font's advance (``MONO_ADVANCE``): the
    renderer rounds each glyph's advance to whole pixels at the figure's dpi,
    so a typed advance drifts by a character over a long line and highlight
    bands land beside their text.
    """
    if size not in _WIDTHS:
        fig = plt.figure(dpi=plt.rcParams["figure.dpi"])
        n = 200
        t = fig.text(0, 0, "M" * n, fontsize=size, family=MONO)
        _WIDTHS[size] = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / n
        plt.close(fig)
    return _WIDTHS[size]


def wrap_offsets(text: str, width: int) -> list[tuple[int, str]]:
    """``text`` broken into lines of at most ``width`` characters, each with its
    offset in ``text``. Breaks after the last space that fits (mid-word if
    none), keeps hard newlines; the pieces put back together give ``text``."""
    out, pos = [], 0
    for line in text.split("\n"):
        while len(line) > width:
            cut = line.rfind(" ", 0, width) + 1 or width
            out.append((pos, line[:cut]))
            pos += cut
            line = line[cut:]
        out.append((pos, line))
        pos += len(line) + 1
    return out


def mono(ax, x: float, y_top: float, text: str, *, width: int, size: float, color: str,
         spans: list[tuple[int, int, str]] = (), bold_spans: bool = True,
         line_h: float | None = None) -> float:
    """Draw ``text`` in monospace from ``(x, y_top)`` down, wrapped at ``width``.

    ``spans`` are ``(start, end, fill)`` character ranges of ``text`` drawn on
    a filled band with ink from the fill's luminance. Returns the y of the
    last line's bottom.
    """
    cw = char_width(size)
    lh = line_h or size * 1.5 / 72
    lines = wrap_offsets(text, width)
    for i, (off, line) in enumerate(lines):
        y = y_top - (i + 0.5) * lh
        marks = [""] * len(line)
        for a, b, fill in spans:
            for c in range(max(a, off), min(b, off + len(line))):
                marks[c - off] = fill
        # runs of equal marking, drawn piece by piece at their column
        c = 0
        while c < len(line):
            d = c
            while d < len(line) and marks[d] == marks[c]:
                d += 1
            piece, fill = line[c:d], marks[c]
            if fill:
                ax.add_patch(Rectangle((x + c * cw, y - 0.47 * lh), (d - c) * cw, 0.94 * lh,
                                       facecolor=fill, edgecolor="none", zorder=3))
            if piece.strip():
                ax.text(x + c * cw, y, piece, ha="left", va="center", fontsize=size,
                        family=MONO, color=ink_for(fill) if fill else color, zorder=5,
                        fontweight="bold" if (fill and bold_spans) else "normal")
            c = d
    return y_top - len(lines) * lh


def mono_height(text: str, width: int, size: float) -> float:
    return len(wrap_offsets(text, width)) * size * 1.5 / 72


# --------------------------------------------------------------------------
# F1: one call's round trip
# --------------------------------------------------------------------------


def figure_round_trip(theme: Theme, rt: RoundTrip, out: Path | None = None) -> Path:
    """Program lane left, model lane right; every crossing between them is text."""
    ink, sec = theme.ink, theme.secondary
    grey = theme.muted  # the untrained model
    band = tint(theme)
    W = 12.8
    fig, ax = canvas(W, 11.0)
    size = 9.0
    cw = char_width(size)
    lh = size * 1.5 / 72

    px0, px1 = 0.2, 7.75          # program lane
    mx0, mx1 = 8.85, W - 0.2      # model lane
    top = 10.7

    # lane headers and the model lane's shading
    ax.text(px0, top, "A program", ha="left", va="center", fontsize=13, fontweight="bold",
            color=ink)
    ax.text(px0 + 1.15, top, "builds the prompt, parses the reply, runs the tool",
            ha="left", va="center", fontsize=11, color=sec)
    ax.text(mx0, top, "The model", ha="left", va="center", fontsize=13, fontweight="bold",
            color=ink)
    ax.text(mx0 + 1.15, top, "reads text, writes text", ha="left", va="center",
            fontsize=11, color=sec)

    def card(x0, x1, y_top, height, title, edge, *, fill="none"):
        box(ax, x0, y_top - height, x1 - x0, height, edge, fill=fill, lw=1.5)
        ax.text(x0 + 0.6, y_top - 0.27, title, ha="left", va="center", fontsize=11.2,
                fontweight="bold", color=ink, zorder=5)

    def step(x, y, k):
        ax.add_patch(plt.Circle((x, y), 0.15, facecolor="none", edgecolor=sec, lw=1.3, zorder=5))
        ax.text(x, y, str(k), ha="center", va="center", fontsize=10, color=sec,
                fontweight="bold", zorder=6)

    # -- 1: the prompt -----------------------------------------------------------
    inner = px1 - px0 - 0.36
    label_w = 1.25
    chars = int((inner - label_w) / cw)
    excerpt, tools_span, req_span = prompt_excerpt(rt.prompt, chars)
    body_h = mono_height(excerpt, chars, size)
    c1_h = 0.55 + body_h + 0.3 + 0.62
    c1_top = top - 0.35
    card(px0, px1, c1_top, c1_h, "The chat template puts the tool list and the request "
         "into one prompt", sec)
    step(px0 + 0.33, c1_top - 0.27, 1)
    y_text = c1_top - 0.55
    mono(ax, px0 + 0.18, y_text, excerpt, width=chars, size=size, color=sec,
         spans=[(tools_span[0], tools_span[1], band), (req_span[0], req_span[1], band)],
         bold_spans=False)
    # side labels for the two highlighted spans, at their lines' middle
    lines = wrap_offsets(excerpt, chars)

    def span_mid(span):
        rows = [i for i, (off, line) in enumerate(lines)
                if off < span[1] and off + len(line) > span[0]]
        return y_text - (rows[0] + rows[-1] + 1) / 2 * lh, rows

    lx = px0 + 0.18 + chars * cw + 0.15
    for span, words in ((tools_span, ["the tool list"]),
                        (req_span, ["the request"])):
        y, rows = span_mid(span)
        y0 = y_text - rows[0] * lh - 0.03
        y1 = y_text - (rows[-1] + 1) * lh + 0.03
        ax.plot([lx, lx + 0.06, lx + 0.06, lx], [y0, y0, y1, y1], color=sec, lw=1.0, zorder=4)
        for j, w in enumerate(words):
            ax.text(lx + 0.14, y + (len(words) - 1) * 0.1 - j * 0.2, w, ha="left",
                    va="center", fontsize=10, color=sec, zorder=5)

    # token bars: the whole prompt, and the tool list, on one scale
    by = c1_top - 0.55 - body_h - 0.3
    bx0 = px0 + 0.18
    bar_w = inner - 2.3
    per = bar_w / rt.prompt_tokens
    for k, (n, label) in enumerate(((rt.prompt_tokens, "tokens in the prompt"),
                                    (rt.tool_list_tokens, "of them, the tool list"))):
        y = by - k * 0.36
        ax.add_patch(Rectangle((bx0, y - 0.11), n * per, 0.22, facecolor=band if k else grey,
                               edgecolor=theme.axis if k else "none", lw=0.8, zorder=3))
        ax.text(bx0 + n * per + 0.1, y, f"{n:,} {label}", ha="left", va="center",
                fontsize=10.5, color=ink, zorder=5)
    c1_bottom = c1_top - c1_h

    # -- 2: the model writes text -------------------------------------------------
    mchars = int((mx1 - mx0 - 0.36) / cw)
    reply_h = mono_height(rt.reply, mchars, size)
    c2_h = 0.55 + reply_h + 0.55
    c2_top = c1_bottom + 1.35
    model_fill = theme.grid
    card(mx0, mx1, c2_top, c2_h, "The model writes text", grey, fill=model_fill)
    step(mx0 + 0.33, c2_top - 0.27, 2)
    mono(ax, mx0 + 0.18, c2_top - 0.55, rt.reply, width=mchars, size=size, color=ink)
    ax.text(mx0 + 0.18, c2_top - c2_h + 0.25, f"one of the {count_tools(rt.prompt)} tools, "
            "named in its reply", ha="left", va="center", fontsize=10, color=sec, zorder=5)
    y_a1 = c2_top - 0.27
    if not c1_bottom + 0.2 < y_a1 < c1_top - 0.2:
        raise CaptureError("layout: the prompt arrow misses card 1")
    arrow(ax, px1, y_a1, mx0, y_a1, sec)
    ax.text((px1 + mx0) / 2, y_a1 + 0.17, "text", ha="center", va="center", fontsize=10.5,
            color=sec, fontweight="bold")

    # -- 3: a program parses it -----------------------------------------------------
    name, args = rt.call
    parsed = "\n".join([f"name       {name}"] + [
        f"{'arguments' if i == 0 else '':<10} {k} = {json.dumps(v)}"
        for i, (k, v) in enumerate(args.items())])
    pchars = int((px1 - px0 - 0.36) / cw)
    wrong = differing_arguments(rt.reference[1], args) if rt.reference[0] == name else []
    c3_h = 0.55 + mono_height(parsed, pchars, size) + (0.5 if wrong else 0.2)
    c3_top = c2_top - c2_h + 0.45 - 0.05
    c3_top = min(c3_top, c1_bottom - 0.45)
    card(px0, px1, c3_top, c3_h, "A program parses the text into a call", sec)
    step(px0 + 0.33, c3_top - 0.27, 3)
    spans = []
    if wrong:
        off = 0
        for line in parsed.split("\n"):
            for k in wrong:
                if k in args and re.search(rf"\b{re.escape(k)} = ", line):
                    spans.append((off + 11, off + len(line), band))
            off += len(line) + 1
    mono(ax, px0 + 0.18, c3_top - 0.55, parsed, width=pchars, size=size, color=ink, spans=spans)
    if wrong:
        ax.text(px0 + 0.18, c3_top - c3_h + 0.25,
                f"{join_names(wrong)}: not what the request asked for. "
                "The program parses and runs the call as written.",
                ha="left", va="center", fontsize=10, color=sec, zorder=5)
    y_a2 = c3_top - 0.27
    if not c2_top - c2_h + 0.1 < y_a2 < c2_top - 0.1:
        # the reply arrow must leave the model's card: move card 3 up under it
        raise CaptureError("layout: the reply arrow misses card 2")
    arrow(ax, mx0, y_a2, px1, y_a2, grey)
    ax.text((px1 + mx0) / 2, y_a2 + 0.17, "text", ha="center", va="center", fontsize=10.5,
            color=sec, fontweight="bold")

    # -- 4: runs the tool, feeds the result back ------------------------------------
    c4_h = 0.55 + mono_height(rt.fed_back, pchars, size) + 0.2
    c4_top = c3_top - c3_h - 0.5
    arrow(ax, (px0 + px1) / 2, c3_top - c3_h, (px0 + px1) / 2, c4_top, sec)
    card(px0, px1, c4_top, c4_h, f"It runs {name}, and writes the result into the "
         "conversation as text", sec)
    step(px0 + 0.33, c4_top - 0.27, 4)
    mono(ax, px0 + 0.18, c4_top - 0.55, rt.fed_back, width=pchars, size=size, color=ink)

    # -- 5: back to the model -----------------------------------------------------
    c5_h = 0.85
    y_a3 = c4_top - c4_h / 2
    c5_top = y_a3 + 0.27
    card(mx0, mx1, c5_top, c5_h, "The model reads it", grey, fill=model_fill)
    step(mx0 + 0.33, c5_top - 0.27, 5)
    ax.text(mx0 + 0.18, c5_top - 0.62, "as more prompt, and writes again", ha="left",
            va="center", fontsize=10.5, color=sec, zorder=5)
    arrow(ax, px1, y_a3, mx0, y_a3, sec)
    ax.text((px1 + mx0) / 2, y_a3 + 0.17, "text", ha="center", va="center", fontsize=10.5,
            color=sec, fontweight="bold")

    bottom = min(c4_top - c4_h, c5_top - c5_h) - 0.15
    ax.set_ylim(bottom, top + 0.3)
    fig.set_size_inches(W, top + 0.3 - bottom)
    return save_both(fig, "tool-call-round-trip", theme, out)


def count_tools(prompt: str) -> int:
    lines = prompt.split("\n")
    try:
        return lines.index("</tools>") - lines.index("<tools>") - 1
    except ValueError:
        raise CaptureError("the prompt has no <tools> block") from None


# --------------------------------------------------------------------------
# F2: three calls, two checks
# --------------------------------------------------------------------------


def two_checks_claims(calls: list[CheckedCall]) -> None:
    """The figure's shape assumes: one passes both, one passes execution only,
    one fails both. Raise if check-demo printed anything else."""
    got = [(c.exact, c.execution) for c in calls]
    if got != [(True, True), (False, True), (False, False)]:
        raise CaptureError(f"check-demo's verdicts are {got}, not pass both / "
                           "execution only / fail both")


def figure_two_checks(theme: Theme, calls: list[CheckedCall], out: Path | None = None) -> Path:
    two_checks_claims(calls)
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    band = tint(theme)
    yes_fill = OUTCOME_COLOURS[RIGHT]
    W = 12.8
    size = 9.2
    cw = char_width(size)
    tx = 0.2                  # text column
    lab_w = 1.0               # "reference" / "checked" labels
    cx = tx + lab_w           # where the calls start
    chars = int((9.3 - cx) / cw)
    ex_x, run_x = 10.25, 11.75  # badge column centres
    fig, ax = canvas(W, 9.0)
    top = 8.8
    ax.text(ex_x, top, "exact match", ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=ink)
    ax.text(run_x, top, "execution match", ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=ink)

    def badge(x, y, ok):
        w, h = 0.78, 0.36
        box(ax, x - w / 2, y - h / 2, w, h, yes_fill if ok else muted,
            fill=yes_fill if ok else "none", lw=1.3, r=0.12)
        ax.text(x, y, "yes" if ok else "no", ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=ink_for(yes_fill) if ok else sec, zorder=5)

    y = top - 0.35
    for c in calls:
        ax.plot([tx, W - 0.2], [y, y], color=theme.grid, lw=1.0, zorder=1)
        y -= 0.33
        ax.text(tx, y, f"{c.number}.  {c.title[0].upper()}{c.title[1:]}", ha="left",
                va="center", fontsize=11.5, fontweight="bold", color=ink)
        y -= 0.34
        ax.text(tx, y, "request", ha="left", va="center", fontsize=10, color=muted)
        ax.text(cx, y, c.request, ha="left", va="center", fontsize=10.5, color=ink)
        y -= 0.3
        row_top = y + 0.1
        ax.text(tx, y + 0.12 - 0.5 * size * 1.5 / 72, "reference", ha="left", va="center", fontsize=10, color=muted)
        y = mono(ax, cx, y + 0.12, c.reference_text, width=chars, size=size, color=sec) - 0.08
        ax.text(tx, y - 0.5 * size * 1.5 / 72, "checked", ha="left", va="center", fontsize=10, color=muted)
        diff = (differing_arguments(c.reference[1], c.checked[1])
                if c.reference[0] == c.checked[0] else [])
        spans = [(*argument_span(c.checked_text, k, c.checked[1][k]), band)
                 for k in diff if k in c.checked[1]]
        y = mono(ax, cx, y, c.checked_text, width=chars, size=size, color=ink, spans=spans)
        notes = []
        if diff:
            notes.append(f"differs in {join_names(diff)}")
        if list(c.reference[1]) != list(c.checked[1]) and set(c.reference[1]) == set(c.checked[1]):
            notes.append("arguments in another order")
        if notes:
            y -= 0.2
            ax.text(cx, y, "; ".join(notes), ha="left", va="center", fontsize=10, color=sec)
        mid = (row_top + y) / 2
        badge(ex_x, mid + 0.12, c.exact)
        badge(run_x, mid + 0.12, c.execution)
        ax.text(run_x, mid - 0.25, c.compared, ha="center", va="center", fontsize=9.5,
                color=sec)
        y -= 0.3
    ax.plot([tx, W - 0.2], [y, y], color=theme.grid, lw=1.0, zorder=1)
    ax.set_ylim(y - 0.1, top + 0.3)
    fig.set_size_inches(W, top + 0.4 - y)
    return save_both(fig, "two-checks", theme, out)


# --------------------------------------------------------------------------
# F3: the untrained model's outcomes, per category
# --------------------------------------------------------------------------


def figure_untrained_outcomes(theme: Theme, bfcl: Outcomes, ours: Outcomes,
                              out: Path | None = None) -> Path:
    if bfcl.model != ours.model:
        raise CaptureError(f"the BFCL capture scores {bfcl.model}, ours {ours.model}")
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    groups = [
        ("BFCL, a call is needed", [(r.label, r) for r in bfcl.rows if r.needs_call]),
        ("Our tasks, a call is needed", [(r.label, r) for r in ours.rows if r.needs_call]),
        ("No tool fits: declining is right",
         [(f"BFCL {r.label}", r) for r in bfcl.rows if not r.needs_call]
         + [(f"ours: {r.label}", r) for r in ours.rows if not r.needs_call]),
    ]
    W = 12.8
    fig, ax = canvas(W, 8.0)
    x0, x1 = 2.6, 11.4
    bar_h, row_gap, group_gap = 0.42, 0.2, 0.2
    top = 7.0
    ax.text(0.2, top + 0.75, f"Every greedy reply of the untrained {ours.model.split('/')[-1]}, "
            "by outcome", ha="left", va="center", fontsize=13.5, fontweight="bold", color=ink)

    # legend: the call-needing outcomes, then the two no-fit ones
    def legend(y, items):
        x = x0
        for o, label in items:
            fill = OUTCOME_COLOURS[o]
            ax.add_patch(Rectangle((x, y - 0.09), 0.26, 0.18, facecolor=fill,
                                   edgecolor=theme.axis, lw=0.6, zorder=3))
            t = ax.text(x + 0.34, y, label, ha="left", va="center", fontsize=10.5, color=sec)
            x += 0.34 + text_width(fig, t) + 0.35
    legend(top + 0.32, [(o, o) for o in CALL_ORDER])

    y = top - 0.2
    for gi, (title, rows) in enumerate(groups):
        if gi:
            y -= group_gap
        ax.text(0.2, y, title, ha="left", va="center", fontsize=11.2, fontweight="bold",
                color=ink)
        if gi == 2:
            legend(y, [(DECLINED, "declined (right here)"), (CALLED_ANYWAY, CALLED_ANYWAY)])
        y -= 0.42
        for label, r in rows:
            order = CALL_ORDER if r.needs_call else NO_FIT_ORDER
            ax.text(x0 - 0.15, y, label, ha="right", va="center", fontsize=10.5, color=ink)
            x = x0
            for o in order:
                n = r.counts[o]
                if n == 0:
                    continue
                w = (x1 - x0) * n / r.n
                fill = OUTCOME_COLOURS[o]
                ax.add_patch(Rectangle((x, y - bar_h / 2), w, bar_h, facecolor=fill,
                                       edgecolor=theme.surface, lw=1.0, zorder=3))
                t = f"{n}"
                if w >= 0.14 * len(t) + 0.1:
                    ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=9.5,
                            color=ink_for(fill), zorder=5)
                x += w
            ax.text(x1 + 0.15, y, f"{_right(r)} of {r.n} right", ha="left", va="center",
                    fontsize=10, color=sec)
            y -= bar_h + row_gap
    # share axis under the bars
    y_axis = y + row_gap - 0.05
    ax.plot([x0, x1], [y_axis, y_axis], color=theme.axis, lw=0.8)
    for k in range(0, 101, 25):
        xx = x0 + (x1 - x0) * k / 100
        ax.plot([xx, xx], [y_axis, y_axis - 0.06], color=theme.axis, lw=0.8)
        ax.text(xx, y_axis - 0.2, f"{k}%", ha="center", va="center", fontsize=9.5, color=muted)
    ax.text((x0 + x1) / 2, y_axis - 0.46, "share of the category's replies (counts where they fit)",
            ha="center", va="center", fontsize=10, color=sec)
    bottom = y_axis - 0.65
    ax.set_ylim(bottom, top + 1.0)
    fig.set_size_inches(W, top + 1.0 - bottom)
    return save_both(fig, "untrained-outcomes", theme, out)


def _right(r: OutcomeRow) -> int:
    return r.counts[RIGHT] if r.needs_call else r.counts[DECLINED]


def text_width(fig, t) -> float:
    """A text artist's rendered width in inches."""
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return bb.width / fig.dpi


# --------------------------------------------------------------------------
# F4: what SFT trains
# --------------------------------------------------------------------------


def figure_what_sft_trains(theme: Theme, ex: SftExample, out: Path | None = None) -> Path:
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    blue, green = theme.series[0], theme.series[2]
    band = tint(theme)
    masked_fill = theme.grid
    W = 12.8
    fig, ax = canvas(W, 8.0)
    x0 = 2.0
    longest = max(sum(t.tokens for t in ex.real), sum(t.tokens for t in ex.control))
    per = (W - 0.3 - x0) / longest
    size = 9.2
    cw = char_width(size)
    top = 7.6

    # legend
    lx = x0
    for fill, edge, label in ((blue, blue, "scored (in the loss): real arm"),
                              (green, green, "scored: control"),
                              (masked_fill, theme.axis, "masked (read, not scored)")):
        ax.add_patch(Rectangle((lx, top - 0.09), 0.3, 0.18, facecolor=fill, edgecolor=edge,
                               lw=0.8, zorder=3))
        t = ax.text(lx + 0.4, top, label, ha="left", va="center", fontsize=10.5, color=sec)
        lx += 0.4 + text_width(fig, t) + 0.45

    def strip(y, turns, colour, label, sub):
        ax.text(0.2, y + 0.1, label, ha="left", va="center", fontsize=12, fontweight="bold",
                color=colour)
        ax.text(0.2, y - 0.17, sub, ha="left", va="center", fontsize=10, color=sec)
        x = x0
        h = 0.5
        spans = {}
        for t in turns:
            for n, scored in ((t.masked, False), (t.scored, True)):
                if n == 0:
                    continue
                w = n * per
                fill = colour if scored else masked_fill
                ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=fill,
                                       edgecolor=theme.surface if scored else theme.axis,
                                       lw=0.8, zorder=3))
                spans.setdefault(t.role, []).append((x, x + w, scored, n))
                x += w
        return spans

    def above(spans, role, lines, y, *, inside=False):
        """Label a turn: inside its block if it fits, else above it."""
        xa, xb = spans[role][0][0], spans[role][-1][1]
        if inside:
            for j, line in enumerate(lines):
                ax.text(xa + 0.15, y + 0.1 - 0.22 * j, line, ha="left", va="center",
                        fontsize=10.2, color=ink_for(masked_fill), zorder=5)
            return
        for j, line in enumerate(lines):
            ax.text((xa + xb) / 2, y + 0.42 + 0.2 * (len(lines) - 1 - j), line,
                    ha="center", va="center", fontsize=10, color=sec, zorder=5)

    def assistant_labels(spans, y, colour):
        (ma, mb, _, mn), (sa, sb, _, sn) = spans["assistant"]
        # masked header: leader to the left; scored body: above
        ax.text(sb, y + 0.42, f"{sn} scored", ha="right", va="center", fontsize=10.2,
                color=colour, fontweight="bold")
        ax.text(sb, y + 0.62, f"assistant: {mn + sn} tokens, {mn} masked (its header)",
                ha="right", va="center", fontsize=10, color=sec)

    roles = [t.role for t in ex.real]
    if roles != ["system", "user", "assistant"]:
        raise CaptureError(f"the example's turns are {roles}, not system, user, assistant")
    for row in (ex.real, ex.control):
        sysr, user, asst = row
        if sysr.scored or user.scored or asst.masked == 0 or asst.scored == 0:
            raise CaptureError("the example is not masked the way the figure draws it")
    if ex.real[:2] != ex.control[:2]:
        raise CaptureError("the twin's system and user turns differ from the real row's")

    # -- real row ----------------------------------------------------------------
    y_real = top - 1.25
    sp = strip(y_real, ex.real, blue, "Real arm", "the row as written")
    sysr, user, asst = ex.real
    above(sp, "system", [f"system: {sysr.tokens} tokens, masked",
                         f"the tool list ({', '.join(ex.tools)})"], y_real, inside=True)
    ux = (sp["user"][0][0] + sp["user"][0][1]) / 2
    ax.text(ux, y_real + 0.42, f"user: {user.tokens}", ha="center", va="center",
            fontsize=10, color=sec)
    assistant_labels(sp, y_real, blue)

    # the real call, zoomed out of the scored block
    def zoom(spans, y, text, colour, card_top, extra_spans=(), note_lines=()):
        sa, sb = spans["assistant"][-1][0], spans["assistant"][-1][1]
        width = int((W - 0.3 - x0 - 0.36) / cw)
        h = 0.3 + mono_height(text, width, size) + 0.22 * len(note_lines) + (0.1 if note_lines else 0)
        cx0, cx1 = x0, W - 0.3
        ax.add_patch(Polygon([(sa, y - 0.25), (sb, y - 0.25), (cx1, card_top), (cx0, card_top)],
                             closed=True, facecolor=colour, alpha=0.10, edgecolor="none",
                             zorder=1))
        box(ax, cx0, card_top - h, cx1 - cx0, h, colour, lw=1.4)
        yb = mono(ax, cx0 + 0.18, card_top - 0.15, text, width=width, size=size, color=ink,
                  spans=list(extra_spans))
        for j, line in enumerate(note_lines):
            ax.text(cx0 + 0.18, yb - 0.17 - 0.22 * j, line, ha="left", va="center",
                    fontsize=10.2, color=sec, zorder=5)
        return card_top - h

    ax.text(0.2, y_real - 0.75, "its call:", ha="left", va="center", fontsize=10, color=sec)
    card_bottom = zoom(sp, y_real, ex.real_text, blue, y_real - 0.6)

    # the request, beside the row label
    # -- control row ---------------------------------------------------------------
    y_ctrl = card_bottom - 1.15
    sp2 = strip(y_ctrl, ex.control, green, "Control", "a random-call twin")
    sysc, userc, asstc = ex.control
    above(sp2, "system", [f"system: {sysc.tokens} tokens, masked", "the same as the real row, and so is the user turn"],
          y_ctrl, inside=True)
    ux = (sp2["user"][0][0] + sp2["user"][0][1]) / 2
    ax.text(ux, y_ctrl + 0.42, f"user: {userc.tokens}", ha="center", va="center",
            fontsize=10, color=sec)
    assistant_labels(sp2, y_ctrl, green)

    name, args = ex.control_call
    changed = [(*argument_span(ex.control_text, "name", name), band)]
    for k, v in args.items():
        changed.append((*argument_span(ex.control_text, k, v), band))
    notes = [f"tool drawn at random from the row's {len(ex.tools)}: {ex.drawn} "
             f"(the real call used {ex.replaced})"]
    if ex.source.startswith("another row"):
        notes.append(f"arguments copied from another row ({ex.copied_from})")
    else:
        notes.append(f"arguments: {ex.source}, from {ex.copied_from}")
    ax.text(0.2, y_ctrl - 0.75, "its call:", ha="left", va="center", fontsize=10, color=sec)
    bottom = zoom(sp2, y_ctrl, ex.control_text, green, y_ctrl - 0.6, changed, notes)

    # the request both rows answer
    ax.text(0.2, top - 0.5, "request:", ha="left", va="center", fontsize=10, color=muted)
    ax.text(x0, top - 0.5, ex.user, ha="left", va="center", fontsize=10.5, color=ink)

    ax.set_ylim(bottom - 0.15, top + 0.3)
    fig.set_size_inches(W, top + 0.45 - bottom)
    return save_both(fig, "what-sft-trains", theme, out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

NAMES = ("tool-call-round-trip", "two-checks", "untrained-outcomes", "what-sft-trains")


def default_score_capture(model: str = MODEL) -> Path:
    """The greedy score capture over the whole test split; while it doesn't exist,
    the 30-per-kind trial in old-labels/ (its r1 block is the same task)."""
    full = OUTPUTS / f"score-{model}-test-greedy-limitall.txt"
    return full if full.exists() else OUTPUTS / "old-labels" / f"score-{model}-test-greedy-limit30.txt"


def default_bfcl_capture(model: str = MODEL) -> Path | None:
    """The one greedy BFCL capture under outputs/ (any limit); None if there is none."""
    found = sorted(OUTPUTS.glob(f"bfcl-{model}-greedy-limit*.txt"))
    if len(found) > 1:
        raise SystemExit(f"several greedy BFCL captures: {[p.name for p in found]}; "
                         "choose one with --bfcl")
    return found[0] if found else None


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def make_figures(only: str | None = None, *, score: Path | None = None,
                 bfcl: Path | None = None, out: Path | None = None) -> list[Path]:
    """Render the figures (both themes); return the files written."""
    if only is not None and only not in NAMES:
        raise SystemExit(f"unknown figure {only!r}; choose from {list(NAMES)}")
    want = [only] if only else list(NAMES)
    written: list[Path] = []

    def render(fn, *args):
        for theme in THEMES:
            with styled(theme):
                written.append(fn(theme, *args, out=out))

    if "tool-call-round-trip" in want:
        path = score or default_score_capture()
        _say(f"tool-call-round-trip: r1 from {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        rt = parse_round_trip(path.read_text(), (OUTPUTS / "r1-prompt.txt").read_text())
        render(figure_round_trip, rt)
    if "two-checks" in want:
        render(figure_two_checks, parse_check_demo((OUTPUTS / "check-demo.txt").read_text()))
    if "untrained-outcomes" in want:
        s_path = score or OUTPUTS / f"score-{MODEL}-test-greedy-limitall.txt"
        b_path = bfcl or default_bfcl_capture()
        missing = [p.name for p in (s_path, b_path) if p is not None and not p.exists()]
        if b_path is None:
            missing.append(f"bfcl-{MODEL}-greedy-limit<N>.txt")
        if missing:
            _say(f"untrained-outcomes: skipped, capture missing: {', '.join(missing)}")
        else:
            s_text, b_text = s_path.read_text(), b_path.read_text()
            unfinished = [p.name for p, t in ((s_path, s_text), (b_path, b_text))
                          if not has_section(t, LAST_SECTION)]
            if unfinished:
                _say(f"untrained-outcomes: skipped, still being written: {', '.join(unfinished)}")
            else:
                _say(f"untrained-outcomes: from {s_path.name} and {b_path.name}")
                render(figure_untrained_outcomes, parse_bfcl_outcomes(b_text),
                       parse_score_outcomes(s_text))
    if "what-sft-trains" in want:
        render(figure_what_sft_trains, parse_sft_example((OUTPUTS / "sft-data.txt").read_text()))
    for p in written:
        _say(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return written
