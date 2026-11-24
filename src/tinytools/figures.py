"""Figures for post 9 (teaching a small model to call tools).

Nine figures, in the post's order:

* ``tool-call-round-trip`` (section 1, :func:`figure_round_trip`): one call's
  round trip. A program renders the tool list and the request into the
  prompt; the model writes text; a program parses it, runs the tool and feeds
  the result back as more text. From the r1 block of a greedy ``score``
  capture and ``outputs/r1-prompt.txt``.
* ``two-checks`` (section 2, :func:`figure_two_checks`): ``check-demo``'s
  three calls, each checked by exact match and by execution match.
* ``untrained-outcomes`` (section 3, :func:`figure_untrained_outcomes`): every
  greedy reply of the untrained model, split by outcome, one bar per BFCL
  category and per kind of our tasks. From the greedy ``score`` capture and
  the greedy ``bfcl-score --split eval`` capture; skipped with a message while
  either is missing or still being written.
* ``toolace-funnel`` (section 4, :func:`figure_toolace_funnel`): ToolACE's rows
  down to the ones each arm trains on, bars sized by row count. From the
  ``toolace-stats`` and ``sft-data`` captures.
* ``what-sft-trains`` (section 4, :func:`figure_what_sft_trains`): the
  ``sft-data`` example row, its turns as blocks sized by token count, scored
  vs masked, above its random-call control twin.
* ``sft-effects`` (section 5, :func:`figure_effects`): r5d's four effects, each
  set a row of three paired differences (SFT - untrained, control - untrained,
  SFT - control) with their intervals. From ``outputs/sft.txt``.
* ``group-movements`` (section 6, :func:`figure_movements`): r6a's movements on
  the tasks that need a call, the noise floor beside SFT and the control, and
  each arm's difference from the floor. From ``outputs/sft.txt``.
* ``base-vs-instruct`` (section 5, :func:`figure_base_vs_instruct`): each model
  untrained to SFT on the post's eight rows, from r5b and r5d of both reports.
* ``onetool-rerun`` (section 5, :func:`figure_rerun`): right on BFCL by category
  for the untrained model, the main run and the rerun without the one-tool
  no-call rows, with each arm's no-call share. From ``sft-onetool.txt`` (its
  per-category counts), ``sft-data-onetool.txt`` (the rows left out) and
  ``sft.txt`` (r5a's per-seed counts, checked against the main run's).

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
MODEL = "Qwen2.5-1.5B-Instruct"  # the post's model (the main run, 2026-11-23)
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


def report_model(text: str) -> str:
    """The model a report scored (its Setup's ``model`` line), without the org prefix."""
    return kv(section(text, "Setup"), "model", "Setup").split("/")[-1]


def has_section(text: str, title: str) -> bool:
    return any(t == title for t, _ in sections(text))


NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten")


def number_word(k: int) -> str:
    """``k`` as a word (zero to ten), as a sentence in a figure says it."""
    if not 0 <= k < len(NUMBER_WORDS):
        raise CaptureError(f"no word for {k}")
    return NUMBER_WORDS[k]


def seed_list(value: str, what: str) -> list[int]:
    """A printed seed list (``1, 2, 3``) as ints."""
    return [int(x) for x in match(r"\d+(?:, \d+)*", value, what).group(0).split(", ")]


def seed_count(text: str) -> int:
    """How many seeds an sft-report's arms are means of: its Setup's ``seeds with a whole
    pair`` line."""
    label = "seeds with a whole pair"
    return len(seed_list(kv(section(text, "Setup"), label, "Setup"), f"Setup, {label}"))


def same(values, what: str):
    """The one value every panel shares; a CaptureError if they differ."""
    got = set(values)
    if len(got) != 1:
        raise CaptureError(f"the panels differ in {what}: {sorted(got)}")
    return got.pop()


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
    prompt_exact: str  # r1-prompt.txt as written (``prompt`` is it without the final newline)
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
                     prompt_exact=prompt_file,
                     tool_list_tokens=tool_list, reply=reply, call=call, call_text=call_text,
                     fed_back=fed, reference=call_of(ref, "reference call"),
                     outcome=kv(lines, "outcome", where), exact=ex.group(1) == "yes",
                     execution=ex.group(2) == "yes")


def prompt_excerpt(prompt: str, width: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """The prompt shortened to fit a card ``width`` characters wide (see
    :func:`_excerpt`)."""
    return _excerpt(prompt, width)[:3]


def _excerpt(prompt: str, width: int):
    """The prompt shortened to fit a card ``width`` characters wide.

    Each tool's JSON line becomes ``{"type": "function", "function": {"name":
    "<name>", ...}}`` (its name parsed from the line); blank lines go; other
    long lines end in ``...``, except the user's request, which is kept whole
    (the caller wraps it). Returns the excerpt and the character spans of the
    tool list (``<tools>`` to ``</tools>``) and of the request, and per kept line
    ``(start, end)`` in the prompt and ``(start, end)`` in the excerpt.
    """
    lines = prompt.split("\n")
    kept: list[tuple[tuple[int, int], tuple[int, int]]] = []
    orig = 0
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
        o_start, orig = orig, orig + len(line) + 1
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
        kept.append(((o_start, o_start + len(lines[k])), (pos, pos + len(line))))
        pos += len(line) + 1
        if k == b:
            tools_end = pos - 1
    return "\n".join(out), (tools_start, tools_end), (req_start, req_end), kept


def part_spans(prompt: str, exact: str, width: int) -> list[tuple[str, tuple[int, int]]]:
    """Each part of the prompt (``promptview.split_prompt``'s names, the readable view's)
    as a character span of :func:`_excerpt`'s text: from the first to the last kept line
    holding one of its characters. A part with no characters on a kept line (the empty
    assistant turn) is left out. ``exact``: the prompt as the template wrote it, which
    ``split_prompt`` cuts (``prompt`` is it, final newline stripped)."""
    from .promptview import split_prompt

    if exact.rstrip("\n") != prompt:
        raise CaptureError("the exact prompt is not the excerpt's prompt")
    kept = _excerpt(prompt, width)[3]
    out = []
    for seg in split_prompt(exact):
        if seg.kind != "part":
            continue
        hit = [ex for (o0, o1), ex in kept if o1 > o0 and o0 < seg.end and o1 > seg.start
               and prompt[max(o0, seg.start):min(o1, seg.end)].strip()]
        if hit:
            out.append((seg.name, (hit[0][0], hit[-1][1])))
    return out


PROMPT_VIEW_PARTS = "The prompt, part by part"
PROMPT_VIEW_TOKENS = "The prompt's tokens, part by part"


def check_prompt_view(view: str, rt: RoundTrip) -> None:
    """``outputs/r1-prompt-view.txt`` against the figure: its part labels are
    ``split_prompt``'s, in order, and its token table gives r1's prompt and tool-list
    counts (the bar's two numbers)."""
    from .promptview import PARTS, split_prompt

    labels = [m.group(1) for x in section(view, PROMPT_VIEW_PARTS)
              if (m := re.fullmatch(r"  \[(\w+)\]", x))]
    names = [s.name for s in split_prompt(rt.prompt_exact) if s.kind == "part"]
    if labels != names or tuple(names) != PARTS:
        raise CaptureError(f"the prompt view's parts {labels}, the prompt's {names}")
    rows = {r[0]: r[1] for r in table(section(view, PROMPT_VIEW_TOKENS), ["part", "tokens"],
                                      "the prompt view's tokens")}
    got = (to_int(rows.get("prompt tokens (indented rows aside)", ""), "view, prompt tokens"),
           to_int(rows.get("the tool list itself", ""), "view, tool list tokens"))
    if got != (rt.prompt_tokens, rt.tool_list_tokens):
        raise CaptureError(f"the prompt view counts {got}, r1 "
                           f"{(rt.prompt_tokens, rt.tool_list_tokens)}")


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

# The outcome ramps, one per theme: warm neutrals (never an arm's colour: grey
# untrained, blue SFT, green control, orange post 10), stepped in CIE L* so the
# right answer is the highest-contrast fill on either surface. Light theme: right
# darkest (L* 28), then 44, 59, 71, 82, no call palest (92). Dark theme, reversed:
# right lightest (L* 89), then 74, 59, 48, 39, no call darkest (30; every bar has a
# 1 px mid-grey outline, so its end stays visible). Wrong arguments and wrong
# number of calls, side by side in BFCL's parallel rows, are 15 L* apart in both.
# On no-tool-fits tasks "declined" is right and takes right's colour, and "called
# anyway" (a function where none fits) takes wrong function's: the two never share
# a bar with the call-needing failures, and F3 keys them apart.
_RAMP_ORDER = (RIGHT, WRONG_ARGUMENTS, WRONG_COUNT, WRONG_FUNCTION, UNPARSEABLE, NO_CALL)
_RAMPS = {
    "light": ("#494137", "#70665d", "#978c82", "#b7aca1", "#d5cabf", "#f2e6db"),
    "dark": ("#e9ded2", "#bfb4a9", "#978c82", "#7a7066", "#645a51", "#4e453c"),
}


def outcome_colours(theme: Theme) -> dict[str, str]:
    """Each outcome's fill on this theme's surface (see the ramps above)."""
    out = dict(zip(_RAMP_ORDER, _RAMPS[theme.name], strict=True))
    out[DECLINED] = out[RIGHT]
    out[CALLED_ANYWAY] = out[WRONG_FUNCTION]
    return out


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


# The tool list's solid fill, the same in both themes (the muted grey: 3.5:1 on the light
# surface, 4.8:1 on the dark), on the token bar and behind the tool list in the prompt
# text, so the two read as the same thing; its ink comes from plotting.ink_for.
TOOL_LIST_FILL = "#898781"


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
    label_w = 1.6   # the gutter for two levels of brackets: parts, and the tool list
    chars = int((inner - label_w) / cw)
    excerpt, tools_span, req_span = prompt_excerpt(rt.prompt, chars)
    body_h = mono_height(excerpt, chars, size)
    c1_h = 0.55 + body_h + 0.3 + 0.35  # the prompt, then one token bar
    c1_top = top - 0.35
    card(px0, px1, c1_top, c1_h, "The chat template puts the tool list and the request "
         "into one prompt", sec)
    step(px0 + 0.33, c1_top - 0.27, 1)
    y_text = c1_top - 0.55
    mono(ax, px0 + 0.18, y_text, excerpt, width=chars, size=size, color=sec,
         spans=[(tools_span[0], tools_span[1], TOOL_LIST_FILL),
                (req_span[0], req_span[1], band)],
         bold_spans=False)
    # the right gutter: each part of the prompt under a bracket with its name (the
    # readable view's), and the tool list's own bracket nested inside [tools], nearer the
    # text; turn-marker lines have none. Strokes and labels in the theme's text grey.
    lines = wrap_offsets(excerpt, chars)

    def rows_of(span):
        rows = [i for i, (off, line) in enumerate(lines)
                if off < span[1] and off + len(line) > span[0]]
        return rows[0], rows[-1]

    def bracket(x, r0, r1):
        y0 = y_text - r0 * lh - 0.03
        y1 = y_text - (r1 + 1) * lh + 0.03
        ax.plot([x, x + 0.06, x + 0.06, x], [y0, y0, y1, y1], color=sec, lw=1.0, zorder=4)

    def label(x, y, words):
        for j, w in enumerate(words):
            ax.text(x, y + (len(words) - 1) * 0.1 - j * 0.2, w, ha="left", va="center",
                    fontsize=10, color=sec, zorder=5)

    lx_in = px0 + 0.18 + chars * cw + 0.1     # the tool list's bracket
    lx_out = lx_in + 0.14                     # the parts' brackets
    lx_lab = lx_out + 0.16                    # every label
    t0, t1 = rows_of(tools_span)
    bracket(lx_in, t0, t1)
    label(lx_lab, y_text - (t0 + t1 + 1) / 2 * lh, ["the tool list"])
    for name, span in part_spans(rt.prompt, rt.prompt_exact, chars):
        r0, r1 = rows_of(span)
        bracket(lx_out, r0, r1)
        if name == "tools":  # its name over the rows above the nested tool list
            if t0 <= r0 or t1 > r1:
                raise CaptureError("layout: the tool list is not inside [tools]")
            label(lx_lab, y_text - (r0 + t0) / 2 * lh, [f"[{name}]"])
        elif name == "user":
            label(lx_lab, y_text - (r0 + r1 + 1) / 2 * lh, [f"[{name}]", "the request"])
        else:
            label(lx_lab, y_text - (r0 + r1 + 1) / 2 * lh, [f"[{name}]"])

    # token bars, one scale and one left edge: the whole prompt an outline, the tool list
    # a solid fill inside it (the same encoding in both themes)
    by = c1_top - 0.55 - body_h - 0.3
    bx0 = px0 + 0.18
    bar_w = inner - 2.3
    per = bar_w / rt.prompt_tokens
    bar_h = 0.3
    ax.add_patch(Rectangle((bx0, by - bar_h / 2), rt.prompt_tokens * per, bar_h,
                           facecolor="none", edgecolor=sec, lw=1.1, zorder=4))
    ax.add_patch(Rectangle((bx0, by - bar_h / 2), rt.tool_list_tokens * per, bar_h,
                           facecolor=TOOL_LIST_FILL, edgecolor="none", zorder=3))
    ax.text(bx0 + rt.prompt_tokens * per + 0.1, by, f"{rt.prompt_tokens:,} tokens in the prompt",
            ha="left", va="center", fontsize=10.5, color=ink, zorder=5)
    t = ax.text(bx0 + 0.1, by, f"{rt.tool_list_tokens:,} of them, the tool list", ha="left",
                va="center", fontsize=10.5, color=ink_for(TOOL_LIST_FILL), zorder=5)
    if text_width(fig, t) + 0.2 > rt.tool_list_tokens * per:
        raise CaptureError("layout: the tool list's label is wider than its bar")
    c1_bottom = c1_top - c1_h

    # -- 2: the model writes text -------------------------------------------------
    mchars = int((mx1 - mx0 - 0.36) / cw)
    reply_h = mono_height(rt.reply, mchars, size)
    note2 = [f"it picks {rt.call[0]},", f"one of the {count_tools(rt.prompt)} tools offered"]
    c2_h = 0.55 + reply_h + 0.55 + 0.2 * (len(note2) - 1)
    c2_top = c1_bottom + 1.35
    model_fill = theme.grid
    card(mx0, mx1, c2_top, c2_h, "The model writes text", grey, fill=model_fill)
    step(mx0 + 0.33, c2_top - 0.27, 2)
    mono(ax, mx0 + 0.18, c2_top - 0.55, rt.reply, width=mchars, size=size, color=ink)
    for j, line in enumerate(note2):  # two lines: one is wider than the card
        ax.text(mx0 + 0.18, c2_top - c2_h + 0.25 + 0.2 * (len(note2) - 1 - j), line,
                ha="left", va="center", fontsize=10, color=sec, zorder=5)
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
    notes = call_note(rt.reference[1], args, wrong, rt.exact, rt.execution) if wrong else []
    c3_h = 0.55 + mono_height(parsed, pchars, size) + (0.3 + 0.22 * len(notes) if notes else 0.2)
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
    for j, line in enumerate(notes):
        ax.text(px0 + 0.18, c3_top - c3_h + 0.25 + 0.22 * (len(notes) - 1 - j), line,
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


def call_note(ref: dict, got: dict, differ: list[str], exact: bool,
              execution: bool) -> list[str]:
    """Card 3's note on the arguments that differ from the reference, built from the
    verdicts: wrong ones when execution match fails, harmless ones when it passes."""
    if not differ:
        raise ValueError("no differing arguments to note")
    if exact:
        raise CaptureError("exact match passed, yet arguments differ")
    extra = [k for k in differ if k not in ref]
    if execution:
        what = []
        if extra:
            what.append(f"{join_names(extra)}: not in the reference call")
        rest = [k for k in differ if k in ref]
        if rest:
            what.append(f"{join_names(rest)}: written differently from the reference")
        return ["; ".join(what) + ". Exact match says no, execution match yes.",
                "The program parses and runs the call as written, right or wrong."]
    return [f"{join_names(differ)}: not what the request asked for. "
            "The program parses and runs the call as written."]


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


def compared_label(compared: str) -> str:
    """The note under an execution-match pill: a search's line shortened to its counts
    (``documents found: 1 vs 4 (different)`` -> ``documents found: 1 vs 4``);
    the calculator's and the calendar's as printed."""
    m = re.fullmatch(r"(\w+) found: (\d+) vs (\d+) \((?:the same|different)\)", compared)
    if m:
        return f"{m.group(1)} found: {m.group(2)} vs {m.group(3)}"
    if "found" in compared:
        raise CaptureError(f"execution match compared: {compared!r} is not in the printed format")
    return compared


def figure_two_checks(theme: Theme, calls: list[CheckedCall], out: Path | None = None) -> Path:
    two_checks_claims(calls)
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    band = tint(theme)
    yes_fill = outcome_colours(theme)[RIGHT]
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
        """'yes' a filled pill with a 1 px outline in the text colour; 'no' outlined."""
        w, h = 0.78, 0.36
        box(ax, x - w / 2, y - h / 2, w, h, ink if ok else muted,
            fill=yes_fill if ok else "none", lw=0.72 if ok else 1.3, r=0.12)
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
        ax.text(run_x, mid - 0.25, compared_label(c.compared), ha="center", va="center",
                fontsize=9.5, color=sec)
        y -= 0.3
    ax.plot([tx, W - 0.2], [y, y], color=theme.grid, lw=1.0, zorder=1)
    ax.set_ylim(y - 0.1, top + 0.3)
    fig.set_size_inches(W, top + 0.4 - y)
    return save_both(fig, "two-checks", theme, out)


# --------------------------------------------------------------------------
# F3: the untrained model's outcomes, per category
# --------------------------------------------------------------------------


def outcome_groups(bfcl: Outcomes, ours: Outcomes) -> list[tuple[str, list[tuple[str, OutcomeRow]]]]:
    """F3's groups, in section 3's order: our tasks, then BFCL, then no tool fits
    (ours above BFCL irrelevance)."""
    return [
        ("Our tasks, a call is needed", [(r.label, r) for r in ours.rows if r.needs_call]),
        ("BFCL, a call is needed", [(r.label, r) for r in bfcl.rows if r.needs_call]),
        ("No tool fits: declining is right",
         [(f"ours: {r.label}", r) for r in ours.rows if not r.needs_call]
         + [(f"BFCL {r.label}", r) for r in bfcl.rows if not r.needs_call]),
    ]


def figure_untrained_outcomes(theme: Theme, bfcl: Outcomes, ours: Outcomes,
                              out: Path | None = None) -> Path:
    if bfcl.model != ours.model:
        raise CaptureError(f"the BFCL capture scores {bfcl.model}, ours {ours.model}")
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    colours = outcome_colours(theme)
    groups = outcome_groups(bfcl, ours)
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
            ax.add_patch(Rectangle((x, y - 0.09), 0.26, 0.18, facecolor=colours[o],
                                   edgecolor=muted, lw=0.6, zorder=3))
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
                fill = colours[o]
                ax.add_patch(Rectangle((x, y - bar_h / 2), w, bar_h, facecolor=fill,
                                       edgecolor=theme.surface, lw=1.0, zorder=3))
                t = f"{n}"
                if w >= 0.14 * len(t) + 0.1:
                    ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=9.5,
                            color=ink_for(fill), zorder=5)
                x += w
            # the whole bar outlined, so a fill close to the surface keeps its edge
            ax.add_patch(Rectangle((x0, y - bar_h / 2), x1 - x0, bar_h, facecolor="none",
                                   edgecolor=muted, lw=0.72, zorder=4))
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
        ax.text(sb, y + 0.62, f"assistant: {mn + sn} tokens, {mn} masked (header line, final newline)",
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
# F5: ToolACE's rows, down to the training rows (toolace-stats + sft-data)
# --------------------------------------------------------------------------

# sft-data's labels (sft.ROW_TYPES, sft.DROP_REASONS), as printed.
R_CALL_LABEL = "2+ tools, with a call"
R_NO_CALL_LABELS = ("no call, 2+ tools", "no call, 1 tool", "no call, empty tool list")
R_OUT_LABEL = "1 tool, with a call (left out)"
DROP_LABELS = {"an empty assistant turn": "with an empty reply turn",
               "a bracket-syntax call in a text turn": "with a call written in a text reply"}
NOT_JSON_LABEL = "tool list not JSON (format-variation row)"
SUB_NO_CALL = "no call (the three rows above)"
SUB_TRAINS = "all but the left-out rows"


@dataclass(frozen=True)
class Funnel:
    total: int
    chinese: int
    english: int
    not_json: int
    converted: int
    left_out: int  # one tool and a call
    offered: int   # rows of a training type, before the drops
    dropped: tuple[tuple[str, int], ...]  # (figure wording, rows), in sft-data's order
    each_arm: int
    with_call: int  # in each arm
    no_call: int    # in each arm, the same rows in both
    no_call_parts: tuple[int, ...] = ()  # no_call by tools offered: 2+, one, none


def parse_funnel(stats: str, sft_text: str) -> Funnel:
    """The row counts from ``toolace-stats`` and ``sft-data``, each subtraction checked."""
    f = section(stats, "The file")
    total = to_int(kv(f, "rows", "toolace-stats, The file"), "rows")
    lang = section(stats, "Language")
    chinese = to_int(kv(lang, "Chinese rows (dropped)", "Language"), "Chinese rows")
    english = to_int(kv(lang, "English rows", "Language"), "English rows")
    conv = section(stats, "English rows that do not convert, by reason")
    reasons = table(conv, ["reason", "rows"], "the conversion failures")
    if [r[0] for r in reasons] != [NOT_JSON_LABEL]:
        raise CaptureError(f"conversion failures {[r[0] for r in reasons]}: the funnel draws "
                           "only the not-JSON drop")
    not_json = to_int(reasons[0][1], "not JSON rows")
    converted = to_int(kv(conv, "rows converted", "the conversion"), "rows converted")

    lines = section(sft_text, "Which rows train")
    rows = table(lines, ["row type", "converted", "dropped", "each arm"], "Which rows train")
    by = {r[0]: r for r in rows}
    want = [R_CALL_LABEL, *R_NO_CALL_LABELS, R_OUT_LABEL, "total"]
    if [r[0] for r in rows] != want:
        raise CaptureError(f"Which rows train: rows {[r[0] for r in rows]}, expected {want}")

    def n(label, col):
        return to_int(by[label][col], f"Which rows train, {label}")

    if by[R_OUT_LABEL][2] != "-" or n(R_OUT_LABEL, 3) != 0:
        raise CaptureError("the left-out rows are not left out")
    drops = table(lines, ["reason", "rows"], "the dropped rows")
    if drops[-1][0] != "total" or [r[0] for r in drops[:-1]] != list(DROP_LABELS):
        raise CaptureError(f"dropped rows: reasons {[r[0] for r in drops]}")
    dropped = tuple((DROP_LABELS[r[0]], to_int(r[1], r[0])) for r in drops[:-1])
    fn = Funnel(
        total=total, chinese=chinese, english=english, not_json=not_json, converted=converted,
        left_out=n(R_OUT_LABEL, 1),
        offered=n("total", 1) - n(R_OUT_LABEL, 1),
        dropped=dropped, each_arm=n("total", 3),
        with_call=n(R_CALL_LABEL, 3), no_call=sum(n(lab, 3) for lab in R_NO_CALL_LABELS),
        no_call_parts=tuple(n(lab, 3) for lab in R_NO_CALL_LABELS))
    checks = [
        (fn.total - fn.chinese == fn.english, "rows - Chinese != English"),
        (fn.english - fn.not_json == fn.converted, "English - not JSON != converted"),
        (n("total", 1) == fn.converted, "sft-data's converted rows != toolace-stats'"),
        (fn.offered - sum(k for _, k in dropped) == fn.each_arm, "offered - dropped != each arm"),
        (sum(k for _, k in dropped) == to_int(drops[-1][1], "total") == n("total", 2),
         "the drop counts disagree"),
        (fn.with_call + fn.no_call == fn.each_arm, "with a call + no call != each arm"),
    ]
    # The printed subtotals are what the figure's labels quote (8,518 offered; no-call rows each arm).
    subs = {r[0]: r for r in table(lines, ["subtotal", "converted", "dropped", "each arm"],
                                   "the subtotals")}
    if list(subs) != [SUB_NO_CALL, SUB_TRAINS]:
        raise CaptureError(f"subtotals: rows {list(subs)}")
    checks += [
        (to_int(subs[SUB_TRAINS][1], SUB_TRAINS) == fn.offered, "printed subtotal != offered"),
        (to_int(subs[SUB_TRAINS][3], SUB_TRAINS) == fn.each_arm, "printed subtotal != each arm"),
        (to_int(subs[SUB_NO_CALL][3], SUB_NO_CALL) == fn.no_call, "printed subtotal != no call"),
    ]
    for ok, what in checks:
        if not ok:
            raise CaptureError(f"funnel: {what}")
    return fn


FUNNEL_GREY = {"light": ("#c9c6bd", "#9f9b91"), "dark": ("#6f6c66", "#57544f")}
# The one-tool no-call rows: the no-call grey under a hatch, no new colour (blue and green
# are the arms'). The hatch is darker than the grey on light, lighter on dark.
FUNNEL_HATCH = {"light": "#6f6b62", "dark": "#b3afa6"}
NO_CALL_TOOLS = ("2+ tools", "1 tool", "none")  # R_NO_CALL_LABELS' order, as the bracket says


def figure_toolace_funnel(theme: Theme, fn: Funnel, out: Path | None = None) -> Path:
    """Left-aligned bars on one scale; each drop a hollow outline to the bar's right."""
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    blue, green = theme.series[0], theme.series[2]
    grey, grey2 = FUNNEL_GREY[theme.name]
    W = 12.8
    fig, ax = canvas(W, 8.0)
    x0, x1 = 3.35, 8.6          # bar lane: x1 is 11,300 rows
    side = x1 + 0.25            # side labels for the drops
    per = (x1 - x0) / fn.total
    bar_h, step = 0.46, 0.86
    top = 7.6

    def bar(y, n, fill):
        ax.add_patch(Rectangle((x0, y - bar_h / 2), n * per, bar_h, facecolor=fill,
                               edgecolor="none", zorder=3))

    def count_in(x, y, text, fill, ha="left"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=10.5, color=ink_for(fill), zorder=5)

    def hollow(y, start, n, lines):
        """A dropped slice: outline only, from where the bar ends, and its label."""
        ax.add_patch(Rectangle((x0 + start * per, y - bar_h / 2), max(n * per, 0.025), bar_h,
                               facecolor="none", edgecolor=muted, lw=0.9, linestyle=(0, (3, 2)),
                               zorder=3))
        for j, line in enumerate(lines):
            ax.text(side, y + 0.11 * (len(lines) - 1) - 0.22 * j, line, ha="left",
                    va="center", fontsize=10, color=sec if j else ink, zorder=5)

    hatch = FUNNEL_HATCH[theme.name]
    one_tool = fn.no_call_parts[1]

    def no_call_segments(y):
        """The no-call rows split by tools offered, left to right: 2+, one (hatched), none;
        each labelled inside if its count fits, else above the bar."""
        start = fn.with_call
        spans, above = [], []
        for k, n in enumerate(fn.no_call_parts):
            ax.add_patch(Rectangle((x0 + start * per, y - bar_h / 2), n * per, bar_h,
                                   facecolor=grey, edgecolor="none", zorder=3,
                                   hatch="////" if k == 1 else None, hatchcolor=hatch))
            ax.add_patch(Rectangle((x0 + start * per, y - bar_h / 2), n * per, bar_h,
                                   facecolor="none", edgecolor=theme.surface, lw=1.0, zorder=4))
            mid = x0 + (start + n / 2) * per
            t = ax.text(mid, y, f"{n:,}", ha="center", va="center", fontsize=9.2,
                        color=ink_for(grey), zorder=6,
                        bbox=dict(boxstyle="square,pad=0.12", facecolor=grey, edgecolor="none"))
            if text_width(fig, t) + 0.04 > n * per:
                above.append((t, mid))
            spans.append((x0 + start * per, x0 + (start + n) * per))
            start += n
        # counts too wide for their segment go above the bar, spread apart (fan), each
        # with a leader to its segment when it had to move
        if above:
            gap = max(text_width(fig, t) for t, _ in above) + 0.1
            ya = y + bar_h / 2 + 0.17
            for (t, mid), xl in zip(above, fan([m for _, m in above], gap), strict=True):
                t.set_position((xl, ya))
                t.set_color(sec)
                t.set_bbox(None)
                if abs(xl - mid) > 0.03:
                    ax.plot([mid, xl], [y + bar_h / 2 + 0.01, ya - 0.08], color=sec, lw=0.7,
                            zorder=5)
        return spans

    def title(y, text, sub=None):
        ax.text(x0 - 0.15, y + (0.1 if sub else 0), text, ha="right", va="center",
                fontsize=10.8, color=ink, fontweight="bold")
        if sub:
            ax.text(x0 - 0.15, y - 0.15, sub, ha="right", va="center", fontsize=9.8, color=sec)

    ax.text(0.2, top + 0.45, "From ToolACE's rows to the rows each arm trains on",
            ha="left", va="center", fontsize=13.5, fontweight="bold", color=ink)
    ax.text(x0 + (x1 - x0) / 2, top - 0.05, "bar length = number of rows, one scale",
            ha="center", va="center", fontsize=9.8, color=muted)

    y = top - 0.6
    title(y, "ToolACE on Hugging Face")
    bar(y, fn.total, grey)
    count_in(x0 + 0.12, y, f"{fn.total:,} rows", grey)

    y -= step
    title(y, "English")
    bar(y, fn.english, grey)
    count_in(x0 + 0.12, y, f"{fn.english:,}", grey)
    hollow(y, fn.english, fn.chinese, [f"{fn.chinese:,} Chinese rows dropped"])

    y -= step
    title(y, "converted to Qwen's format")
    bar(y, fn.converted, grey)
    count_in(x0 + 0.12, y, f"{fn.converted:,}", grey)
    hollow(y, fn.converted, fn.not_json, [f"{fn.not_json:,} tool lists not in JSON, dropped"])

    y -= step
    title(y, "offer a choice, or make no call")
    bar(y, fn.offered, grey)
    count_in(x0 + 0.12, y, f"{fn.offered:,}", grey)
    hollow(y, fn.offered, fn.left_out,
           [f"{fn.left_out:,} rows with one tool and a call, left out:",
            "the control would always draw the right tool"])

    y -= step
    n_drop = sum(k for _, k in fn.dropped)
    title(y, "without broken turns")
    bar(y, fn.with_call, grey2)
    count_in(x0 + 0.12, y, f"{fn.with_call:,} with a call", grey2)
    spans = no_call_segments(y)
    hollow(y, fn.each_arm, n_drop,
           [f"{n_drop:,} rows dropped:",
            ", ".join(f"{k:,} {label}" for label, k in fn.dropped)])
    # a bracket under the three no-call segments, and what they offer
    ya, yb = y - bar_h / 2 - 0.05, y - bar_h / 2 - 0.12
    ax.plot([spans[0][0], spans[0][0], spans[-1][1], spans[-1][1]], [ya, yb, yb, ya],
            color=sec, lw=1.0)
    offers = [f"{n:,} offer {t}" if t != "none" else f"{n:,} offer none"
              for n, t in zip(fn.no_call_parts, NO_CALL_TOOLS, strict=True)]
    ax.text((spans[0][0] + spans[-1][1]) / 2, yb - 0.17, "no call: " + ", ".join(offers),
            ha="center", va="center", fontsize=9.6, color=sec)

    # the two arms, the same rows; only the call turns differ
    y -= step + 0.6
    ax.plot([0.2, W - 0.2], [y + step / 2 + 0.12, y + step / 2 + 0.12], color=theme.grid, lw=1.0)
    for label, call, colour in (("real arm", "ToolACE's call", blue),
                                ("control", "a random call", green)):
        title(y, f"{label}: {fn.each_arm:,}")
        bar(y, fn.with_call, colour)
        count_in(x0 + 0.12, y, f"{fn.with_call:,} with {call}", colour)
        no_call_segments(y)
        y -= step * 0.8 + 0.12
    # one bracket across both arms' no-call segments
    yb_top, yb_bot = y + 2 * (step * 0.8 + 0.12) + bar_h / 2, y + step * 0.8 + 0.12 - bar_h / 2
    bx = x0 + fn.each_arm * per + 0.1
    ax.plot([bx, bx + 0.07, bx + 0.07, bx], [yb_top, yb_top, yb_bot, yb_bot], color=sec, lw=1.0)
    ax.text(bx + 0.2, (yb_top + yb_bot) / 2 + 0.12, "no-call rows: the same rows in both arms",
            ha="left", va="center", fontsize=10, color=sec)
    ax.text(bx + 0.2, (yb_top + yb_bot) / 2 - 0.12, f"the {one_tool:,} hatched rows are the "
            "only one-tool rows left (§5)", ha="left", va="center", fontsize=10, color=sec)
    ax.text(x0, y + 0.15, "Every row is cut after its first call turn.", ha="left",
            va="center", fontsize=10, color=sec)
    bottom = y - 0.15
    ax.set_ylim(bottom, top + 0.8)
    fig.set_size_inches(W, top + 0.8 - bottom)
    return save_both(fig, "toolace-funnel", theme, out)


# --------------------------------------------------------------------------
# F5: section 5's effects (sft-report, r5d)
# --------------------------------------------------------------------------

R5D_TITLE = "The effects: three paired differences each (r5d)"
DIFF_HEADERS = ["difference", "points", "95% interval", "excludes 0"]
DIFF_LABELS = ("SFT - untrained", "control - untrained", "SFT - control")
# The four effects F5 draws (r5d's names), in the post's order; "right" is not one.
F5_EFFECTS = ("calling when a tool fits", "filling", "declining when none fits", "choosing")


@dataclass(frozen=True)
class Diff:
    label: str
    points: float
    lo: float
    hi: float
    excludes_zero: bool


@dataclass(frozen=True)
class EffectBlock:
    name: str
    harm_check: bool
    primary: int | None
    variable: str
    on: str
    n: int
    shares: dict[str, float]  # arm -> percent
    diffs: tuple[Diff, ...]


def pts(x: float) -> str:
    """A difference as ``effects.points`` prints it: signed, one decimal, ``0.0`` unsigned."""
    return "0.0" if round(x, 1) == 0 else f"{x:+.1f}"


def signed(cell: str, what: str) -> float:
    """``+12.4`` / ``-0.5`` / ``0.0`` as printed by ``effects.points``."""
    return float(match(r"[+-]?\d+\.\d", cell, what).group(0))


def parse_effects(text: str) -> list[EffectBlock]:
    """Every block of r5d, each difference checked: its point inside its interval and
    "excludes 0" agreeing with the interval."""
    lines = section(text, R5D_TITLE)
    out = []
    for i, line in enumerate(lines):
        m = re.fullmatch(r"  (\S.*?)(?:  \[primary (\d)\])?", line)
        if not m or i + 3 >= len(lines) or not lines[i + 1].startswith("    1 = "):
            continue
        title = m.group(1)
        harm = title.endswith(" (harm check only)")
        name = title.removesuffix(" (harm check only)")
        where = f"r5d, {name}"
        on = match(r"    on (.+), (\d+) tasks", lines[i + 2], f"{where}, on")
        sh = match(r"    untrained (\d+\.\d)%   SFT (\d+\.\d)%   control (\d+\.\d)%   "
                   r"\(seed means\)", lines[i + 3], f"{where}, shares")
        rows = table(lines, DIFF_HEADERS, where, start=i + 4)
        if [r[0] for r in rows] != list(DIFF_LABELS):
            raise CaptureError(f"{where}: differences {[r[0] for r in rows]}")
        diffs = []
        for r in rows:
            iv = match(r"\[([+-]?\d+\.\d), ([+-]?\d+\.\d)\]", r[2], f"{where}, {r[0]}")
            d = Diff(r[0], signed(r[1], where), float(iv.group(1)), float(iv.group(2)),
                     yes_no(r[3], f"{where}, {r[0]}"))
            if not d.lo <= d.points <= d.hi:
                raise CaptureError(f"{where}, {r[0]}: {d.points} outside [{d.lo}, {d.hi}]")
            if d.excludes_zero != (d.lo > 0 or d.hi < 0):
                raise CaptureError(f"{where}, {r[0]}: 'excludes 0' disagrees with the interval")
            diffs.append(d)
        out.append(EffectBlock(name, harm, int(m.group(2)) if m.group(2) else None,
                               lines[i + 1][len("    1 = "):], on.group(1), int(on.group(2)),
                               dict(zip(("untrained", "SFT", "control"),
                                        map(float, sh.groups()), strict=True)),
                               tuple(diffs)))
    if not out:
        raise CaptureError("r5d has no effect blocks")
    return out


def f5_blocks(blocks: list[EffectBlock]) -> list[EffectBlock]:
    """F5's rows: the four effects, in :data:`F5_EFFECTS` order, each set as r5d prints."""
    got = [b for e in F5_EFFECTS for b in blocks if b.name == e]
    missing = [e for e in F5_EFFECTS if not any(b.name == e for b in got)]
    if missing:
        raise CaptureError(f"r5d lacks {missing}")
    return got


def primary_summary(blocks: list[EffectBlock]) -> tuple[int, EffectBlock]:
    """How many distinct primaries r5d names, and its block named ``right`` (a primary,
    not one of F5's failure shares)."""
    primaries = {b.primary for b in blocks if b.primary}
    right = [b for b in blocks if b.name == "right"]
    if len(right) != 1 or right[0].primary is None:
        raise CaptureError(f"r5d: expected one primary block named 'right', found {len(right)}")
    return len(primaries), right[0]


def set_label(on: str) -> str:
    """r5d's ``on`` text, shortened for a row label."""
    return {"ours, tasks that need a call": "ours, a call is needed",
            "BFCL call items, pooled": "BFCL call items",
            "ours, no tool fits": "ours, no tool fits",
            "BFCL irrelevance": "BFCL irrelevance"}.get(on, on)


def figure_effects(theme: Theme, blocks: list[EffectBlock], model: str,
                   out: Path | None = None) -> Path:
    """Each effect and set a row of three paired differences with their intervals.
    ``model``: the report's model, named in the title (the figure is one model's)."""
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    blue, green = theme.series[0], theme.series[2]
    rows = f5_blocks(blocks)
    W = 12.8
    fig, ax = canvas(W, 10.0)
    x0, x1 = 4.2, 10.0          # the points axis
    lo_lim, hi_lim = -70, 70
    for b in rows:
        for d in b.diffs:
            if d.lo < lo_lim or d.hi > hi_lim:
                raise CaptureError(f"{b.name}: {d.label} [{d.lo}, {d.hi}] off the axis")

    def X(v):
        return x0 + (x1 - x0) * (v - lo_lim) / (hi_lim - lo_lim)

    styles = {"SFT - untrained": (blue, "o", True), "control - untrained": (green, "s", True),
              "SFT - control": (ink, "D", False)}
    top = 9.5
    ax.text(0.2, top + 0.3, f"What SFT changed on {model}: each effect as three paired "
            "differences", ha="left", va="center", fontsize=13.5, fontweight="bold", color=ink)
    ax.text(0.2, top - 0.05, "Each effect is the share of tasks with one failure outcome, so "
            "left of 0 is fewer failures. Bars: 95% intervals (paired bootstrap over tasks).",
            ha="left", va="center", fontsize=10, color=sec)
    n_primary, right = primary_summary(blocks)
    right_set = right.on.split(",")[0].split()[0]  # "BFCL call items, pooled" -> "BFCL"
    ax.text(0.2, top - 0.3, f"primary N: one of the {number_word(n_primary)} comparisons fixed "
            f"before the run ({right.primary}, {right_set} {right.name}, is not a failure share "
            "and isn't shown).", ha="left", va="center", fontsize=10, color=sec)
    # legend
    lx = x0
    ly = top - 0.7
    for label, (c, mk, filled) in styles.items():
        ax.plot([lx, lx + 0.3], [ly, ly], color=c, lw=1.6)
        ax.plot(lx + 0.15, ly, marker=mk, ms=7, color=c, mfc=c if filled else theme.surface,
                mew=1.4)
        t = ax.text(lx + 0.42, ly, label.replace(" - ", " − "), ha="left", va="center",
                    fontsize=10.5, color=sec)
        lx += 0.42 + text_width(fig, t) + 0.45
    sub = 0.24
    y = ly - 0.45
    y_axis_top = y + 0.15
    prev = None
    for b in rows:
        if b.name != prev:
            y -= 0.12 if prev else 0
            title = b.name + (" (a harm check)" if b.harm_check else "")
            ax.text(0.2, y, title, ha="left", va="center", fontsize=11.2, fontweight="bold",
                    color=ink)
            ax.text(0.2, y - 0.22, f"counts: {b.variable}", ha="left", va="center", fontsize=9.6,
                    color=muted)
            prev = b.name
            y -= 0.5
        ax.text(x0 - 1.35, y, f"{set_label(b.on)} ({b.n:,})", ha="right", va="center",
                fontsize=10.3, color=ink)
        if b.primary:
            ax.text(x0 - 1.35, y - 0.2, f"primary {b.primary}", ha="right", va="center",
                    fontsize=9.2, color=muted)
        shares = "   ".join(f"{a} {v:.1f}%" for a, v in b.shares.items())
        for j, d in enumerate(b.diffs):
            yy = y + sub - j * sub
            c, mk, filled = styles[d.label]
            ax.plot([X(d.lo), X(d.hi)], [yy, yy], color=c, lw=1.6, zorder=3,
                    solid_capstyle="butt")
            ax.plot(X(d.points), yy, marker=mk, ms=6.5, color=c,
                    mfc=c if filled else theme.surface, mew=1.3, zorder=4)
            ax.text(x1 + 0.12, yy, f"{pts(d.points)}  [{pts(d.lo)}, {pts(d.hi)}]",
                    ha="left", va="center", fontsize=9.4, color=c if filled else sec,
                    family=MONO)
        ax.text(x0 - 0.1, y + sub * 1.9, shares, ha="left", va="bottom", fontsize=9.2,
                color=muted)
        ax.plot([0.2, W - 0.2], [y - sub * 1.6, y - sub * 1.6], color=theme.grid, lw=0.8,
                zorder=1)
        y -= 1.05
    y_axis = y + 1.05 - sub * 1.6
    # zero line and the axis
    ax.plot([X(0), X(0)], [y_axis, y_axis_top], color=theme.axis, lw=1.0, zorder=2)
    for v in range(lo_lim + 10, hi_lim, 20):
        ax.plot([X(v), X(v)], [y_axis, y_axis - 0.06], color=theme.axis, lw=0.8)
        ax.text(X(v), y_axis - 0.2, f"{v:+d}" if v else "0", ha="center", va="center",
                fontsize=9.5, color=muted)
    ax.plot([X(lo_lim), X(hi_lim)], [y_axis, y_axis], color=theme.axis, lw=0.8)
    ax.text(X(0), y_axis - 0.46, "difference, percentage points (left of 0: fewer failures; "
            "right: more)", ha="center", va="center", fontsize=10, color=sec)
    ax.text(x1 + 0.12, y_axis_top + 0.05, "points  [95% interval]", ha="left", va="bottom",
            fontsize=9.4, color=muted)
    bottom = y_axis - 0.7
    ax.set_ylim(bottom, top + 0.6)
    fig.set_size_inches(W, top + 0.6 - bottom)
    return save_both(fig, "sft-effects", theme, out)


# --------------------------------------------------------------------------
# F6: group movements against the noise floor (sft-report, r6a)
# --------------------------------------------------------------------------

R6A_FLOOR_TITLE = "r6a: untrained -> untrained again, the noise floor"
R6A_ARM_TITLE = "r6a: untrained -> {arm}, task by task"
MOVEMENTS = ("mixed -> all right", "all wrong -> mixed", "mixed -> all wrong",
             "all wrong -> all wrong")


@dataclass(frozen=True)
class Movement:
    name: str
    floor: float           # percent
    floor_lo: float
    floor_hi: float
    sft: float
    sft_beyond: Diff
    control: float
    control_beyond: Diff


@dataclass(frozen=True)
class Movements:
    n: int
    rows: tuple[Movement, ...]
    seeds: int          # SFT and the control: means of this many seeds (Setup)
    samples: int        # answers per task (r6a's samples line)
    temperature: str    # as printed, e.g. "1"


R6A_SETUP_TITLE = r"Groups of (\d+) on .* \(r6a\)"


def r6a_sampling(text: str) -> tuple[int, str]:
    """r6a's ``samples`` line (``8 per task at T=1, seed 0``): the group size and the
    temperature as printed; the section title's group size must agree."""
    hits = section_matching(text, R6A_SETUP_TITLE)
    if len(hits) != 1:
        raise CaptureError(f"expected one r6a setup section, found {len(hits)}")
    m, body = hits[0]
    s = match(r"(\d+) per task at T=(\d+(?:\.\d+)?), seed \d+", kv(body, "samples", "r6a"),
              "r6a, samples")
    if s.group(1) != m.group(1):
        raise CaptureError(f"r6a: groups of {m.group(1)} but {s.group(1)} samples per task")
    return int(s.group(1)), s.group(2)


def _pct(cell: str, what: str) -> float:
    return float(match(r"(\d+\.\d)%", cell, what).group(1))


def parse_movements(text: str) -> Movements:
    """r6a's movement tables on the tasks that need a call: the floor's, SFT's and the
    control's, each arm's floor column checked against the floor's own table."""
    lines = section(text, R6A_FLOOR_TITLE)
    hits = [r for r in (re.split(r"\s{2,}", x.strip()) for x in lines)
            if r and re.fullmatch(r"movement, need a call", r[0])]
    if len(hits) != 1:
        raise CaptureError("r6a floor: no 'movement, need a call' table")
    head = hits[0]
    n = to_int(match(r"share of (\d+)", head[1], "r6a floor header").group(1), "floor n")
    floor_rows = table(lines, head, "r6a floor")
    floor = {}
    for r in floor_rows:
        iv = match(r"\[(\d+\.\d), (\d+\.\d)\]", r[2], f"r6a floor, {r[0]}")
        floor[r[0]] = (_pct(r[1], r[0]), float(iv.group(1)), float(iv.group(2)))
    arms = {}
    for arm in ("SFT", "control"):
        where = f"r6a {arm}"
        body = section(text, R6A_ARM_TITLE.format(arm=arm))
        head = [f"movement, need a call ({n})", arm, "floor", "beyond", "95% interval"]
        got = {}
        for r in table(body, head, where):
            iv = match(r"\[([+-]?\d+\.\d), ([+-]?\d+\.\d)\]", r[4], f"{where}, {r[0]}")
            share, fl = _pct(r[1], f"{where}, {r[0]}"), _pct(r[2], f"{where}, {r[0]}")
            beyond = signed(r[3], f"{where}, {r[0]}")
            if r[0] not in floor or fl != floor[r[0]][0]:
                raise CaptureError(f"{where}, {r[0]}: floor {fl} is not the floor table's")
            if abs(share - fl - beyond) > 0.15:
                raise CaptureError(f"{where}, {r[0]}: {share} - {fl} != {beyond}")
            lo, hi = float(iv.group(1)), float(iv.group(2))
            if not lo <= beyond <= hi:
                raise CaptureError(f"{where}, {r[0]}: {beyond} outside its interval")
            got[r[0]] = (share, Diff(f"{arm} - floor", beyond, lo, hi, lo > 0 or hi < 0))
        arms[arm] = got
    if not (list(floor) == list(arms["SFT"]) == list(arms["control"]) == list(MOVEMENTS)):
        raise CaptureError(f"r6a movements {list(floor)}, expected {list(MOVEMENTS)}")
    return Movements(n, tuple(
        Movement(m, *floor[m], arms["SFT"][m][0], arms["SFT"][m][1], arms["control"][m][0],
                 arms["control"][m][1]) for m in MOVEMENTS), seed_count(text), *r6a_sampling(text))


def largest_beyond(mv: Movements, arm: str = "sft") -> Movement:
    """The movement with the largest gain beyond the floor whose interval excludes 0."""
    key = f"{arm}_beyond"
    ok = [m for m in mv.rows if getattr(m, key).excludes_zero]
    if not ok:
        raise CaptureError(f"no {arm} movement is beyond the floor")
    return max(ok, key=lambda m: getattr(m, key).points)


def panel_title(text: str) -> str:
    """A movements panel's title from its report: the model, and for the base profile
    "(base), read tag-free" (the reading the post uses for the base model)."""
    head = text.split("\n")[1]
    model = report_model(text)
    if "--profile base" not in head:
        return model
    reading = kv(section(text, "Setup"), "reading", "Setup")
    if not reading.startswith("tag-free"):
        raise CaptureError("the base model's movements are drawn on the tag-free reading")
    return f"{model} (base), read tag-free"


def figure_movements(theme: Theme, panels: list[tuple[str, Movements]],
                     out: Path | None = None) -> Path:
    """One panel per model, stacked on one share axis (0-100%): per movement, three
    bars, the floor (with its interval), SFT and the control; beside them each arm's
    difference from the floor, the largest beyond the floor in bold."""
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    blue, green, grey = theme.series[0], theme.series[2], theme.muted
    ns = {mv.n for _, mv in panels}
    if len(ns) != 1:
        raise CaptureError(f"the panels count different tasks: {sorted(ns)}")
    n = ns.pop()
    seeds = same((mv.seeds for _, mv in panels), "seed count")
    samples = same((mv.samples for _, mv in panels), "samples per task")
    temperature = same((mv.temperature for _, mv in panels), "temperature")
    W = 12.8
    fig, ax = canvas(W, 14.0)
    x0, x1 = 3.0, 7.9
    top_share = 100.0
    if max(max(m.sft, m.control, m.floor_hi) for _, mv in panels for m in mv.rows) > top_share:
        raise CaptureError("a movement over 100% of tasks")

    def X(v):
        return x0 + (x1 - x0) * v / top_share

    top = 13.4
    ax.text(0.2, top + 0.35, f"Where the {n} tasks that need a call move, from the untrained "
            "model's group to the trained model's", ha="left", va="center", fontsize=13.5,
            fontweight="bold", color=ink)
    ax.text(0.2, top, f"Groups of {samples} answers at T = {temperature}: all wrong, mixed, "
            "all right. "
            "The noise floor is the untrained model sampled again, no training.",
            ha="left", va="center", fontsize=10, color=sec)
    bar_h, gap = 0.22, 0.04
    arms = (("noise floor (sampled again)", grey), (f"SFT, mean of {seeds} seeds", blue),
            (f"control, mean of {seeds} seeds", green))
    lx, ly = x0, top - 0.45
    for label, c in arms:
        ax.add_patch(Rectangle((lx, ly - 0.09), 0.3, 0.18, facecolor=c, edgecolor="none"))
        t = ax.text(lx + 0.4, ly, label, ha="left", va="center", fontsize=10.5, color=sec)
        lx += 0.4 + text_width(fig, t) + 0.45
    cx_s, cx_c = 8.55, 10.75   # beyond-the-floor columns
    y = ly - 0.55
    ax.text(cx_s, y, "SFT − floor", ha="left", va="center", fontsize=10.3,
            fontweight="bold", color=blue)
    ax.text(cx_c, y, "control − floor", ha="left", va="center", fontsize=10.3,
            fontweight="bold", color=green)
    ax.text(cx_s, y - 0.2, "points [95% interval]", ha="left", va="center", fontsize=9,
            color=muted)
    ax.text(cx_c, y - 0.2, "points [95% interval]", ha="left", va="center", fontsize=9,
            color=muted)
    y -= 0.4
    reading = {"mixed -> all right": "SFT taught it; less left for a reward",
               "all wrong -> mixed": "new signal for a reward",
               "mixed -> all wrong": "damage",
               "all wrong -> all wrong": "neither model gets it in 8 tries"}
    row_h = 1.0
    spans = []  # each panel's (top, bottom) for the gridlines
    for title, mv in panels:
        best = largest_beyond(mv)
        ax.text(0.2, y, title, ha="left", va="center", fontsize=12, fontweight="bold",
                color=ink)
        ax.plot([0.2, W - 0.2], [y - 0.2, y - 0.2], color=theme.axis, lw=0.9, zorder=1)
        y -= 0.75
        y_top = y + 0.45
        for m in mv.rows:
            label = m.name.replace(" -> ", " to ")
            ax.text(x0 - 0.15, y + 0.1, label, ha="right", va="center", fontsize=11,
                    fontweight="bold" if m is best else "normal", color=ink)
            ax.text(x0 - 0.15, y - 0.13, reading[m.name], ha="right", va="center",
                    fontsize=9.2, color=muted)
            for k, ((_, c), v) in enumerate(zip(arms, (m.floor, m.sft, m.control),
                                                strict=True)):
                yy = y + (bar_h + gap) - k * (bar_h + gap)
                ax.add_patch(Rectangle((x0, yy - bar_h / 2), max(X(v) - x0, 0.012), bar_h,
                                       facecolor=c, edgecolor="none", zorder=3))
                if k == 0:  # the floor's interval
                    ax.plot([X(m.floor_lo), X(m.floor_hi)], [yy, yy], color=ink, lw=1.1,
                            zorder=4)
                    for e in (m.floor_lo, m.floor_hi):
                        ax.plot([X(e), X(e)], [yy - 0.06, yy + 0.06], color=ink, lw=1.1,
                                zorder=4)
                ax.text(max(X(v), X(m.floor_hi) if k == 0 else X(v)) + 0.08, yy,
                        f"{v:.1f}%", ha="left", va="center", fontsize=9.2, color=sec, zorder=5)
            for cx, d, c in ((cx_s, m.sft_beyond, blue), (cx_c, m.control_beyond, green)):
                ax.text(cx, y, f"{pts(d.points)}  [{pts(d.lo)}, {pts(d.hi)}]", ha="left",
                        va="center", fontsize=9.6, family=MONO, color=ink if d.excludes_zero
                        else muted, fontweight="bold" if (m is best and c == blue) else "normal")
            y -= row_h
            if m is not mv.rows[-1]:
                ax.plot([x0 - 2.8, W - 0.2], [y + row_h / 2, y + row_h / 2], color=theme.grid,
                        lw=0.8, zorder=1)
        spans.append((y_top, y + row_h / 2))
        y -= 0.15
    y_axis = spans[-1][1]
    ax.plot([x0, x1], [y_axis, y_axis], color=theme.axis, lw=0.8)
    for v in range(0, int(top_share) + 1, 10):
        ax.plot([X(v), X(v)], [y_axis, y_axis - 0.06], color=theme.axis, lw=0.8)
        for a, b in spans:
            ax.plot([X(v), X(v)], [b, a], color=theme.grid, lw=0.6, zorder=0)
        ax.text(X(v), y_axis - 0.2, f"{v}%", ha="center", va="center", fontsize=9.5,
                color=muted)
    ax.text((x0 + x1) / 2, y_axis - 0.45, f"share of the {n} tasks (both panels)",
            ha="center", va="center", fontsize=10, color=sec)
    if not all(m.sft_beyond.excludes_zero and m.control_beyond.excludes_zero
               for _, mv in panels for m in mv.rows):
        ax.text(cx_s, y_axis - 0.2, "grey: interval includes 0", ha="left", va="center",
                fontsize=9, color=muted)
    bottom = y_axis - 0.7
    ax.set_ylim(bottom, top + 0.7)
    fig.set_size_inches(W, top + 0.7 - bottom)
    return save_both(fig, "group-movements", theme, out)


# --------------------------------------------------------------------------
# F7: base vs instruct, untrained -> SFT (sft-report r5b and r5d, both models)
# --------------------------------------------------------------------------

R5B_TITLE = "Outcomes per arm, tasks that need a call (r5b)"
R5B_HEADERS = ["arm", "no call", "unparse", "wrong fn", "wrong n", "bad args", "right"]
R5B_TABLES = {"ours": "ours, tasks that need a call (800), greedy:",
              "bfcl": "BFCL call items, pooled (900), greedy:"}


def parse_r5b(text: str) -> dict[str, dict[str, list[dict[str, int]]]]:
    """r5b's two count tables: {"ours"|"bfcl": {"untrained": [counts], "SFT": [per seed],
    "control": [per seed]}}, counts by column; every row checked to sum to the table's n."""
    lines = section(text, R5B_TITLE)
    out = {}
    for key, note in R5B_TABLES.items():
        n = to_int(match(r".*\((\d+)\), greedy:", note, "r5b").group(1), "r5b n")
        rows = table(lines, R5B_HEADERS, f"r5b, {key}", start=note_index(lines, note, "r5b"))
        arms: dict[str, list[dict[str, int]]] = {"untrained": [], "SFT": [], "control": []}
        arm = None
        for r in rows:
            if r[0] in ("SFT, mean", "control, mean"):
                arm = r[0].split(",")[0]
                continue
            if r[0] == "untrained":
                arm = "untrained"
            elif not re.fullmatch(r"seed \d", r[0]) or arm in (None, "untrained"):
                raise CaptureError(f"r5b, {key}: unexpected row {r[0]!r}")
            counts = {h: to_int(c, f"r5b, {key}, {r[0]}") for h, c in zip(R5B_HEADERS[1:], r[1:],
                                                                          strict=True)}
            if sum(counts.values()) != n:
                raise CaptureError(f"r5b, {key}, {r[0]}: counts sum to {sum(counts.values())}, "
                                   f"not {n}")
            arms[arm].append(counts)
        if len(arms["untrained"]) != 1 or not arms["SFT"]:
            raise CaptureError(f"r5b, {key}: no untrained row or no SFT seeds")
        out[key] = {a: v for a, v in arms.items()}
        out[key]["n"] = n  # type: ignore[assignment]
    return out


# (group, label, source table, r5b columns counted, the r5d block that also prints it)
BVI_ROWS = (
    ("Our tasks, a call is needed", "right", "ours", ("right",), None),
    ("Our tasks, a call is needed", "no call or unparseable", "ours", ("no call", "unparse"),
     ("calling when a tool fits", "ours, tasks that need a call")),
    ("Our tasks, a call is needed", "wrong arguments", "ours", ("bad args",),
     ("filling", "ours, tasks that need a call")),
    ("Our tasks, a call is needed", "wrong function or number of calls", "ours",
     ("wrong fn", "wrong n"), ("choosing", "ours, tasks that need a call")),
    ("Our tasks, no tool fits", "called anyway", None, (),
     ("declining when none fits", "ours, no tool fits")),
    ("BFCL, a call is needed", "right", "bfcl", ("right",), ("right", "BFCL call items, pooled")),
    ("BFCL, a call is needed", "no call or unparseable", "bfcl", ("no call", "unparse"),
     ("calling when a tool fits", "BFCL call items, pooled")),
    ("BFCL irrelevance", "called anyway", None, (),
     ("declining when none fits", "BFCL irrelevance")),
)


@dataclass(frozen=True)
class Shift:
    group: str
    label: str
    untrained: float   # percent, one decimal, as the reports print it
    sft: float         # the mean over SFT seeds


def model_shifts(text: str) -> list[Shift]:
    """The eight rows of the post's base-vs-instruct table for one report: from r5b's
    counts (exact seed means, rounded half up as the reports round) where r5b has them,
    else r5d's printed shares; every r5b value checked against r5d where both print it."""
    from fractions import Fraction

    from .effects import percent

    r5b = parse_r5b(text)
    r5d = {(b.name, b.on): b for b in parse_effects(text)}

    def share(counts: list[dict[str, int]], cols, n) -> float:
        k = Fraction(sum(sum(c[x] for x in cols) for c in counts), len(counts) * n)
        return float(percent(k)[:-1])

    out = []
    for group, label, src, cols, block in BVI_ROWS:
        b = r5d.get(block) if block else None
        if block and b is None:
            raise CaptureError(f"r5d has no {block[0]!r} block on {block[1]!r}")
        if src is None:
            un, sft = b.shares["untrained"], b.shares["SFT"]
        else:
            t = r5b[src]
            un, sft = share(t["untrained"], cols, t["n"]), share(t["SFT"], cols, t["n"])
            if b is not None and (b.shares["untrained"], b.shares["SFT"]) != (un, sft):
                raise CaptureError(f"{group}, {label}: r5b gives {un} -> {sft}, r5d prints "
                                   f"{b.shares['untrained']} -> {b.shares['SFT']}")
        out.append(Shift(group, label, un, sft))
    return out


def figure_base_vs_instruct(theme: Theme, panels: list[tuple[str, list[Shift]]],
                            seeds: list[int], out: Path | None = None) -> Path:
    """Two dumbbell panels sharing one row order: per row, a hollow circle untrained, a
    filled circle after SFT (the mean of its seeds), an arrow between, both values printed.
    ``seeds``: each panel's seed count (:func:`seed_count`); they must agree."""
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    blue = theme.series[0]
    if len(panels) != 2 or [(s.group, s.label) for s in panels[0][1]] != [
            (s.group, s.label) for s in panels[1][1]]:
        raise CaptureError("base-vs-instruct needs two panels with the same rows")
    rows = panels[0][1]
    if len(seeds) != len(panels):
        raise CaptureError("base-vs-instruct: one seed count per panel")
    n_seeds = same(seeds, "seed count")
    W = 12.8
    fig, ax = canvas(W, 9.0)
    label_x = 2.85                 # row labels end here
    pw, pgap = 3.5, 1.05           # panel width and the gap between panels
    px = [label_x + 0.6, label_x + 0.6 + pw + pgap]
    note_x = px[1] + pw + 0.2

    def X(k, v):
        return px[k] + pw * v / 100

    top = 8.6
    ax.text(0.2, top + 0.3, "Where SFT takes each model, untrained to trained", ha="left",
            va="center", fontsize=13.5, fontweight="bold", color=ink)
    ax.text(0.2, top - 0.05, "Share of tasks with each outcome. Hollow circle: untrained; "
            f"filled: after SFT, mean of {n_seeds} seeds; the arrow is SFT's change.",
            ha="left", va="center", fontsize=10, color=sec)
    y = top - 0.6
    for k, (title, _) in enumerate(panels):
        ax.text(px[k] + pw / 2, y, title, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=ink)
    y -= 0.35
    y_grid_top = y + 0.1
    row_h, group_gap = 0.5, 0.42
    prev = None
    ys = []
    for i, r in enumerate(rows):
        if r.group != prev:
            y -= group_gap if prev else 0.2
            ax.text(0.2, y + 0.3, r.group, ha="left", va="center", fontsize=10.3,
                    fontweight="bold", color=sec)
            prev = r.group
        right = r.label == "right"
        ax.text(label_x, y, r.label, ha="right", va="center", fontsize=10.5, color=ink,
                fontweight="bold" if right else "normal")
        ax.text(note_x, y, "higher is better" if right else "lower is better", ha="left",
                va="center", fontsize=9.2, color=muted)
        for k, (_, shifts) in enumerate(panels):
            s = shifts[i]
            a, b = X(k, s.untrained), X(k, s.sft)
            if abs(b - a) > 0.12:
                ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(
                    arrowstyle="-|>", color=blue, lw=1.6, shrinkA=5, shrinkB=5,
                    mutation_scale=11), zorder=3)
            ax.plot(a, y, marker="o", ms=8, mfc=theme.surface, mec=sec, mew=1.5, zorder=4)
            ax.plot(b, y, marker="o", ms=8, mfc=blue, mec=blue, mew=1.5, zorder=5)
            lo, hi = (a, b) if a <= b else (b, a)
            fw = "bold" if right else "normal"
            kw = dict(va="center", fontsize=9.2, family=MONO, fontweight=fw, zorder=6)
            if hi - lo < 0.3:  # too close for a label each side: "untrained -> SFT" after
                t = ax.text(hi + 0.12, y, f"{s.untrained:.1f} → ", ha="left", color=sec, **kw)
                ax.text(hi + 0.12 + text_width(fig, t), y, f"{s.sft:.1f}", ha="left",
                        color=blue, **kw)
            else:
                v_lo, v_hi = (s.untrained, s.sft) if a <= b else (s.sft, s.untrained)
                c_lo, c_hi = (sec, blue) if a <= b else (blue, sec)
                ax.text(lo - 0.1, y, f"{v_lo:.1f}", ha="right", color=c_lo, **kw)
                ax.text(hi + 0.1, y, f"{v_hi:.1f}", ha="left", color=c_hi, **kw)
        ys.append(y)
        y -= row_h
    y_axis = ys[-1] - 0.35
    for k in range(2):
        ax.plot([X(k, 0), X(k, 100)], [y_axis, y_axis], color=theme.axis, lw=0.8)
        for v in range(0, 101, 25):
            ax.plot([X(k, v), X(k, v)], [y_axis, y_grid_top], color=theme.grid, lw=0.7,
                    zorder=0)
            ax.text(X(k, v), y_axis - 0.18, f"{v}%", ha="center", va="center", fontsize=9.2,
                    color=muted)
        ax.text(px[k] + pw / 2, y_axis - 0.42, "share of tasks", ha="center", va="center",
                fontsize=10, color=sec)
    # the centre line between the panels
    cx = px[0] + pw + pgap / 2
    ax.plot([cx, cx], [y_axis - 0.1, y_grid_top + 0.35], color=theme.axis, lw=0.9)
    bottom = y_axis - 0.7
    ax.set_ylim(bottom, top + 0.6)
    fig.set_size_inches(W, top + 0.6 - bottom)
    return save_both(fig, "base-vs-instruct", theme, out)


# --------------------------------------------------------------------------
# F8: the one-tool rerun, right on BFCL by category (sft-report --profile onetool)
# --------------------------------------------------------------------------

ONETOOL_CAT_TITLE = "BFCL call items: outcomes per category"
ONETOOL_CAT_HEADERS = ["category", "arm", "n", "right", "no call", "bad args", "wrong fn",
                       "wrong n"]
ONETOOL_ARMS = ("untrained", "main real", "onetool")
# (group header, the categories under it); the capture's note says which offer one function
ONETOOL_GROUPS = (("one function offered", ("simple", "parallel")),
                  ("several functions offered", ("multiple", "par_multi")))
CATEGORY_LABELS = dict(BFCL_CALL_ROWS)


def aligned_cells(line: str, header: str, n_cols: int) -> list[str]:
    """A ``Report.table`` row's cells by column, blanks kept: the first column is
    left-aligned at column 2, the rest right-aligned to their header's end."""
    ends = [m.end() for m in re.finditer(r"\S+(?: \S+)*", header)]
    if len(ends) != n_cols:
        raise CaptureError(f"header {header!r}: {len(ends)} columns, not {n_cols}")
    cells = [""] * n_cols
    for m in re.finditer(r"\S+(?: \S+)*", line):
        if m.start() == 2:
            cells[0] = m.group(0)
        elif m.end() in ends[1:]:
            cells[ends.index(m.end())] = m.group(0)
        else:
            raise CaptureError(f"{line!r}: {m.group(0)!r} is under no column")
    return cells


def exact_total(cell: str, k: int, what: str) -> int:
    """The seed total behind a printed mean of ``k`` seeds (one decimal): the one
    integer whose mean prints as ``cell``."""
    mean = float(match(r"\d+\.\d", cell, what).group(0))
    total = round(mean * k)
    if f"{total / k:.1f}" != cell:
        raise CaptureError(f"{what}: no total over {k} seeds has the mean {cell}")
    return total


@dataclass(frozen=True)
class RerunRow:
    category: str      # as the capture prints it
    n: int
    right: dict[str, str]    # arm -> share, as effects.percent prints it ("90.6%")
    no_call: dict[str, str]
    counts: dict[str, tuple[str, str]]  # arm -> (right, no call) as printed


@dataclass(frozen=True)
class Rerun:
    model: str
    seeds: int
    left_out: int      # one-tool no-call rows left out of the training set
    untrained_runs: int  # the untrained arm's runs (the reading table's; one)
    decoding: str        # Setup's decoding, e.g. "greedy"
    rows: tuple[RerunRow, ...]


def parse_rerun(onetool: str, data: str, main: str | None = None) -> Rerun:
    """The onetool report's per-category counts as shares of n. Untrained counts are
    single runs; main real and onetool are means of their seeds, turned back into
    exact seed totals so a share rounds as the reports round (``effects.percent``).
    ``data``: the onetool sft-data capture (the left-out row count, checked against its
    count of real.jsonl's rows of those types). ``main``: the main run's sft-report, if
    given; its r5a per-seed right counts must give the same totals."""
    from fractions import Fraction

    from .effects import percent

    setup = section(onetool, "Setup")
    ks = {len(seed_list(kv(setup, label, "Setup"), label))
          for label in ("onetool real, seeds", "  real, seeds")}
    seeds = same(ks, "seed count (onetool real, main real)")
    lines = section(onetool, ONETOOL_CAT_TITLE)
    head = [i for i, x in enumerate(lines)
            if re.split(r"\s{2,}", x.strip()) == ONETOOL_CAT_HEADERS]
    if len(head) != 1 or not re.fullmatch(r"  -+", lines[head[0] + 1]):
        raise CaptureError(f"{ONETOOL_CAT_TITLE}: no table headed {ONETOOL_CAT_HEADERS}")
    cols = {h: i for i, h in enumerate(ONETOOL_CAT_HEADERS)}
    by_cat: dict[str, dict] = {}
    cat = None
    for line in lines[head[0] + 2:]:
        if not line.strip() or not re.fullmatch(r"  .*\d", line):
            break
        c = aligned_cells(line, lines[head[0]], len(ONETOOL_CAT_HEADERS))
        if c[0]:
            cat = c[0]
            if cat in by_cat or c[1] != "untrained":
                raise CaptureError(f"{ONETOOL_CAT_TITLE}: {cat!r} starts on {c[1]!r}")
            by_cat[cat] = {"n": to_int(c[cols["n"]], f"{cat}, n"), "arms": {}}
        elif cat is None or c[cols["n"]]:
            raise CaptureError(f"{ONETOOL_CAT_TITLE}: unexpected row {line!r}")
        by_cat[cat]["arms"][c[1]] = c
    want = [c for _, cats in ONETOOL_GROUPS for c in cats]
    if sorted(by_cat) != sorted(want):
        raise CaptureError(f"{ONETOOL_CAT_TITLE}: categories {list(by_cat)}, expected {want}")
    left = kv(section(data, "Which rows train"), "  left out (no call, 1 tool)",
              "Which rows train")
    check = kv(section(data, "Against data/sft/real.jsonl"), "  its rows of the left-out types",
               "Against data/sft/real.jsonl")
    if left != check:
        raise CaptureError(f"sft-data: {left} rows left out, but real.jsonl has {check}")
    main_right = r5a_right_totals(main, seeds) if main is not None else None
    rows = []
    for cat in want:
        n, arms = by_cat[cat]["n"], by_cat[cat]["arms"]
        if tuple(arms) != ONETOOL_ARMS:
            raise CaptureError(f"{cat}: arms {list(arms)}, expected {list(ONETOOL_ARMS)}")
        right, no_call, counts = {}, {}, {}
        for arm, c in arms.items():
            where = f"{cat}, {arm}"
            outcome = c[cols["right"]:]
            if arm == "untrained":
                totals = [to_int(x, where) for x in outcome]
                k = 1
            else:
                totals = [exact_total(x, seeds, where) for x in outcome]
                k = seeds
            if sum(totals) > k * n:
                raise CaptureError(f"{where}: outcomes over n")
            if main_right is not None and arm in main_right[cat]:
                if main_right[cat][arm] != totals[0]:
                    raise CaptureError(f"{where}: right totals {totals[0]}, the main report's "
                                       f"r5a {main_right[cat][arm]}")
            right[arm] = percent(Fraction(totals[0], k * n))
            no_call[arm] = percent(Fraction(totals[1], k * n))
            counts[arm] = (c[cols["right"]], c[cols["no call"]])
        rows.append(RerunRow(cat, n, right, no_call, counts))
    reading = section_matching(onetool, r"The reading: \w+(?:-\w+)*")
    if len(reading) != 1:
        raise CaptureError(f"expected one 'The reading' section, found {len(reading)}")
    runs = {r[0]: to_int(r[1], f"the reading, {r[0]}") for r in table(
        reading[0][1], ["arm", "runs", "replies", "changed"], "the reading")}
    if runs.get("main real") != seeds or runs.get("onetool real") != seeds:
        raise CaptureError(f"the reading's runs {runs}, but {seeds} seeds per SFT arm")
    if "untrained" not in runs:
        raise CaptureError("the reading has no untrained row")
    return Rerun(model=report_model(onetool), seeds=seeds, left_out=to_int(left, "left out"),
                 untrained_runs=runs["untrained"], decoding=kv(setup, "decoding", "Setup"),
                 rows=tuple(rows))


R5A_TITLE = "Right, per arm and category (r5a)"


def r5a_right_totals(text: str, seeds: int) -> dict[str, dict[str, int]]:
    """The main report's r5a per-seed right counts: per category, {"untrained": count,
    "main real": the SFT seeds' total}."""
    lines = section(text, R5A_TITLE)
    head = (["category", "n", "untr"] + [f"SFT {s}" for s in range(1, seeds + 1)]
            + [f"ctl {s}" for s in range(1, seeds + 1)])
    out = {}
    for r in table(lines, head, "r5a"):
        out[r[0]] = {"untrained": to_int(r[2], f"r5a, {r[0]}"),
                     "main real": sum(to_int(x, f"r5a, {r[0]}") for x in r[3:3 + seeds])}
    return out


LABEL_GAP = 0.5  # inches between the centres of two value labels in a row


def fan(xs: list[float], gap: float) -> list[float]:
    """Label centres for markers at ``xs`` (ascending): each run of markers closer than
    ``gap`` gets labels ``gap`` apart, centred on the run, order kept."""
    out: list[float] = []
    runs: list[list[float]] = []
    for x in xs:
        if runs and x - runs[-1][-1] < gap:
            runs[-1].append(x)
        else:
            runs.append([x])
    for run in runs:
        mid = (run[0] + run[-1]) / 2
        out += [mid + gap * (i - (len(run) - 1) / 2) for i in range(len(run))]
    return out


def share_value(cell: str) -> float:
    return float(cell.rstrip("%"))


def figure_rerun(theme: Theme, rr: Rerun, out: Path | None = None) -> Path:
    """One dumbbell panel: per BFCL category, right for the untrained model (hollow), the
    main run's SFT (blue circle) and the rerun without the one-tool no-call rows (blue
    diamond), an arrow from the main run to the rerun; a thin 'no call' column beside."""
    ink, sec, muted = theme.ink, theme.secondary, theme.muted
    blue = theme.series[0]
    W = 12.8
    fig, ax = canvas(W, 6.0)
    label_x = 2.6
    x0, x1 = label_x + 0.75, 9.45          # the share axis
    nc_x = [10.3, 11.15, 12.0]             # the no-call column: one value per arm

    def X(v):
        return x0 + (x1 - x0) * v / 100

    top = 5.6
    ax.text(0.2, top + 0.3, "Leaving out the one-tool rows that don't call: right on BFCL, "
            "by category", ha="left", va="center", fontsize=13.5, fontweight="bold", color=ink)
    ax.text(0.2, top - 0.05, f"{rr.model}. Hollow: untrained; blue circle: SFT, main run; "
            f"blue diamond: SFT without the {rr.left_out} one-tool no-call rows.",
            ha="left", va="center", fontsize=10, color=sec)
    runs = "run" if rr.untrained_runs == 1 else "runs"
    ax.text(0.2, top - 0.3, f"The two SFT arms are means of {rr.seeds} seeds; untrained is "
            f"{number_word(rr.untrained_runs)} {rr.decoding} {runs}.", ha="left", va="center",
            fontsize=10, color=sec)
    top -= 0.25  # everything below sits under the subtitle's second line
    # untrained hollow and drawn last, so its ring shows over an SFT marker it overlaps
    styles = {"untrained": ("o", "none", sec), "main real": ("o", blue, blue),
              "onetool": ("D", blue, blue)}
    # legend
    lx, ly = x0, top - 0.5
    for arm, label in (("untrained", "untrained"), ("main real", "SFT, main run"),
                       ("onetool", "SFT without the one-tool no-call rows")):
        mk, fc, ec = styles[arm]
        ax.plot(lx + 0.1, ly, marker=mk, ms=8 if mk == "o" else 7, mfc=fc, mec=ec, mew=1.5)
        t = ax.text(lx + 0.28, ly, label, ha="left", va="center", fontsize=10.5, color=sec)
        lx += 0.28 + text_width(fig, t) + 0.45
    y = ly - 0.55
    ax.text(nc_x[1], y, "no call", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=ink)
    for xx, name in zip(nc_x, ("untrained", "main", "rerun"), strict=True):
        ax.text(xx, y - 0.24, name, ha="center", va="center", fontsize=8.8, color=muted)
    y -= 0.35
    y_grid_top = y
    row_h, group_gap = 0.62, 0.15
    by_cat = {r.category: r for r in rr.rows}
    ys = []
    first = True
    for group, cats in ONETOOL_GROUPS:
        y -= 0.2 if first else group_gap
        first = False
        ax.text(0.2, y + 0.05, group, ha="left", va="center", fontsize=10.3,
                fontweight="bold", color=sec)
        y -= 0.45
        for cat in cats:
            r = by_cat[cat]
            ax.text(label_x, y, f"{CATEGORY_LABELS[cat]} (n={r.n})", ha="right", va="center",
                    fontsize=10.5, color=ink)
            v = {a: share_value(r.right[a]) for a in ONETOOL_ARMS}
            a, b = X(v["main real"]), X(v["onetool"])
            if abs(b - a) > 0.15:
                ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(
                    arrowstyle="-|>", color=blue, lw=1.6, shrinkA=5, shrinkB=5,
                    mutation_scale=11), zorder=3)
            for arm in ONETOOL_ARMS:
                mk, fc, ec = styles[arm]
                ax.plot(X(v[arm]), y, marker=mk, ms=8 if mk == "o" else 7.5, mfc=fc, mec=ec,
                        mew=1.5, zorder=6 if arm == "untrained" else 5)
            # values above their markers, in the markers' left-to-right order; labels
            # of markers closer than a label's width fan out, with a leader to the marker
            colour = {"untrained": sec, "main real": blue, "onetool": blue}
            order = sorted(ONETOOL_ARMS, key=lambda k: (v[k], ONETOOL_ARMS.index(k)))
            lx_ = fan([X(v[k]) for k in order], LABEL_GAP)
            for k, xl in zip(order, lx_, strict=True):
                ax.text(xl, y + 0.27, f"{v[k]:.1f}", ha="center", va="center", fontsize=9.2,
                        family=MONO, color=colour[k], zorder=6)
                if abs(xl - X(v[k])) > 0.04:
                    ax.plot([X(v[k]), xl], [y + 0.09, y + 0.18], color=theme.grid, lw=0.8,
                            zorder=2)
            for xx, arm in zip(nc_x, ONETOOL_ARMS, strict=True):
                ax.text(xx, y, r.no_call[arm], ha="center", va="center", fontsize=9,
                        family=MONO, color=colour[arm])
            ys.append(y)
            y -= row_h
    y_axis = ys[-1] - 0.35
    ax.plot([x0, x1], [y_axis, y_axis], color=theme.axis, lw=0.8)
    for v in range(0, 101, 25):
        ax.plot([X(v), X(v)], [y_axis, y_grid_top], color=theme.grid, lw=0.7, zorder=0)
        ax.text(X(v), y_axis - 0.18, f"{v}%", ha="center", va="center", fontsize=9.2,
                color=muted)
    sep = (x1 + nc_x[0]) / 2 + 0.05
    ax.plot([sep, sep], [y_axis, y_grid_top + 0.45], color=theme.axis, lw=0.9)
    ax.text((x0 + x1) / 2, y_axis - 0.42, "right, share of items", ha="center",
            va="center", fontsize=10, color=sec)
    bottom = y_axis - 0.7
    ax.set_ylim(bottom, top + 0.25 + 0.6)
    fig.set_size_inches(W, top + 0.25 + 0.6 - bottom)
    return save_both(fig, "onetool-rerun", theme, out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

NAMES = ("tool-call-round-trip", "two-checks", "untrained-outcomes", "toolace-funnel",
         "what-sft-trains", "sft-effects", "group-movements", "base-vs-instruct",
         "onetool-rerun")


def default_score_capture(model: str = MODEL) -> Path:
    """The greedy score capture over the whole test split; while it doesn't exist,
    the 30-per-kind trial in old-labels/ (its r1 block is the same task)."""
    full = OUTPUTS / f"score-{model}-test-greedy-limitall.txt"
    return full if full.exists() else OUTPUTS / "old-labels" / f"score-{model}-test-greedy-limit30.txt"


def default_bfcl_capture(model: str = MODEL) -> Path | None:
    """The one greedy BFCL capture of the eval split under outputs/ (any limit); None if
    there is none. The eval split is every item minus the dev slice, the one the post
    publishes; an all-split or dev capture is used only when passed with --bfcl."""
    found = sorted(OUTPUTS.glob(f"bfcl-{model}-eval-greedy-limit*.txt"))
    if len(found) > 1:
        raise SystemExit(f"several greedy eval BFCL captures: {[p.name for p in found]}; "
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
        check_prompt_view((OUTPUTS / "r1-prompt-view.txt").read_text(), rt)
        render(figure_round_trip, rt)
    if "two-checks" in want:
        render(figure_two_checks, parse_check_demo((OUTPUTS / "check-demo.txt").read_text()))
    if "untrained-outcomes" in want:
        s_path = score or OUTPUTS / f"score-{MODEL}-test-greedy-limitall.txt"
        b_path = bfcl or default_bfcl_capture()
        missing = [p.name for p in (s_path, b_path) if p is not None and not p.exists()]
        if b_path is None:
            missing.append(f"bfcl-{MODEL}-eval-greedy-limit<N>.txt")
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
    if "toolace-funnel" in want:
        render(figure_toolace_funnel, parse_funnel((OUTPUTS / "toolace-stats.txt").read_text(),
                                                   (OUTPUTS / "sft-data.txt").read_text()))
    if "what-sft-trains" in want:
        render(figure_what_sft_trains, parse_sft_example((OUTPUTS / "sft-data.txt").read_text()))
    if "sft-effects" in want or "group-movements" in want:
        report = OUTPUTS / "sft.txt"
        r_text = report.read_text()
        if not has_section(r_text, R5D_TITLE):
            _say("sft-effects, group-movements: skipped, outputs/sft.txt has no r5d")
        else:
            if "sft-effects" in want:
                render(figure_effects, parse_effects(r_text), report_model(r_text))
            if "group-movements" in want:
                base = OUTPUTS / "sft-tagfree-base.txt"
                panels = [(panel_title(r_text), parse_movements(r_text))]
                if base.exists():
                    b_text = base.read_text()
                    panels.append((panel_title(b_text), parse_movements(b_text)))
                else:
                    _say("group-movements: outputs/sft-tagfree-base.txt missing, one panel")
                render(figure_movements, panels)
    if "base-vs-instruct" in want:
        base, inst = OUTPUTS / "sft-tagfree-base.txt", OUTPUTS / "sft.txt"
        if not (base.exists() and inst.exists()):
            _say("base-vs-instruct: skipped, outputs/sft-tagfree-base.txt or sft.txt missing")
        else:
            b_text, i_text = base.read_text(), inst.read_text()
            render(figure_base_vs_instruct, [(panel_title(b_text), model_shifts(b_text)),
                                             (panel_title(i_text), model_shifts(i_text))],
                   [seed_count(b_text), seed_count(i_text)])
    if "onetool-rerun" in want:
        paths = [OUTPUTS / x for x in ("sft-onetool.txt", "sft-data-onetool.txt", "sft.txt")]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            _say(f"onetool-rerun: skipped, capture missing: {', '.join(missing)}")
        else:
            render(figure_rerun, parse_rerun(*(p.read_text() for p in paths)))
    for p in written:
        _say(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return written
