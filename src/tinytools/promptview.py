"""A readable view of one templated prompt (post 9, r1): its parts, their tokens, its tool.

The exact prompt (``outputs/r1-prompt.txt``) is one block of ChatML with each tool as
one line of JSON; printed at 74 columns its lines break mid-JSON. This module lays the
same text out part by part. Nothing here is typed template text: the parts are cut out
of the rendered prompt at its markers, :func:`split_prompt` checks that they join back
to the exact prompt, and the tool table is parsed from the JSON inside the prompt.

The pieces are model-free and tested in ``tests/test_promptview.py``; the command is
``tinytools prompt-view`` (``cli.cmd_prompt_view``), which brings the tokenizer.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass

from .ours import tokens_in_span

START, END = "<|im_start|>", "<|im_end|>"  # ChatML's turn markers, as the template writes them
_TURN = re.compile(re.escape(START) + r"(\w+)\n(.*?)(?:" + re.escape(END) + r"\n|\Z)", re.S)

# The parts, in the order the prompt has them; "tools" and "format" are cut out of the
# system turn (Qwen2.5's template appends both to the system text when tools are given).
PARTS = ("system", "tools", "format", "user", "assistant")


@dataclass(frozen=True)
class Segment:
    kind: str  # "marker" (a turn's start or end, replaced by a label in the view) or "part"
    name: str  # the part's name, or the marker's role ("" for an end marker)
    start: int  # character offsets in the prompt: [start, end)
    end: int
    text: str


def split_prompt(text: str) -> list[Segment]:
    """``text`` cut into markers and parts, which join back to exactly ``text``.

    A part runs from its first character up to the next marker or part, so the
    newlines the template puts between two parts belong to the part before them.
    """
    segs: list[Segment] = []

    def add(kind: str, name: str, a: int, b: int) -> None:
        segs.append(Segment(kind, name, a, b, text[a:b]))

    pos = 0
    for m in _TURN.finditer(text):
        if m.start() != pos:
            raise ValueError(f"text between turns at character {pos}: {text[pos:m.start()]!r}")
        role = m.group(1)
        add("marker", role, m.start(), m.start(2))
        a, b = m.start(2), m.end(2)
        if role == "system" and "\n<tools>\n" in text[a:b]:
            t = text.index("# Tools", a, b)
            close = text.index("\n</tools>", t, b) + len("\n</tools>")
            f = close + len(text[close:b]) - len(text[close:b].lstrip("\n"))
            add("part", "system", a, t)
            add("part", "tools", t, f)
            add("part", "format", f, b)
        else:
            add("part", role, a, b)
        if m.end() > b:
            add("marker", "", b, m.end())
        pos = m.end()
    if pos != len(text):
        raise ValueError(f"text after the last turn: {text[pos:]!r}")
    if "".join(s.text for s in segs) != text:  # the view can't drift from the prompt
        raise ValueError("the parts do not join back to the prompt")
    names = tuple(s.name for s in segs if s.kind == "part")
    if names != PARTS:
        raise ValueError(f"expected the parts {PARTS}, got {names}")
    return segs


def part(segs: list[Segment], name: str) -> Segment:
    return next(s for s in segs if s.kind == "part" and s.name == name)


def tool_list_span(text: str) -> tuple[int, int]:
    """The tool list's characters: the ``<tools>`` line through ``</tools>``, the span
    ``ours.prompt_census`` counts (the sentence before it also says "<tools></tools>")."""
    a = text.index("\n<tools>\n") + 1
    b = text.index("\n</tools>\n", a) + len("\n</tools>")
    return a, b


def tool_lines(text: str) -> list[str]:
    """The tool list's lines between ``<tools>`` and ``</tools>``: one JSON object each."""
    a, b = tool_list_span(text)
    return text[a:b].split("\n")[1:-1]


def tools_in(text: str) -> list[dict]:
    """Each tool's JSON, parsed back out of the prompt."""
    return [json.loads(line) for line in tool_lines(text)]


@dataclass(frozen=True)
class PartTokens:
    """The prompt's tokens by part; every token falls in exactly one row."""
    markers: int  # tokens wholly inside the turn markers
    parts: dict[str, int]  # tokens wholly inside each part
    across_parts: int  # tokens crossing a marker or part edge (0 for r1's prompt)
    preamble: int  # within "tools": the template's sentences before the list
    tool_list: int  # within "tools": the <tools> block, as ours.prompt_census counts it
    across_list_end: int  # within "tools": tokens crossing the list's ends
    after_list: int  # within "tools": the blank line after </tools>
    total: int

    def rows_sum(self) -> int:
        return self.markers + sum(self.parts.values()) + self.across_parts


def part_tokens(text: str, offsets: list[tuple[int, int]]) -> PartTokens:
    """Count ``offsets`` (the whole prompt's tokens) into the parts of :func:`split_prompt`."""
    segs = split_prompt(text)
    inside = {s: tokens_in_span(offsets, s.start, s.end)[0] for s in segs}
    across = len(offsets) - sum(inside.values())
    tools = part(segs, "tools")
    la, lb = tool_list_span(text)
    pre = tokens_in_span(offsets, tools.start, la)[0]
    lst, st = tokens_in_span(offsets, la, lb)
    after = tokens_in_span(offsets, lb, tools.end)[0]
    if pre + lst + after + st != inside[tools]:
        raise ValueError("a token crosses the tools part's own edges and a list edge at once")
    out = PartTokens(markers=sum(n for s, n in inside.items() if s.kind == "marker"),
                     parts={s.name: n for s, n in inside.items() if s.kind == "part"},
                     across_parts=across, preamble=pre, tool_list=lst, across_list_end=st,
                     after_list=after, total=len(offsets))
    if out.rows_sum() != out.total:
        raise ValueError(f"the rows sum to {out.rows_sum()}, not {out.total}")
    return out


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    required: bool
    description: str


def type_label(spec: dict) -> str:
    """A parameter's JSON Schema type; an array names its items' type ("string array")."""
    t = spec.get("type", "?")
    if t == "array" and isinstance(spec.get("items"), dict) and "type" in spec["items"]:
        return f"{spec['items']['type']} array"
    return t


def params(tool: dict) -> list[Param]:
    """One row per parameter, in the JSON's order."""
    p = tool["function"]["parameters"]
    req = set(p.get("required", []))
    return [Param(n, type_label(s), n in req, s.get("description", ""))
            for n, s in p.get("properties", {}).items()]


def wrap(text: str, width: int) -> list[str]:
    """``text``'s lines, each wrapped at spaces to ``width`` (never mid-word unless a word
    is longer than ``width``, never at a hyphen); blank lines kept."""
    out = []
    for line in text.split("\n"):
        out += textwrap.wrap(line, width=width, break_on_hyphens=False,
                             replace_whitespace=False) or [""]
    return out


def param_table(rows: list[Param], width: int) -> list[str]:
    """``rows`` as an aligned table no wider than ``width``; descriptions wrap in their column."""
    head = ("parameter", "type", "required", "description")
    cells = [(r.name, r.type, "yes" if r.required else "no", r.description) for r in rows]
    w = [max(len(c[i]) for c in [head, *cells]) for i in range(3)]
    lead = sum(w) + 2 * 3
    if width - lead < 20:
        raise ValueError(f"no room for descriptions in {width} columns")

    def line(c, desc):
        return f"{c[0]:<{w[0]}}  {c[1]:<{w[1]}}  {c[2]:<{w[2]}}  {desc}".rstrip()

    out = [line(head, head[3]), "-" * width]
    for c in cells:
        d = wrap(c[3], width - lead)
        out.append(line(c, d[0]))
        out += [(" " * lead + x).rstrip() for x in d[1:]]
    return out
