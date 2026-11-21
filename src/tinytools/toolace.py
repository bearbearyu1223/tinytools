"""ToolACE: read the released ``data.json``, count it, convert it to Qwen2.5's tool format.

Each ToolACE row is ``{"system": str, "conversations": [{"from", "value"}, ...]}``.
The system prompt holds instructions and the tool list; assistant turns that call
tools are written ``[Name(arg=value, ...)]`` (parsed by :mod:`tinytools.calls`).

Conversion produces Qwen2.5's native chat format: the tool list goes through
``apply_chat_template(..., tools=...)`` and each call becomes an assistant
message with ``tool_calls``, which the template writes as
``<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>``.

ToolACE's system prompt is fixed boilerplate ("You are an expert in composing
functions..."), the tool list, the bracket-syntax instruction, and in about half
the rows one context line: ``The current time is ...`` or ``Today is ...``. The
converted row keeps only that context line, after Qwen's default system sentence,
so its prompt has the shape a BFCL prompt has (Qwen's sentence, then the tools
block) and still carries the date that relative requests ("next Friday") need.
Rows without a context line get no system message (the template's default).
Every assistant turn is trainable; system, user and tool turns are masked.

Everything here except :func:`render` is model-free and unit-tested.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .calls import CallParseError, looks_like_call, parse_calls

REPO_ID = "Team-ACE/ToolACE"
# The dataset revision every count here was taken from; pinned so a re-download
# cannot change them silently.
REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
FILENAME = "data.json"

_CJK = re.compile(r"[一-鿿]")
# Where the JSON tool list starts, in the English and the Chinese system prompts.
_ANCHOR = re.compile(r"(in JSON format that you can invoke:|的JSON格式列表[:：])\s*")
ZH_HEAD = 200  # characters of the system prompt the language rule looks at
# Qwen2.5's chat template writes this when there is no system message.
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
# One sentence, to its first period: "Today is 2022-03-20, Sunday.." runs straight
# on into "Here is a list of functions" on the same line.
_CONTEXT = re.compile(r"^(?:The current time is |Today is )[^\n.]*\.", re.MULTILINE)


def download() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, FILENAME, repo_type="dataset", revision=REVISION))


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -- language -----------------------------------------------------------------


def is_chinese(row: dict) -> bool:
    """The rule: a row is Chinese when the first 200 characters of its system
    prompt (its instructions, before any tool list) contain a CJK ideograph."""
    return bool(_CJK.search(row["system"][:ZH_HEAD]))


def has_any_cjk(row: dict) -> bool:
    """Contrast for :func:`is_chinese`: a CJK ideograph anywhere in the row."""
    text = row["system"] + "".join(t["value"] for t in row["conversations"])
    return bool(_CJK.search(text))


# -- conversion -----------------------------------------------------------------


class ConvertError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# The format-variation rows write their tool list after this sentence, in one of
# these formats (or give no tool list at all). Detected from the list's opening.
_LIST_INTRO = re.compile(r"Here are the tools you can use:\s*")
NOT_JSON = "tool list not JSON (format-variation row)"
F_LATEX = "LaTeX tabular"
F_HTML = "HTML table"
F_XML = "XML-like tags"
F_MARKDOWN = "Markdown list"
F_YAML = "YAML-like lines"
F_REKEYED = "JSON with other keys"
F_NONE = "no tool list at all"
F_OTHER = "other"
LIST_FORMATS = (F_YAML, F_MARKDOWN, F_LATEX, F_HTML, F_REKEYED, F_XML, F_NONE, F_OTHER)


def tool_list_format(system: str) -> str:
    """Which format a non-JSON tool list is written in, from how it starts. Pure.

    LaTeX ``\\begin{tabular}``, an HTML ``<table``, XML-like ``<tool_name>``
    tags, a Markdown ``- **tool_name**`` list, YAML-like ``tool_name:`` lines,
    JSON objects keyed ``"tool_name"`` / ``"definition"`` instead of ``name`` /
    ``description`` ("JSON with other keys"), or no tool list in the prompt.
    """
    m = _LIST_INTRO.search(system)
    if m is None:
        return F_NONE
    head = system[m.end():m.end() + 40]
    for prefix, name in (("\\begin{tabular}", F_LATEX), ("<table", F_HTML),
                         ("<tool_name>", F_XML), ("- **tool_name**", F_MARKDOWN),
                         ("tool_name:", F_YAML)):
        if head.startswith(prefix):
            return name
    if re.match(r'[\[{]\s*\{?\s*"tool_name"\s*:', head):
        return F_REKEYED
    return F_OTHER


def extract_tools(system: str) -> list[dict]:
    """The JSON tool list that follows the anchor sentence in the system prompt."""
    m = _ANCHOR.search(system)
    if not m:
        raise ConvertError(NOT_JSON)
    try:
        tools, _ = json.JSONDecoder().raw_decode(system, m.end())
    except ValueError:
        raise ConvertError("tool list JSON does not decode") from None
    if not isinstance(tools, list) or not all(
        isinstance(t, dict) and isinstance(t.get("name"), str) for t in tools
    ):
        raise ConvertError("tool list is not a list of named tools")
    return tools


def context_lines(system: str) -> list[str]:
    """The date/time lines in the instructions (the text before the tool list)."""
    m = _ANCHOR.search(system)
    head = system[: m.start()] if m else system
    return [x.group(0).strip() for x in _CONTEXT.finditer(head)]


def wrap_tool(tool: dict) -> dict:
    """OpenAI-style wrapper, the shape Qwen's function-calling docs pass as ``tools``."""
    return {"type": "function", "function": tool}


@dataclass(frozen=True)
class Converted:
    index: int  # position in data.json
    tools: list[dict]  # ToolACE's tool dicts, unwrapped
    messages: list[dict]  # Qwen chat messages; a system message only for a context line


def convert_row(index: int, row: dict) -> Converted:
    """One ToolACE row as Qwen chat messages, or :class:`ConvertError` with a reason."""
    tools = extract_tools(row["system"])
    names = {t["name"] for t in tools}
    messages: list[dict] = []
    context = context_lines(row["system"])
    if context:
        messages.append({"role": "system", "content": "\n".join([QWEN_DEFAULT_SYSTEM, *context])})
    for turn in row["conversations"]:
        who, value = turn["from"], turn["value"]
        if who == "user":
            messages.append({"role": "user", "content": value})
        elif who == "tool":
            messages.append({"role": "tool", "content": value})
        elif who == "assistant":
            if looks_like_call(value):
                try:
                    calls = parse_calls(value, sorted(names))
                except CallParseError as err:
                    raise ConvertError(f"call does not parse: {err.reason}") from None
                if any(c.name not in names for c in calls):
                    raise ConvertError("call names a tool not in the list")
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                        for c in calls
                    ],
                })
            else:
                messages.append({"role": "assistant", "content": value})
        else:
            raise ConvertError(f"unknown role {who!r}")
    first = next((m["role"] for m in messages if m["role"] != "system"), None)
    if first != "user":
        raise ConvertError("conversation does not start with a user turn")
    return Converted(index, tools, messages)


def unknown_argument_calls(conv: Converted) -> int:
    """Calls that pass an argument the tool's schema doesn't list (kept, only counted)."""
    props = {t["name"]: set((t.get("parameters") or {}).get("properties") or {}) for t in conv.tools}
    n = 0
    for m in conv.messages:
        for tc in m.get("tool_calls", []):
            f = tc["function"]
            if set(f["arguments"]) - props.get(f["name"], set()):
                n += 1
    return n


# -- masking --------------------------------------------------------------------

Renderer = Callable[[list[dict], bool], str]


def assistant_spans(messages: list[dict], render: Renderer) -> tuple[str, list[tuple[int, int]]]:
    """The full rendered text and the character span of each assistant turn.

    ``render(messages, add_generation_prompt)`` is the chat template. A turn's
    span starts after the ``<|im_start|>assistant\\n`` the generation prompt adds
    and ends after its ``<|im_end|>`` (so the model learns to stop), excluding
    the newline the template writes after it. Relies on the template being
    prefix-stable, and checks that it is.
    """
    full = render(messages, False)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        before = render(messages[:i], True)
        through = render(messages[: i + 1], False)
        if not (through.startswith(before) and full.startswith(through)):
            raise ConvertError("chat template is not prefix-stable")
        end = len(through)
        if through.endswith("\n"):
            end -= 1
        spans.append((len(before), end))
    return full, spans


def tokens_in_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> tuple[int, int]:
    """(tokens fully inside a span, tokens that straddle a span edge).

    A straddling token would make the mask ambiguous; the statistics print the
    count so a nonzero value is visible.
    """
    inside = straddle = 0
    for a, b in offsets:
        if b <= a:
            continue
        for s, e in spans:
            if s <= a and b <= e:
                inside += 1
                break
            if a < e and b > s:
                straddle += 1
                break
    return inside, straddle


# -- the census -----------------------------------------------------------------


@dataclass
class Census:
    """Every ToolACE count the commands print, computed once."""

    total: int = 0
    chinese: int = 0
    any_cjk: int = 0
    english: int = 0
    failures: Counter = field(default_factory=Counter)
    not_json_formats: Counter = field(default_factory=Counter)  # NOT_JSON rows, by format
    converted: int = 0
    empty_tool_list: int = 0
    context_kept: int = 0
    by_assistant_turns: Counter = field(default_factory=Counter)
    assistant_turns: int = 0
    call_turns: int = 0
    multi_call_turns: int = 0
    calls: int = 0
    text_after_tool: int = 0
    text_after_user: int = 0
    unknown_arg_calls: int = 0
    two_plus_tools: int = 0
    two_plus_tools_with_call: int = 0
    tool_turn_rows: int = 0
    tokens_total: int = 0
    tokens_trainable: int = 0
    tokens_straddling: int = 0


def census_and_convert(rows: list[dict]) -> tuple[Census, list[Converted]]:
    """Count the raw file and convert the English rows. Model-free."""
    c = Census(total=len(rows))
    out: list[Converted] = []
    for i, row in enumerate(rows):
        c.any_cjk += has_any_cjk(row)
        if is_chinese(row):
            c.chinese += 1
            continue
        c.english += 1
        try:
            conv = convert_row(i, row)
        except ConvertError as err:
            c.failures[err.reason] += 1
            if err.reason == NOT_JSON:
                c.not_json_formats[tool_list_format(row["system"])] += 1
            continue
        out.append(conv)
        c.converted += 1
        c.empty_tool_list += not conv.tools
        c.context_kept += conv.messages[0]["role"] == "system"
        roles = [m["role"] for m in conv.messages]
        c.tool_turn_rows += "tool" in roles
        n_asst = 0
        has_call = False
        for j, m in enumerate(conv.messages):
            if m["role"] != "assistant":
                continue
            n_asst += 1
            if m.get("tool_calls"):
                has_call = True
                c.call_turns += 1
                c.calls += len(m["tool_calls"])
                c.multi_call_turns += len(m["tool_calls"]) > 1
            elif j > 0 and conv.messages[j - 1]["role"] == "tool":
                c.text_after_tool += 1
            else:
                c.text_after_user += 1
        c.assistant_turns += n_asst
        c.by_assistant_turns[n_asst] += 1
        c.unknown_arg_calls += unknown_argument_calls(conv)
        if len(conv.tools) >= 2:
            c.two_plus_tools += 1
            c.two_plus_tools_with_call += has_call
    return c, out


# -- rendering with the real tokenizer -------------------------------------------


def make_renderer(tokenizer, tools: list[dict]) -> Renderer:
    wrapped = [wrap_tool(t) for t in tools] or None

    def render(messages: list[dict], add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            messages, tools=wrapped, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return render


def render(conv: Converted, tokenizer, census: Census) -> dict[str, Any]:
    """One JSONL record: messages, tools, the rendered text and its trainable spans.

    Also adds this row's token counts to ``census``.
    """
    text, spans = assistant_spans(conv.messages, make_renderer(tokenizer, conv.tools))
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    inside, straddle = tokens_in_spans(enc["offset_mapping"], spans)
    census.tokens_total += len(enc["input_ids"])
    census.tokens_trainable += inside
    census.tokens_straddling += straddle
    return {
        "id": f"toolace-{conv.index}",
        "tools": [wrap_tool(t) for t in conv.tools],
        "messages": conv.messages,
        "text": text,
        "train_spans": [list(s) for s in spans],
        "n_tools": len(conv.tools),
    }
