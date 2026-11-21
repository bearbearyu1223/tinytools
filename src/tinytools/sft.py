"""The SFT training sets: ToolACE truncated (real arm) and its random-call twins (control).

Both arms start from the converted English ToolACE (``data/toolace-en-qwen.jsonl``).

**Which rows train (plan 3d, fixes after T3).** Every row with 2+ tools in its list
and a call, and every row with no call at all, whatever its tool list (one tool, or
an empty list): those are the declining and asking examples, and they go into both
arms unchanged and identical. Rows with one tool and a call stay out (the control
would always draw the right tool there).

**Truncation (both arms).** Each conversation is cut right after its first
assistant turn that makes a call; any not-calling assistant turn before it (asking
for a missing argument, say) stays. A row with no call is kept whole. After the cut
no row contains a tool result, so a control call is never followed by the result of
a different call.

**Random-call control.** Same rows, same truncation, same not-calling turns. Every
call in an assistant call turn is replaced, one for one (a parallel turn keeps its
number of calls), by a call to a tool drawn uniformly from the row's whole tool list,
the right one included. No argument of the row itself is ever used. The arguments
come from the first of these sources that has one:

1. ``another row``: a call to that tool name in a different row whose arguments fit
   this row's schema for it (keys listed, required keys present, values of the
   declared type);
2. ``another row, other schema``: a call to that name in a different row whose
   arguments don't fit this row's schema (ToolACE reuses names with edited schemas);
3. ``built``: no other row calls the tool, even when it is the replaced call's own
   tool. Its required parameters are filled one by one: an enum value of the schema,
   else the value of a same-named argument of the declared type from a call in
   another row, else the schema's default, else a value of the declared type copied
   from any argument of a call in another row. Empty values (``""``, ``[]``, ``{}``,
   ``null``) are never used, so no built call has a placeholder. Optional parameters
   are left out (77% of ToolACE's real calls pass exactly the required ones).

A control call can still equal the real call by coincidence (a donor row asking the
same thing); the report counts those.

**One control set per training seed** (seed N trains control N). The real set has no
randomness. Every draw comes from ``random.Random(f"{seed}:{row id}")``, so a row's
control depends only on the seed and the row, and candidate lists are in file order.

Everything here except :func:`render` and :func:`build_sets` is model-free and
unit-tested.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from . import toolace

MIN_TOOLS = 2  # rows with a call train only with 2+ tools in the list
DEFAULT_SEEDS = (1, 2, 3)  # one control set per training seed
CONVERTED = "data/toolace-en-qwen.jsonl"
# sha256 of the converted file every count here was taken from (toolace-convert, f748ace).
CONVERTED_SHA256 = "beaa841596a4f4279b0d3982308bf3dd0f0182f7f62c380040f91e7ba64512d1"
# The r4b example, shown with the first seed's control: a short row with 2 tools and
# one call where seed 1 draws the other tool and copies its arguments from another row
# (the usual case), picked by hand and pinned in tests/test_sft.py.
EXAMPLE_ID = "toolace-4231"

ANOTHER_ROW = "another row"
OTHER_SCHEMA = "another row, other schema"
BUILT = "built"
SOURCES = (ANOTHER_ROW, OTHER_SCHEMA, BUILT)

# How a built call's parameters were filled, in the order tried.
P_ENUM = "enum value of the schema"
P_COPIED = "same-named argument, another row"
P_DEFAULT = "schema default"
P_TYPED = "same-typed argument, another row"
PARAM_SOURCES = (P_ENUM, P_COPIED, P_DEFAULT, P_TYPED)

# Which rows train, by type.
R_CALL = "2+ tools, with a call"
R_NO_CALL_2 = "no call, 2+ tools"
R_NO_CALL_1 = "no call, 1 tool"
R_NO_CALL_0 = "no call, empty tool list"
R_OUT = "1 tool, with a call (left out)"
ROW_TYPES = (R_CALL, R_NO_CALL_2, R_NO_CALL_1, R_NO_CALL_0, R_OUT)


# -- truncation -----------------------------------------------------------------


def first_call_index(messages: list[dict]) -> int | None:
    """Index of the first assistant message that makes a call, or None."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            return i
    return None


def truncate(messages: list[dict]) -> list[dict]:
    """The conversation cut right after its first call turn (whole if it has none)."""
    i = first_call_index(messages)
    return list(messages) if i is None else list(messages[: i + 1])


def row_type(n_tools: int, messages: list[dict]) -> str:
    """Which of :data:`ROW_TYPES` a row is (every type but ``R_OUT`` trains)."""
    if first_call_index(messages) is not None:
        return R_CALL if n_tools >= MIN_TOOLS else R_OUT
    return R_NO_CALL_2 if n_tools >= 2 else R_NO_CALL_1 if n_tools == 1 else R_NO_CALL_0


def is_empty(value: Any) -> bool:
    """A value that would be a placeholder: ``""`` (or blank), ``[]``, ``{}``, ``None``."""
    if isinstance(value, str):
        return not value.strip()
    return value is None or value == [] or value == {}


# -- schemas --------------------------------------------------------------------


def type_ok(value: Any, declared: Any) -> bool:
    """Does ``value`` have the JSON-schema type ToolACE declares? Unknown types pass."""
    if isinstance(declared, list):
        return any(type_ok(value, t) for t in declared)
    is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
    checks = {
        "string": isinstance(value, str),
        "int": is_num and isinstance(value, int),
        "integer": is_num and isinstance(value, int),
        "float": is_num,
        "number": is_num,
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "list": isinstance(value, list),
        "dict": isinstance(value, dict),
        "object": isinstance(value, dict),
        "null": value is None,
    }
    return checks.get(str(declared), True)


def _params(tool: dict) -> tuple[dict, list[str]]:
    p = tool.get("parameters") or {}
    props = p.get("properties") or {}
    req = p.get("required") or []
    return (props if isinstance(props, dict) else {}), (req if isinstance(req, list) else [])


def _spec(props: dict, name: str) -> dict:
    spec = props.get(name)
    return spec if isinstance(spec, dict) else {}


def fits(arguments: dict, tool: dict) -> bool:
    """Arguments fit a tool's schema: every key listed, required keys present, types right."""
    props, req = _params(tool)
    if any(k not in props for k in arguments) or any(r not in arguments for r in req):
        return False
    return all(type_ok(v, _spec(props, k).get("type")) for k, v in arguments.items())


# -- the donor pool ---------------------------------------------------------------


@dataclass(frozen=True)
class Donor:
    row: str  # record id, "toolace-N"
    arguments: dict


@dataclass
class Pool:
    """Every real call in the converted data: by tool name, by argument name, and all values."""

    calls: dict[str, list[Donor]] = field(default_factory=lambda: defaultdict(list))
    values: dict[str, list[tuple[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    every_value: list[tuple[str, Any]] = field(default_factory=list)
    _typed: dict[str, list[tuple[str, Any]]] = field(default_factory=dict)

    @classmethod
    def build(cls, records: list[dict]) -> "Pool":
        pool = cls()
        for r in records:
            for m in r["messages"]:
                for tc in m.get("tool_calls") or []:
                    f = tc["function"]
                    pool.calls[f["name"]].append(Donor(r["id"], f["arguments"]))
                    for k, v in f["arguments"].items():
                        pool.values[k].append((r["id"], v))
                        pool.every_value.append((r["id"], v))
        return pool

    def typed(self, declared: Any) -> list[tuple[str, Any]]:
        """Every non-empty argument value of the declared type, in file order (cached)."""
        key = json.dumps(declared, sort_keys=True)
        if key not in self._typed:
            self._typed[key] = [(rid, v) for rid, v in self.every_value
                                if not is_empty(v) and type_ok(v, declared)]
        return self._typed[key]


@dataclass(frozen=True)
class Choice:
    arguments: dict
    source: str  # one of SOURCES
    copied_from: str | None  # the donor row, or None for a built call
    param_sources: tuple[str, ...] = ()  # one per parameter of a built call
    param_donors: tuple[str | None, ...] = ()  # the row each built parameter came from


def build_arguments(tool: dict, row_id: str, pool: Pool, rng: random.Random) -> Choice:
    """Source 3: fill the tool's required parameters one by one (see module doc)."""
    props, req = _params(tool)
    args: dict[str, Any] = {}
    how: list[str] = []
    donors: list[str | None] = []
    for name in req:
        spec = _spec(props, name)
        declared = spec.get("type")
        enum = [v for v in spec.get("enum") or [] if not is_empty(v)] \
            if isinstance(spec.get("enum"), list) else []
        donor = None
        if enum:
            args[name], src = rng.choice(enum), P_ENUM
        else:
            same = [(rid, v) for rid, v in pool.values.get(name, []) if rid != row_id
                    and not is_empty(v) and type_ok(v, declared)]
            if same:
                (donor, args[name]), src = rng.choice(same), P_COPIED
            elif not is_empty(spec.get("default")):
                args[name], src = spec["default"], P_DEFAULT
            else:
                typed = [(rid, v) for rid, v in pool.typed(declared) if rid != row_id]
                if not typed:
                    raise ValueError(f"{row_id}: no value of type {declared!r} in another row "
                                     f"for {tool['name']}.{name}")
                (donor, args[name]), src = rng.choice(typed), P_TYPED
        how.append(src)
        donors.append(donor)
    return Choice(args, BUILT, None, tuple(how), tuple(donors))


def choose_arguments(tool: dict, row_id: str, pool: Pool, rng: random.Random) -> Choice:
    """Arguments for a control call to ``tool`` in row ``row_id`` (sources 1-3, in order).

    Never from row ``row_id`` itself, whether or not ``tool`` is the replaced call's.
    """
    others = [d for d in pool.calls.get(tool["name"], []) if d.row != row_id]
    fitting = [d for d in others if fits(d.arguments, tool)]
    if fitting:
        d = rng.choice(fitting)
        return Choice(d.arguments, ANOTHER_ROW, d.row)
    if others:
        d = rng.choice(others)
        return Choice(d.arguments, OTHER_SCHEMA, d.row)
    return build_arguments(tool, row_id, pool, rng)


def _chars(arguments: dict) -> int:
    return len(json.dumps(arguments, ensure_ascii=False))


def row_rng(seed: int, row_id: str) -> random.Random:
    """The row's own generator: the same draws whatever other rows exist."""
    return random.Random(f"{seed}:{row_id}")


def random_calls(row_id: str, tools: list[dict], messages: list[dict], pool: Pool,
                 seed: int) -> tuple[list[dict], list[dict]]:
    """The control twin of one (truncated) conversation, and a log line per replaced call.

    ``tools`` are ToolACE tool dicts (unwrapped). Messages other than assistant call
    turns are returned unchanged (as the same objects).
    """
    rng = row_rng(seed, row_id)
    by_name = {t["name"]: t for t in tools}
    names = [t["name"] for t in tools]
    out: list[dict] = []
    log: list[dict] = []
    for i, m in enumerate(messages):
        if not (m["role"] == "assistant" and m.get("tool_calls")):
            out.append(m)
            continue
        new_calls = []
        for j, tc in enumerate(m["tool_calls"]):
            drawn = rng.choice(names)
            original = tc["function"]
            ch = choose_arguments(by_name[drawn], row_id, pool, rng)
            new_calls.append({"type": "function",
                              "function": {"name": drawn, "arguments": ch.arguments}})
            log.append({"turn": i, "call": j, "original": original["name"],
                        "drawn": drawn, "source": ch.source, "copied_from": ch.copied_from,
                        "param_sources": list(ch.param_sources),
                        "param_donors": list(ch.param_donors),
                        "identical": drawn == original["name"]
                        and ch.arguments == original["arguments"],
                        "no_arguments": not ch.arguments,
                        "chars_change": _chars(ch.arguments) - _chars(original["arguments"])})
        out.append({**m, "tool_calls": new_calls})
    return out, log


# -- counting ---------------------------------------------------------------------


@dataclass
class Shape:
    """Turn and token counts of one set of conversations (one column of r4a)."""

    rows: int = 0
    assistant_turns: int = 0
    call_turns: int = 0
    parallel_turns: int = 0
    calls: int = 0
    not_calls_after_user: int = 0
    not_calls_after_tool: int = 0
    tool_results: int = 0
    tokens: int = 0
    trained: int = 0

    def add(self, messages: list[dict], tokens: int = 0, trained: int = 0) -> None:
        self.rows += 1
        self.tokens += tokens
        self.trained += trained
        for j, m in enumerate(messages):
            if m["role"] == "tool":
                self.tool_results += 1
            if m["role"] != "assistant":
                continue
            self.assistant_turns += 1
            n = len(m.get("tool_calls") or [])
            if n:
                self.call_turns += 1
                self.calls += n
                self.parallel_turns += n > 1
            elif j > 0 and messages[j - 1]["role"] == "tool":
                self.not_calls_after_tool += 1
            else:
                self.not_calls_after_user += 1


@dataclass
class Cut:
    """What truncation removed from one set of rows."""

    rows: int = 0
    rows_cut: int = 0
    turns_removed: int = 0
    assistant_turns_removed: int = 0
    tokens_removed: int = 0
    trained_removed: int = 0

    def add(self, before: list[dict], after: list[dict], tokens: tuple[int, int],
            trained: tuple[int, int]) -> None:
        self.rows += 1
        self.rows_cut += len(after) < len(before)
        self.turns_removed += len(before) - len(after)
        self.assistant_turns_removed += sum(m["role"] == "assistant"
                                            for m in before[len(after):])
        self.tokens_removed += tokens[0] - tokens[1]
        self.trained_removed += trained[0] - trained[1]


@dataclass
class ControlStats:
    """Over one control set: draws, hits, and where the arguments came from."""

    calls: int = 0
    hits: int = 0  # drawn tool = the tool of the call it replaced
    expected: float = 0.0  # sum of 1/k over calls
    variance: float = 0.0  # sum of (1/k)(1 - 1/k)
    identical: int = 0  # same tool and same arguments as the real call (coincidence)
    identical_no_args: int = 0  # of them, calls with no arguments (only one call is possible)
    from_own_row: int = 0  # arguments taken from the row itself (must stay 0)
    empty_built: int = 0  # built parameters with an empty value (must stay 0)
    rows_by_k: Counter = field(default_factory=Counter)  # rows with a call
    calls_by_k: Counter = field(default_factory=Counter)
    hits_by_k: Counter = field(default_factory=Counter)
    source_hit: Counter = field(default_factory=Counter)
    source_miss: Counter = field(default_factory=Counter)
    param_sources: Counter = field(default_factory=Counter)
    identical_by_source: Counter = field(default_factory=Counter)
    chars_change: Counter = field(default_factory=Counter)  # by source, summed

    def sources(self, source: str) -> int:
        return self.source_hit[source] + self.source_miss[source]

    def mean_chars_change(self, source: str | None = None) -> float:
        if source is None:
            return sum(self.chars_change.values()) / self.calls if self.calls else 0.0
        n = self.sources(source)
        return self.chars_change[source] / n if n else 0.0

    def add(self, row_id: str, k: int, log: list[dict], new_calls: list[dict] = ()) -> None:
        """One row with a call: its log from :func:`random_calls`, and its control calls."""
        self.rows_by_k[k] += 1
        for e in log:
            hit = e["drawn"] == e["original"]
            self.calls += 1
            self.hits += hit
            self.expected += 1 / k
            self.variance += (1 / k) * (1 - 1 / k)
            self.identical += e["identical"]
            self.identical_by_source[e["source"]] += e["identical"]
            self.identical_no_args += e["identical"] and e.get("no_arguments", False)
            self.from_own_row += (e["copied_from"] == row_id
                                  or row_id in (e.get("param_donors") or []))
            self.calls_by_k[k] += 1
            self.hits_by_k[k] += hit
            (self.source_hit if hit else self.source_miss)[e["source"]] += 1
            self.chars_change[e["source"]] += e["chars_change"]
            for src in e["param_sources"]:
                self.param_sources[src] += 1
        for f in new_calls:
            if f["source"] == BUILT:
                self.empty_built += sum(is_empty(v) for v in f["arguments"].values())


# -- rendering with the real tokenizer ---------------------------------------------

Tokenize = Callable[[str], list[tuple[int, int]]]


def count_tokens(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> tuple[int, int]:
    """(tokens, tokens trained) for one rendered row."""
    inside, straddle = toolace.tokens_in_spans(offsets, spans)
    if straddle:
        raise ValueError(f"{straddle} tokens straddle a mask edge")
    return len(offsets), inside


def turn_tokens(text: str, offsets: list[tuple[int, int]],
                spans: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    """Per rendered turn (``<|im_start|>role`` block): (role, tokens, tokens trained).

    A token belongs to the block its first character falls in. The system block
    holds the tool list even when the row has no system message.
    """
    starts = []
    i = text.find("<|im_start|>")
    while i >= 0:
        starts.append(i)
        i = text.find("<|im_start|>", i + 1)
    if not starts or starts[0] != 0:
        raise ValueError("rendered text does not start with <|im_start|>")
    ends = starts[1:] + [len(text)]
    out = []
    for s, e in zip(starts, ends, strict=True):
        head = s + len("<|im_start|>")
        role = text[head:text.find("\n", head)]
        mine = [(a, b) for a, b in offsets if s <= a < e]
        inside, _ = toolace.tokens_in_spans(mine, spans)
        out.append((role, len(mine), inside))
    return out


def render(row_id: str, tools: list[dict], messages: list[dict], tokenizer) -> dict[str, Any]:
    """One training record, the shape ``toolace-convert`` writes plus token counts."""
    text, spans = toolace.assistant_spans(messages, toolace.make_renderer(tokenizer, tools))
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    n, trained = count_tokens(enc["offset_mapping"], spans)
    return {
        "id": row_id,
        "tools": [toolace.wrap_tool(t) for t in tools],
        "messages": messages,
        "text": text,
        "train_spans": [list(s) for s in spans],
        "n_tools": len(tools),
        "n_tokens": n,
        "n_trained": trained,
    }


def jsonl_bytes(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


# -- both sets, from the converted records ----------------------------------------


@dataclass
class ControlSet:
    """One seed's control: its records and what they count."""

    seed: int
    records: list[dict] = field(default_factory=list)
    shape: Shape = field(default_factory=Shape)
    stats: ControlStats = field(default_factory=ControlStats)


@dataclass
class Sets:
    """Everything ``sft-data`` prints, computed once."""

    real: list[dict]
    controls: dict[int, ControlSet]
    shape_converted: Shape  # every converted row, as converted
    shape_truncated: Shape  # every converted row, truncated
    shape_real: Shape  # the training rows (the real arm)
    cut_all: Cut
    cut_train: Cut
    rows_converted: Counter  # by ROW_TYPES
    rows_trained: Counter


STRUCTURE = ("rows", "assistant_turns", "call_turns", "parallel_turns", "calls",
             "not_calls_after_user", "not_calls_after_tool", "tool_results")


def same_structure(a: Shape, b: Shape) -> bool:
    """Equal turn and call counts (tokens aside): the control's one-for-one promise."""
    return all(getattr(a, f) == getattr(b, f) for f in STRUCTURE)


def unwrap(record: dict) -> list[dict]:
    return [t["function"] for t in record["tools"]]


def build_sets(converted: list[dict], seeds: tuple[int, ...], tokenizer,
               on_item: Callable[[int], None] = lambda done: None) -> Sets:
    """Truncate every converted row, keep the training rows (real arm), make each seed's twins.

    Rows with no call go into every control unchanged (the real record, plus an empty
    ``random_calls``); rows with a call are drawn again per seed.
    """
    pool = Pool.build(converted)
    s = Sets([], {seed: ControlSet(seed) for seed in seeds}, Shape(), Shape(), Shape(),
             Cut(), Cut(), Counter(), Counter())
    for done, rec in enumerate(converted, start=1):
        tools = unwrap(rec)
        before = rec["messages"]
        enc = tokenizer(rec["text"], return_offsets_mapping=True, add_special_tokens=False)
        n0, t0 = count_tokens(enc["offset_mapping"], [tuple(x) for x in rec["train_spans"]])
        s.shape_converted.add(before, n0, t0)
        after = truncate(before)
        real = render(rec["id"], tools, after, tokenizer)
        s.shape_truncated.add(after, real["n_tokens"], real["n_trained"])
        cut = ((n0, real["n_tokens"]), (t0, real["n_trained"]))
        s.cut_all.add(before, after, *cut)
        kind = row_type(len(tools), after)
        s.rows_converted[kind] += 1
        if kind != R_OUT:
            s.rows_trained[kind] += 1
            s.cut_train.add(before, after, *cut)
            s.shape_real.add(after, real["n_tokens"], real["n_trained"])
            s.real.append(real)
            for cs in s.controls.values():
                if kind == R_CALL:
                    twin_messages, log = random_calls(rec["id"], tools, after, pool, cs.seed)
                    twin = render(rec["id"], tools, twin_messages, tokenizer)
                    calls = [dict(tc["function"], source=e["source"]) for e, tc in zip(
                        log, [tc for m in twin_messages for tc in m.get("tool_calls") or []],
                        strict=True)]
                    cs.stats.add(rec["id"], len(tools), log, calls)
                else:
                    twin_messages, log, twin = after, [], dict(real)
                twin["random_calls"] = log
                cs.shape.add(twin_messages, twin["n_tokens"], twin["n_trained"])
                cs.records.append(twin)
        on_item(done)
    for cs in s.controls.values():
        if not same_structure(cs.shape, s.shape_real):
            raise AssertionError(f"control seed {cs.seed} changed the turn structure")
    return s
