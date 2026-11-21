"""Two ways to check a reply against a task's reference call: exact match and execution match.

**Exact match**: the reply holds exactly one call, to the reference's tool, and
its arguments equal the reference's after normalization:

* argument order does not matter (arguments compare as a mapping);
* numbers compare by value (``60``, ``60.0``); a string is never a number;
* strings are stripped and runs of whitespace collapsed; case is kept;
* the calculator's ``expression`` compares as a syntax tree, so spacing,
  redundant parentheses and number spelling (``2.50``, ``2.5``) don't matter
  but operand order does (``27 * 348`` is not ``348 * 27``);
* lists keep their order; an omitted optional argument is not filled in.

**Execution match**: the reply holds exactly one call, to the reference's tool;
both calls run on their own fresh copy of the data (:func:`tools.fresh_state`);
the reply's call raises no error; and the results agree:

* calculator: numerically, within ``REL_TOL`` (relative) or ``ABS_TOL``;
  but an expression that is a bare number (no operator: ``9396``, ``-2.5``)
  fails when the reference's has one, since it skips the tool's work (post
  10's reward must not pay for computing the answer by hand);
* search_contacts, get_calendar_event, search_documents: the *same set* of
  records (by id). A broader query that includes the answer fails, and so
  does a narrower one;
* create_calendar_event: the calendar *after* the call, every event compared
  on date, start time, duration, title (case-insensitive, whitespace
  collapsed) and attendees (as a case-insensitive multiset). The tool
  accepts ``"15:00:00"`` for ``"15:00"`` (seconds 00 only), so execution
  match does too; exact match compares the strings and does not.

On a no-tool-fits task (reference ``None``) both checks say right only when
the reply holds no ``<tool_call>`` tag at all (outcome "declined"; any tag
is "called anyway").

Replies are decoded with bfcl-eval's Qwen decoder (:func:`decode`), the one
``bfcl-score`` uses, so a reply decodes the same way in both harnesses. The
outcome labels come from :func:`tinytools.classify.classify` (one place).

Pure apart from the decoder import; tested in ``tests/test_checkers.py``.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any

from . import tools
from .classify import TAG, classify, is_call_list

REL_TOL = 1e-6
ABS_TOL = 1e-9

__all__ = ["REL_TOL", "ABS_TOL", "decode", "normalize_arguments", "exact_match",
           "execution_match", "Check", "check_reply", "call_text", "state_key", "is_bare_number"]


# -- decoding ------------------------------------------------------------------------

_handler = None


def decode(raw: str) -> list[dict] | None:
    """bfcl-eval's ``QwenFCHandler.decode_ast``: ``[{name: {arg: value}}, ...]``, or ``None``.

    It finds ``<tool_call>\\n...\\n</tool_call>`` blocks and silently skips any
    whose JSON doesn't load, so callers compare the decoded count with the tag
    count (as :func:`classify.classify` does).
    """
    global _handler
    from bfcl_eval.constants.enums import ReturnFormat

    if _handler is None:
        from .bfcl import handler

        _handler = handler()
    try:
        return _handler.decode_ast(raw, ReturnFormat.PYTHON, False)
    except Exception:  # noqa: BLE001 - any decode failure means "unparseable"
        return None


def single_call(raw: str, decoded: list | None) -> tuple[str, dict] | None:
    """The one call in a reply that has exactly one tag and it decodes; else ``None``."""
    if raw.count(TAG) != 1 or not is_call_list(decoded) or len(decoded) != 1:
        return None
    (name, args), = decoded[0].items()
    return name, args


def call_text(name: str, args: dict) -> str:
    """A call written the way Qwen writes it (the JSON inside ``<tool_call>``)."""
    return json.dumps({"name": name, "arguments": args})


# -- exact match ---------------------------------------------------------------------


def _canon_expression(expr: str) -> str:
    """An expression's syntax tree, written back out; integral floats written as ints."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return " ".join(expr.split())

    class Ints(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is float and node.value.is_integer():
                return ast.copy_location(ast.Constant(int(node.value)), node)
            return node

    return ast.unparse(Ints().visit(tree))


def _norm(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return ("num", float(value))
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return tuple(_norm(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _norm(v)) for k, v in value.items()))
    return repr(value)


def normalize_arguments(name: str, args: dict) -> tuple:
    """The normalized form exact match compares (see the module docstring)."""
    out = []
    for key, value in args.items():
        if name == "calculator" and key == "expression" and isinstance(value, str):
            out.append((key, ("expr", _canon_expression(value))))
        else:
            out.append((key, _norm(value)))
    return tuple(sorted(out, key=lambda kv: kv[0]))


def exact_match(raw: str, decoded: list | None, reference: dict | None) -> bool:
    if reference is None:
        return TAG not in raw
    call = single_call(raw, decoded)
    if call is None:
        return False
    name, args = call
    return (name == reference["name"]
            and normalize_arguments(name, args)
            == normalize_arguments(reference["name"], reference["arguments"]))


# -- execution match -----------------------------------------------------------------


def state_key(calendar: list[dict]) -> tuple:
    """The calendar as execution match compares it (ids aside: they follow from order)."""
    return tuple(sorted(
        (e["date"], e["start_time"], e["duration_minutes"], " ".join(e["title"].split()).casefold(),
         tuple(sorted(" ".join(a.split()).casefold() for a in e["attendees"])))
        for e in calendar))


@dataclass(frozen=True)
class Run:
    result: dict  # what the tool returned (fed back to the model)
    compared: Any  # what execution match compares


def run(name: str, args: Any) -> Run:
    """Run a call on fresh state; return its result and the value execution match compares."""
    state = tools.fresh_state()
    result = tools.execute(name, args, state)
    if "error" in result:
        return Run(result, None)
    if name == "calculator":
        compared = result["result"]
    elif name == "create_calendar_event":
        compared = state_key(state.calendar)
    else:
        (records,) = result.values()
        compared = frozenset(r["id"] for r in records)
    return Run(result, compared)


def is_bare_number(expression: Any) -> bool:
    """An expression with no operator: a number, maybe signed or parenthesized.

    ``"9396"``, ``"-2.5"``, ``"(42)"`` are bare; ``"348 * 27"`` is not. Anything
    that doesn't parse is not bare (the tool will reject it anyway).
    """
    if not isinstance(expression, str):
        return False
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def skips_the_work(name: str, args: Any, reference: dict) -> bool:
    """A calculator call that hands over the answer as a bare number when the reference computes."""
    if name != "calculator" or reference["name"] != "calculator" or not isinstance(args, dict):
        return False
    return (is_bare_number(args.get("expression"))
            and not is_bare_number(reference["arguments"].get("expression")))


def same_result(name: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False  # an error never matches, not even another error
    if name == "calculator":
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return a == b


def execution_match(raw: str, decoded: list | None, reference: dict | None) -> bool:
    return _execution(raw, decoded, reference)[0]


def _execution(raw: str, decoded, reference) -> tuple[bool, Run | None, Run | None]:
    if reference is None:
        return TAG not in raw, None, None
    ref = run(reference["name"], reference["arguments"])
    call = single_call(raw, decoded)
    if call is None:
        return False, ref, None
    name, args = call
    got = run(name, args)
    ok = (name == reference["name"] and same_result(name, got.compared, ref.compared)
          and not skips_the_work(name, args, reference))
    return ok, ref, got


# -- one reply, both checks, and its outcome ------------------------------------------


@dataclass(frozen=True)
class Check:
    outcome: str  # classify's label, with execution match as the verdict
    exact: bool
    execution: bool
    call: tuple[str, dict] | None  # the reply's single call, if it has exactly one
    reference_run: Run | None
    reply_run: Run | None


def check_reply(raw: str, reference: dict | None, decoded: Any = "decode") -> Check:
    """Both checks on one reply, and its outcome (execution match decides "right")."""
    if decoded == "decode":
        decoded = decode(raw)
    ok, ref_run, got_run = _execution(raw, decoded, reference)
    outcome = classify(raw=raw, decoded=decoded, bfcl_valid=ok, irrelevance=reference is None,
                       expected_names=[reference["name"]] if reference else [])
    return Check(outcome, exact_match(raw, decoded, reference), ok,
                 single_call(raw, decoded), ref_run, got_run)
