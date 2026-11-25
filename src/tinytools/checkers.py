"""Two ways to check a reply against a task's reference call: exact match and execution match.

**Exact match**: the reply holds exactly one call, to the reference's tool, and
its arguments equal the reference's after normalization:

* argument order does not matter (arguments compare as a mapping);
* numbers compare by value (``60``, ``60.0``); a string is never a number;
* strings are stripped and runs of whitespace collapsed; case is kept;
* the calculator's ``expression`` compares as a syntax tree, so spacing,
  redundant parentheses and number spelling (``2.50``, ``2.5``) don't matter
  but operand order does (``27 * 348`` is not ``348 * 27``);
* lists keep their order; an omitted optional argument is not filled in,
  and an optional argument given as ``None`` or a blank string counts as
  omitted (as the tool treats it, and as BFCL's checker treats ``""``).

**Execution match**: the reply holds exactly one call, to the reference's tool;
both calls run on their own fresh copy of the data (:func:`tools.fresh_state`);
the reply's call raises no error; and the results agree:

* calculator: the two expressions must have the same tree after
  :func:`expression_tree`'s normalization, *and* agree numerically within
  ``REL_TOL`` (relative) or ``ABS_TOL``. The normalization sorts the
  operands of ``+`` and ``*`` (flattening chains: ``a * b * c``), removes
  identity operations (``x + 0``, ``0 + x``, ``x - 0``, ``x * 1``, ``1 * x``,
  ``x / 1``, ``x // 1``, ``x ** 1``), cancels double signs and ignores
  parentheses and number spelling. So ``27 * 348`` passes against
  ``348 * 27``, but the answer worked out by hand does not, whether bare
  (``9396``) or dressed up (``9396 * 1``, ``--9396``, ``9396 + 0``): post
  10's reward must not pay for computing the answer without the tool.
  One narrow exception, the percent rule (:func:`same_percent_work`):
  against a reference ``x * p / 100``, a product written ``x * 0.pp``,
  ``0.pp * x`` or ``x * (p / 100)`` is the same work when its number times
  exactly 100 is the reference's ``p`` and the rest of the product matches.
  Nothing else is folded: ``2.5`` still fails against ``250 / 100``, and
  ``2500`` against ``2.5 * 1000``;
* search_contacts, get_calendar_event, search_documents: the *same set* of
  records (by id). A broader query that includes the answer fails, and so
  does a narrower one;
* create_calendar_event: the calendar *after* the call, every event compared
  on date, start time, duration, title (case-insensitive, whitespace
  collapsed) and attendees (as a case-insensitive multiset). The tool
  accepts ``"15:00:00"`` for ``"15:00"`` (seconds 00 only), so execution
  match does too; exact match compares the strings and does not.
* post 10's operation tools (:data:`tools.OPERATIONS`): the math ones
  (differentiate, integrate, factor_polynomial) by the polynomial given *and*
  the one returned, both as canonical text (:mod:`tinytools.poly`), so
  differentiating ``x^2 + 3x`` fails against ``x^2 + 3x + 5`` though the
  derivatives agree; the others by the *whole* state after the call
  (:func:`whole_state_key`: calendar, contacts, documents), so deleting or
  changing a record other than the one asked for fails, and a name or title
  that matches more than one record is an error. A title the model writes
  (create_document, rename_document) compares as the tools compare titles
  (:func:`tools.title_key`: its words, stopwords and punctuation aside); a
  phone by its digits only.

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
from decimal import Decimal
from dataclasses import dataclass
from typing import Any

from . import tools
from .classify import TAG, classify, is_call_list

REL_TOL = 1e-6
ABS_TOL = 1e-9

__all__ = ["REL_TOL", "ABS_TOL", "decode", "normalize_arguments", "exact_match",
           "execution_match", "Check", "check_reply", "call_text", "state_key", "is_bare_number",
           "expression_tree", "same_percent_work"]


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


def _drop_blank_optional(name: str, args: dict) -> dict:
    """``args`` without optional arguments given as ``None`` or blank (a known tool's schema)."""
    if name not in tools.EXECUTABLE_NAMES:
        return args
    required = tools.schema(name)["parameters"]["required"]
    return {k: v for k, v in args.items() if k in required or not tools.is_blank(v)}


def normalize_arguments(name: str, args: dict) -> tuple:
    """The normalized form exact match compares (see the module docstring)."""
    out = []
    for key, value in _drop_blank_optional(name, args).items():
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


# Post 10's operation tools (tools.OPERATIONS): the math ones return a polynomial; the
# others change the state, and execution match compares all of it after the call.
MATH_OPERATIONS = ("differentiate", "integrate", "factor_polynomial")


def whole_state_key(state: tools.State) -> tuple:
    """Calendar, contacts and documents as execution match compares them, so a call that
    changes the wrong record, or one record too many, never matches."""
    return (state_key(state.calendar), tools.contacts_key(state.contacts),
            tools.documents_key(state.documents))


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
    elif name in MATH_OPERATIONS:
        # the input and the result, each as poly's canonical text: d/dx drops the
        # constant, so a call on the wrong polynomial can return the right derivative
        from . import poly

        compared = (poly.normalize(args["polynomial"]), result["result"])
    elif name in tools.OPERATION_NAMES:
        compared = whole_state_key(state)
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


_ZERO, _ONE = ("n", 0.0), ("n", 1.0)
_FLAT = {ast.Add: "+", ast.Mult: "*"}
_DROP = {"+": _ZERO, "*": _ONE}  # the identity operand of each commutative operator
_KEEP = {ast.Sub: "-", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
_RIGHT_IDENTITY = {"-": _ZERO, "/": _ONE, "//": _ONE, "**": _ONE}


def expression_tree(expression: Any) -> tuple | None:
    """A calculator expression's tree, normalized for execution match; ``None`` if not arithmetic.

    Nodes are tuples: ``("n", value)`` for a number (as a float, so ``2.50``,
    ``2.5`` and ``5/2``'s operands compare by value), ``("neg", x)`` for a
    minus sign in front of a non-number, ``("+", operands)`` and
    ``("*", operands)`` with the operands flattened across chains and sorted,
    and ``(op, left, right)`` for ``- / // % **``. Removed: unary plus, double
    minus, ``+ 0`` / ``0 +``, ``* 1`` / ``1 *``, and ``- 0``, ``/ 1``,
    ``// 1``, ``** 1`` on the right. Nothing is computed: a subtree of numbers
    stays a subtree (``348 * 27`` is not ``9396``), which is the point.
    Removing ``// 1`` can change a float's value (``2.5 // 1``); execution
    match still compares the evaluated results, so that never passes wrongly.
    """
    if not isinstance(expression, str):
        return None
    try:
        body = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None
    return _tree(body)


def _tree(node) -> tuple | None:
    if isinstance(node, ast.Constant):
        return ("n", float(node.value)) if type(node.value) in (int, float) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        x = _tree(node.operand)
        if x is None or isinstance(node.op, ast.UAdd):
            return x
        if x[0] == "n":
            return ("n", -x[1])
        return x[1] if x[0] == "neg" else ("neg", x)
    if isinstance(node, ast.BinOp) and type(node.op) in _FLAT:
        op = _FLAT[type(node.op)]
        left, right = _tree(node.left), _tree(node.right)
        if left is None or right is None:
            return None
        operands = []
        for x in (left, right):
            operands.extend(x[1] if x[0] == op else [x])
        operands = [x for x in operands if x != _DROP[op]]
        if not operands:
            return _DROP[op]
        if len(operands) == 1:
            return operands[0]
        return (op, tuple(sorted(operands, key=repr)))
    if isinstance(node, ast.BinOp) and type(node.op) in _KEEP:
        op = _KEEP[type(node.op)]
        left, right = _tree(node.left), _tree(node.right)
        if left is None or right is None:
            return None
        if _RIGHT_IDENTITY.get(op) == right:
            return left
        return (op, left, right)
    return None


def same_work(name: str, args: Any, reference: dict) -> bool:
    """For a calculator call: the reply's expression has the reference's normalized tree.

    True for every other tool (their results are compared on their own).
    """
    if name != "calculator" or reference["name"] != "calculator":
        return True
    if not isinstance(args, dict):
        return False
    got = expression_tree(args.get("expression"))
    want = expression_tree(reference["arguments"].get("expression"))
    return got is not None and (got == want or same_percent_work(got, want))


_HUNDRED = ("n", 100.0)


def _times_100(value: float) -> Decimal:
    """``value`` as written (its shortest repr) times 100, exactly: ``0.48`` gives 48."""
    return Decimal(repr(value)) * 100


def same_percent_work(got: tuple | None, want: tuple | None) -> bool:
    """The percent rule: ``want`` is ``x * p / 100`` and ``got`` the same product with
    one operand ``p`` written as ``p / 100`` or as a decimal ``d`` with ``d * 100 == p``
    exactly (``3438 * 0.48``, ``0.48 * 3438``, ``3438 * (48 / 100)``). Pure.

    Both are :func:`expression_tree` trees. Only a whole reference of that shape
    qualifies, and only when its numerator is a product (so ``250 / 100`` stays
    ``250 / 100``: ``2.5`` and ``250 * 0.01`` don't match it). Nothing is
    computed beyond that one operand, and execution match still compares the
    evaluated results.
    """
    if got is None or want is None or got[0] != "*":
        return False
    if want[0] != "/" or want[2] != _HUNDRED or want[1][0] != "*":
        return False
    target = sorted(want[1][1], key=repr)
    numbers = {x[1] for x in target if x[0] == "n"}
    operands = list(got[1])
    for i, x in enumerate(operands):
        if x[0] == "n":
            p = float(_times_100(x[1]))
        elif x[0] == "/" and x[1][0] == "n" and x[2] == _HUNDRED:
            p = x[1][1]
        else:
            continue
        if p not in numbers:
            continue
        rest = operands[:i] + [("n", p)] + operands[i + 1:]
        if sorted(rest, key=repr) == target:
            return True
    return False


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
          and same_work(name, args, reference))
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
