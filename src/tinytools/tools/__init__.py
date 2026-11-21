"""Five deterministic tools over fixed fake data and a frozen clock.

    calculator             evaluate an arithmetic expression (Python syntax, no names)
    search_contacts        contacts whose name contains every word of the query
    get_calendar_event     events on a date, or whose title contains every word given
    create_calendar_event  add one event; the result that counts is the new calendar
    search_documents       documents whose title contains every word of the query

The data lives in ``tools/data/*.json`` (made once by ``scripts/make_tool_data.py``
and committed). The clock is frozen at the date and time in ``clock.json``; the
tools never read the real clock.

Every call starts from a fresh copy of the data (:func:`fresh_state`), so no call
can see another's effect. Arguments are checked strictly against the schema:
an unknown argument, a missing required one or a wrong type is an error, and an
error never matches anything (see :mod:`tinytools.checkers`). A start time may
carry zero seconds (``"15:00:00"`` is ``"15:00"``); other seconds are an error. Strings are
stripped and their whitespace runs collapsed before use; an integer parameter
accepts an integral float (``60.0``), as JSON does not distinguish them.

Word matching (contacts, titles): case-insensitive, on whole words
(:func:`words`). For titles and document queries a few filler words
(:data:`STOPWORDS`: "the", "doc", "meeting", "meetings", ...) are dropped from
both sides.
A query with no words left is an error, so an empty query cannot return
everything.

Distractor tools (:data:`DISTRACTORS`) have schemas only. They are offered in
prompts but are never a task's right answer, and calling one is an error.

Pure Python; tested in ``tests/test_tools.py``.
"""

from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import math
import operator
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data"

__all__ = [
    "SCHEMAS", "DISTRACTORS", "TOOL_NAMES", "DISTRACTOR_NAMES", "ToolError", "State",
    "clock", "contacts", "calendar", "documents", "fresh_state", "execute", "words",
    "evaluate", "STOPWORDS", "schema", "context_line",
]


@lru_cache(maxsize=None)
def _load(name: str):
    return json.loads((DATA / name).read_text())


def clock() -> dict:
    """The frozen clock: ``{"date", "time", "weekday", "timezone"}``."""
    return dict(_load("clock.json"))


def now() -> dt.datetime:
    c = clock()
    return dt.datetime.fromisoformat(f"{c['date']}T{c['time']}")


def context_line() -> str:
    """The clock as the system prompt states it (ToolACE's "The current time is ..." shape)."""
    c = clock()
    return f"The current time is {c['date']} {c['time']}, {c['weekday']}."


def contacts() -> list[dict]:
    return copy.deepcopy(_load("contacts.json"))


def calendar() -> list[dict]:
    return copy.deepcopy(_load("calendar.json"))


def documents() -> list[dict]:
    return copy.deepcopy(_load("documents.json"))


# -- schemas (Qwen's tool format: the "function" part of {"type": "function", ...}) --

SCHEMAS: dict[str, dict] = {
    "calculator": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the number. Use Python "
                       "syntax: numbers, + - * / ** % and parentheses, e.g. \"(3 + 4) * 2\".",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "The arithmetic expression."}},
            "required": ["expression"],
        },
    },
    "search_contacts": {
        "name": "search_contacts",
        "description": "Search the user's contacts by name. Returns every contact whose name "
                       "contains all the words of the query (case-insensitive), with email, "
                       "phone and team.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A name or part of a name."}},
            "required": ["query"],
        },
    },
    "get_calendar_event": {
        "name": "get_calendar_event",
        "description": "Look up events on the user's calendar by date, by words in the "
                       "title, or both. Give at least one.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The day, as YYYY-MM-DD."},
                "title": {"type": "string",
                          "description": "Words that appear in the event's title."},
            },
            "required": [],
        },
    },
    "create_calendar_event": {
        "name": "create_calendar_event",
        "description": "Add an event to the user's calendar.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The event's title."},
                "date": {"type": "string", "description": "The day, as YYYY-MM-DD."},
                "start_time": {"type": "string",
                               "description": "Start time, 24-hour HH:MM."},
                "duration_minutes": {"type": "integer",
                                     "description": "Length in minutes. Default 60."},
                "attendees": {"type": "array", "items": {"type": "string"},
                              "description": "Names of the people invited."},
            },
            "required": ["title", "date", "start_time"],
        },
    },
    "search_documents": {
        "name": "search_documents",
        "description": "Search the shared documents by title. Returns every document whose "
                       "title contains all the words of the query (case-insensitive).",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "Words from the document's title."}},
            "required": ["query"],
        },
    },
}

DISTRACTORS: dict[str, dict] = {
    "send_email": {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line."},
                "body": {"type": "string", "description": "The message."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    "delete_calendar_event": {
        "name": "delete_calendar_event",
        "description": "Delete an event from the user's calendar.",
        "parameters": {
            "type": "object",
            "properties": {"event_id": {"type": "string", "description": "The event's id."}},
            "required": ["event_id"],
        },
    },
    "get_weather": {
        "name": "get_weather",
        "description": "Get the weather forecast for a city on a date.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
                "date": {"type": "string", "description": "The day, as YYYY-MM-DD."},
            },
            "required": ["city"],
        },
    },
    "set_reminder": {
        "name": "set_reminder",
        "description": "Set a reminder that notifies the user at a given time.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to remind the user of."},
                "time": {"type": "string", "description": "When, as YYYY-MM-DD HH:MM."},
            },
            "required": ["text", "time"],
        },
    },
    "get_stock_price": {
        "name": "get_stock_price",
        "description": "Get the latest share price for a stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker, e.g. AAPL."}},
            "required": ["symbol"],
        },
    },
    "translate_text": {
        "name": "translate_text",
        "description": "Translate text into another language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate."},
                "target_language": {"type": "string", "description": "e.g. French."},
            },
            "required": ["text", "target_language"],
        },
    },
}

TOOL_NAMES = tuple(SCHEMAS)
DISTRACTOR_NAMES = tuple(DISTRACTORS)


def schema(name: str) -> dict:
    """A tool's schema, real or distractor."""
    return copy.deepcopy(SCHEMAS.get(name) or DISTRACTORS[name])


# -- argument checking -------------------------------------------------------------


class ToolError(ValueError):
    """A call the tool refuses: bad arguments, an unknown tool, a bad expression."""


def _clean_str(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ToolError(f"{key} must be a string")
    return " ".join(value.split())


def _clean_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{key} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ToolError(f"{key} must be an integer")
    return int(value)


def _check(name: str, args: Any) -> dict:
    """Validate ``args`` against the tool's schema; return them cleaned."""
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    props = SCHEMAS[name]["parameters"]["properties"]
    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    missing = [k for k in SCHEMAS[name]["parameters"]["required"] if k not in args]
    if missing:
        raise ToolError(f"missing argument(s): {', '.join(missing)}")
    out = {}
    for key, value in args.items():
        kind = props[key]["type"]
        if kind == "string":
            out[key] = _clean_str(value, key)
        elif kind == "integer":
            out[key] = _clean_int(value, key)
        elif kind == "array":
            if not isinstance(value, list):
                raise ToolError(f"{key} must be a list")
            out[key] = [_clean_str(v, key) for v in value]
    return out


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^(\d{1,2}):(\d{2})(?::00)?$")


def _date(value: str) -> str:
    if not _DATE.match(value):
        raise ToolError("date must be YYYY-MM-DD")
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as e:
        raise ToolError(f"no such date: {value}") from e


def _time(value: str) -> str:
    m = _TIME.match(value)
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ToolError("start_time must be 24-hour HH:MM (or HH:MM:00)")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


# -- word matching -------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
STOPWORDS = frozenset("a an the of for and my our on in to with about doc docs document "
                      "documents file files meeting meetings event events".split())


def words(text: str, *, drop_stopwords: bool = False) -> frozenset[str]:
    """Case-insensitive whole words; ``O'Neill`` and ``Leigh-Anne`` stay one word.

    A possessive stays attached (``O'Neill's`` is not ``O'Neill``): the tool
    searches for what it is given.
    """
    found = frozenset(_WORD.findall(text.lower()))
    return found - STOPWORDS if drop_stopwords else found


def _query_words(query: str, *, drop_stopwords: bool) -> frozenset[str]:
    q = words(query, drop_stopwords=drop_stopwords)
    if not q:
        raise ToolError("the query has no words to search for")
    return q


# -- calculator ----------------------------------------------------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
MAX_EXPONENT = 64
MAX_MAGNITUDE = 1e100


def evaluate(expression: str) -> int | float:
    """Evaluate a Python arithmetic expression without ``eval``.

    Numbers, + - * / // % **, unary minus and parentheses only. ``1,250`` is a
    tuple in Python and is refused, as is ``^`` (xor, not power). Exponents are
    capped at 64 and results at 1e100 so no expression can hang the checker.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise ToolError(f"not an arithmetic expression: {expression!r}") from e

    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ToolError("exponent too large")
            try:
                value = _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError as e:
                raise ToolError("division by zero") from e
            except OverflowError as e:
                raise ToolError("result too large") from e
            if isinstance(value, complex):
                raise ToolError("complex result")
            if abs(value) > MAX_MAGNITUDE:
                raise ToolError("result too large")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        raise ToolError(f"not allowed in an expression: {type(node).__name__}")

    value = ev(tree.body)
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("result is not a finite number")
    return value


# -- the tools -----------------------------------------------------------------------


@dataclass
class State:
    """Everything a tool can read or change. A fresh copy for every call."""

    contacts: list[dict]
    calendar: list[dict]
    documents: list[dict]


def fresh_state() -> State:
    return State(contacts(), calendar(), documents())


def _calculator(state: State, expression: str) -> dict:
    return {"result": evaluate(expression)}


def _search_contacts(state: State, query: str) -> dict:
    q = _query_words(query, drop_stopwords=False)
    hits = [c for c in state.contacts if q <= words(c["name"])]
    return {"contacts": [{k: c[k] for k in ("id", "name", "email", "phone", "team")}
                         for c in hits]}


def _get_calendar_event(state: State, date: str | None = None, title: str | None = None) -> dict:
    if date is None and title is None:
        raise ToolError("give a date, a title, or both")
    day = _date(date) if date is not None else None
    q = _query_words(title, drop_stopwords=True) if title is not None else None
    hits = [e for e in state.calendar
            if (day is None or e["date"] == day)
            and (q is None or q <= words(e["title"], drop_stopwords=True))]
    return {"events": copy.deepcopy(hits)}


def _create_calendar_event(state: State, title: str, date: str, start_time: str,
                           duration_minutes: int = 60, attendees: list[str] | None = None) -> dict:
    if not title:
        raise ToolError("title is empty")
    if not 1 <= duration_minutes <= 24 * 60:
        raise ToolError("duration_minutes must be between 1 and 1440")
    event = {"id": f"evt-{len(state.calendar) + 1:03d}", "date": _date(date),
             "start_time": _time(start_time), "duration_minutes": duration_minutes,
             "title": title, "attendees": list(attendees or [])}
    state.calendar.append(event)
    return {"created": copy.deepcopy(event)}


def _search_documents(state: State, query: str) -> dict:
    q = _query_words(query, drop_stopwords=True)
    hits = [d for d in state.documents if q <= words(d["title"], drop_stopwords=True)]
    return {"documents": copy.deepcopy(hits)}


_IMPL = {
    "calculator": _calculator,
    "search_contacts": _search_contacts,
    "get_calendar_event": _get_calendar_event,
    "create_calendar_event": _create_calendar_event,
    "search_documents": _search_documents,
}


def execute(name: str, arguments: Any, state: State | None = None) -> dict:
    """Run one call. Returns the tool's result, or ``{"error": reason}``.

    With no ``state``, the call runs on a fresh copy of the data. The caller
    can pass its own fresh state to inspect what the call changed.
    """
    state = state if state is not None else fresh_state()
    if name not in _IMPL:
        what = "a distractor (schema only)" if name in DISTRACTORS else "unknown"
        return {"error": f"{name} is {what}; it cannot be run"}
    try:
        return _IMPL[name](state, **_check(name, arguments))
    except ToolError as e:
        return {"error": str(e)}
