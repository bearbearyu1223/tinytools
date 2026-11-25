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
An *optional* argument whose value is ``None`` or a blank string counts as
omitted (as BFCL's checker treats ``""``); a required one stays strict.

Word matching (contacts, titles): case-insensitive, on whole words
(:func:`words`). For titles and document queries a few filler words
(:data:`STOPWORDS`: "the", "doc", "meeting", "meetings", ...) are dropped from
both sides.
A query with no words left is an error, so an empty query cannot return
everything. The one exception: get_calendar_event given a date treats a title
with no words left as no title (the date alone narrows the search).

Distractor tools (:data:`DISTRACTORS`) have schemas only. They are offered in
prompts but are never a task's right answer, and calling one is an error.

**Operation tools (post 10)**, :data:`OPERATIONS`: twelve more tools that run, in a
registry of their own so post 9's lists (:data:`TOOL_NAMES`, :data:`DISTRACTOR_NAMES`)
and prompts are unchanged; :data:`EXECUTABLE_NAMES` is both. They change the state or
compute a polynomial:

    cancel_calendar_event, update_calendar_event          the calendar
    add_contact, update_contact, delete_contact           the contacts
    create_document, delete_document, rename_document,    the documents
    share_document
    differentiate, integrate, factor_polynomial           a polynomial in x (:mod:`poly`)

A record is found by name or title, never by a search: a contact by its whole name as
the contact list spells it (case and spacing aside); an event or a document by its
whole title (the same words, :data:`STOPWORDS` aside). No match is an error, and so is
more than one (an event title that occurs twice needs its date, for cancelling). Every
call runs on fresh state, so execution match compares the whole state after the call.

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
    "evaluate", "STOPWORDS", "schema", "context_line", "is_blank", "OPERATIONS",
    "OPERATION_NAMES", "EXECUTABLE_NAMES", "contacts_key", "documents_key", "title_key",
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


# -- operation tools (post 10): their own registry ------------------------------------

_TITLE = {"type": "string", "description": "The event's title."}
_DOC_TITLE = {"type": "string", "description": "The document's title."}
_CONTACT_NAME = {"type": "string",
                 "description": "The contact's full name, as the contact list spells it."}
_POLY = {"type": "string",
         "description": "A polynomial in x with integer coefficients, e.g. \"3x^2 - 4x + 1\"."}

OPERATIONS: dict[str, dict] = {
    "cancel_calendar_event": {
        "name": "cancel_calendar_event",
        "description": "Cancel an event on the user's calendar, found by its title. If two "
                       "events have the same title, give the date too.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": _TITLE,
                "date": {"type": "string",
                         "description": "The event's day, as YYYY-MM-DD (only needed to tell "
                                        "apart events with the same title)."},
            },
            "required": ["title"],
        },
    },
    "update_calendar_event": {
        "name": "update_calendar_event",
        "description": "Move an event on the user's calendar to a new start time, a new day, "
                       "or both. The event is found by its title; its length stays the same.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": _TITLE,
                "start_time": {"type": "string", "description": "The new start time, 24-hour HH:MM."},
                "date": {"type": "string", "description": "The new day, as YYYY-MM-DD."},
            },
            "required": ["title"],
        },
    },
    "add_contact": {
        "name": "add_contact",
        "description": "Add a new contact to the user's contacts.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The person's full name."},
                "phone": {"type": "string", "description": "Phone number."},
                "email": {"type": "string", "description": "Email address."},
                "team": {"type": "string", "description": "The team they work on."},
            },
            "required": ["name"],
        },
    },
    "update_contact": {
        "name": "update_contact",
        "description": "Change the phone number, email address or team of one of the user's "
                       "contacts. Give at least one new value.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": _CONTACT_NAME,
                "phone": {"type": "string", "description": "The new phone number."},
                "email": {"type": "string", "description": "The new email address."},
                "team": {"type": "string", "description": "The new team."},
            },
            "required": ["name"],
        },
    },
    "delete_contact": {
        "name": "delete_contact",
        "description": "Delete one of the user's contacts.",
        "parameters": {
            "type": "object",
            "properties": {"name": _CONTACT_NAME},
            "required": ["name"],
        },
    },
    "create_document": {
        "name": "create_document",
        "description": "Create a new, empty shared document.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "The new document's title."}},
            "required": ["title"],
        },
    },
    "delete_document": {
        "name": "delete_document",
        "description": "Delete a shared document, found by its title.",
        "parameters": {
            "type": "object",
            "properties": {"title": _DOC_TITLE},
            "required": ["title"],
        },
    },
    "rename_document": {
        "name": "rename_document",
        "description": "Rename a shared document.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The document's current title."},
                "new_title": {"type": "string", "description": "The new title."},
            },
            "required": ["title", "new_title"],
        },
    },
    "share_document": {
        "name": "share_document",
        "description": "Share a shared document with one of the user's contacts (edit access).",
        "parameters": {
            "type": "object",
            "properties": {
                "title": _DOC_TITLE,
                "contact": _CONTACT_NAME,
            },
            "required": ["title", "contact"],
        },
    },
    "differentiate": {
        "name": "differentiate",
        "description": "Differentiate a polynomial in x and return the derivative.",
        "parameters": {"type": "object", "properties": {"polynomial": _POLY},
                       "required": ["polynomial"]},
    },
    "integrate": {
        "name": "integrate",
        "description": "Integrate a polynomial in x and return the indefinite integral, "
                       "with constant 0.",
        "parameters": {"type": "object", "properties": {"polynomial": _POLY},
                       "required": ["polynomial"]},
    },
    "factor_polynomial": {
        "name": "factor_polynomial",
        "description": "Factor a polynomial in x into linear factors with integer roots, "
                       "e.g. \"x^2 - x - 6\" gives (x + 2)(x - 3).",
        "parameters": {"type": "object", "properties": {"polynomial": _POLY},
                       "required": ["polynomial"]},
    },
}

OPERATION_NAMES = tuple(OPERATIONS)
EXECUTABLE_NAMES = TOOL_NAMES + OPERATION_NAMES


def schema(name: str) -> dict:
    """A tool's schema: post 9's real tools and distractors, or an operation tool."""
    return copy.deepcopy(SCHEMAS.get(name) or DISTRACTORS.get(name) or OPERATIONS[name])


def _schema_for_check(name: str) -> dict:
    return SCHEMAS[name] if name in SCHEMAS else OPERATIONS[name]


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


def is_blank(value: Any) -> bool:
    """``None`` or a string with nothing but whitespace: an optional argument left out."""
    return value is None or (isinstance(value, str) and not value.strip())


def _check(name: str, args: Any) -> dict:
    """Validate ``args`` against the tool's schema; return them cleaned."""
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    props = _schema_for_check(name)["parameters"]["properties"]
    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    required = _schema_for_check(name)["parameters"]["required"]
    args = {k: v for k, v in args.items() if k in required or not is_blank(v)}
    missing = [k for k in required if k not in args]
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
    if title is not None and day is not None and not words(title, drop_stopwords=True):
        title = None  # only filler words ("meetings"): with a date, that is no title
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


# -- the operation tools ---------------------------------------------------------------


def _same_name(a: str, b: str) -> bool:
    return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


def _same_title(a: str, b: str) -> bool:
    """The same words (case-insensitive, :data:`STOPWORDS` aside); a title of nothing
    but filler words matches nothing."""
    wa = words(a, drop_stopwords=True)
    return bool(wa) and wa == words(b, drop_stopwords=True)


def _one(records: list[dict], what: str, given: str) -> dict:
    if not records:
        raise ToolError(f"no {what} matches {given!r}")
    if len(records) > 1:
        raise ToolError(f"{len(records)} {what}s match {given!r}")
    return records[0]


def _contact(state: State, name: str) -> dict:
    return _one([c for c in state.contacts if _same_name(c["name"], name)], "contact", name)


def _document(state: State, title: str) -> dict:
    return _one([d for d in state.documents if _same_title(d["title"], title)], "document",
                title)


def _event(state: State, title: str, date: str | None = None) -> dict:
    day = _date(date) if date is not None else None
    hits = [e for e in state.calendar if _same_title(e["title"], title)
            and (day is None or e["date"] == day)]
    if len(hits) > 1:
        raise ToolError(f"{len(hits)} events match {title!r}; give the date")
    return _one(hits, "event", title)


def _cancel_calendar_event(state: State, title: str, date: str | None = None) -> dict:
    event = _event(state, title, date)
    state.calendar.remove(event)
    return {"cancelled": copy.deepcopy(event)}


def _update_calendar_event(state: State, title: str, start_time: str | None = None,
                           date: str | None = None) -> dict:
    if start_time is None and date is None:
        raise ToolError("give a new start_time, a new date, or both")
    event = _event(state, title)
    if start_time is not None:
        event["start_time"] = _time(start_time)
    if date is not None:
        event["date"] = _date(date)
    return {"updated": copy.deepcopy(event)}


def _next_id(records: list[dict], prefix: str) -> str:
    return f"{prefix}{max(int(r['id'].split('-')[-1]) for r in records) + 1:03d}"


def _add_contact(state: State, name: str, phone: str | None = None, email: str | None = None,
                 team: str | None = None) -> dict:
    if any(_same_name(c["name"], name) for c in state.contacts):
        raise ToolError(f"{name!r} is already a contact")
    contact = {"id": _next_id(state.contacts, "c-"), "name": name, "email": email or "",
               "phone": phone or "", "team": team or "", "tricky_spelling": False}
    state.contacts.append(contact)
    return {"added": {k: contact[k] for k in ("id", "name", "email", "phone", "team")}}


def _update_contact(state: State, name: str, phone: str | None = None,
                    email: str | None = None, team: str | None = None) -> dict:
    new = {k: v for k, v in (("phone", phone), ("email", email), ("team", team))
           if v is not None}
    if not new:
        raise ToolError("give a new phone, email or team")
    contact = _contact(state, name)
    contact.update(new)
    return {"updated": {k: contact[k] for k in ("id", "name", "email", "phone", "team")}}


def _delete_contact(state: State, name: str) -> dict:
    contact = _contact(state, name)
    state.contacts.remove(contact)
    return {"deleted": {k: contact[k] for k in ("id", "name")}}


def _create_document(state: State, title: str) -> dict:
    if not words(title, drop_stopwords=True):
        raise ToolError("the title has no words")
    if any(_same_title(d["title"], title) for d in state.documents):
        raise ToolError(f"a document titled {title!r} exists")
    doc = {"id": _next_id(state.documents, "doc-"), "title": title, "updated": clock()["date"]}
    state.documents.append(doc)
    return {"created": copy.deepcopy(doc)}


def _delete_document(state: State, title: str) -> dict:
    doc = _document(state, title)
    state.documents.remove(doc)
    return {"deleted": copy.deepcopy(doc)}


def _rename_document(state: State, title: str, new_title: str) -> dict:
    doc = _document(state, title)
    if not words(new_title, drop_stopwords=True):
        raise ToolError("the new title has no words")
    if any(_same_title(d["title"], new_title) for d in state.documents if d is not doc):
        raise ToolError(f"a document titled {new_title!r} exists")
    doc["title"] = new_title
    return {"renamed": copy.deepcopy(doc)}


def _share_document(state: State, title: str, contact: str) -> dict:
    doc = _document(state, title)
    person = _contact(state, contact)["name"]
    shared = doc.setdefault("shared_with", [])
    if person not in shared:
        shared.append(person)
    return {"shared": copy.deepcopy(doc)}


def _math(fn):
    from .. import poly

    def run(state: State, polynomial: str) -> dict:
        try:
            return {"result": fn(poly, poly.parse(polynomial))}
        except poly.PolyError as e:
            raise ToolError(str(e)) from e
    return run


_differentiate = _math(lambda poly, p: poly.to_text(poly.differentiate(p)))
_integrate = _math(lambda poly, p: poly.to_text(poly.integrate(p)))
_factor_polynomial = _math(lambda poly, p: poly.factor_text(*poly.factor(p)))


def title_key(title: str) -> tuple[str, ...]:
    """A title as the tools tell titles apart (:func:`_same_title`): its words,
    case-insensitive, :data:`STOPWORDS` and punctuation aside. ``"Q1 vendor
    scorecard"``, ``'"Q1 vendor scorecard"'``, ``"The Q1 vendor scorecard."`` are one
    title. Execution match compares a title the model writes this way."""
    return tuple(sorted(words(title, drop_stopwords=True)))


def contacts_key(contacts: list[dict]) -> tuple:
    """The contacts as execution match compares them (ids aside): name, email and team
    case-insensitive, the phone by its digits only (``1-555-123-4567`` is
    ``+1-555-123-4567``; ``555-123-4567``, without the country code, is not)."""
    def phone(v: str) -> str:
        return "".join(ch for ch in v if ch.isdigit())
    return tuple(sorted((" ".join(c["name"].split()).casefold(), c["email"].strip().casefold(),
                         phone(c["phone"]), " ".join(c["team"].split()).casefold())
                        for c in contacts))


def documents_key(documents: list[dict]) -> tuple:
    """The documents as execution match compares them (ids aside): title by
    :func:`title_key` (the tool's own equivalence), last update, and who it is shared
    with."""
    return tuple(sorted((title_key(d["title"]), d["updated"],
                         tuple(sorted(n.casefold() for n in d.get("shared_with", []))))
                        for d in documents))


_IMPL = {
    "calculator": _calculator,
    "search_contacts": _search_contacts,
    "get_calendar_event": _get_calendar_event,
    "create_calendar_event": _create_calendar_event,
    "search_documents": _search_documents,
    # post 10's operation tools (their own registry, :data:`OPERATIONS`)
    "cancel_calendar_event": _cancel_calendar_event,
    "update_calendar_event": _update_calendar_event,
    "add_contact": _add_contact,
    "update_contact": _update_contact,
    "delete_contact": _delete_contact,
    "create_document": _create_document,
    "delete_document": _delete_document,
    "rename_document": _rename_document,
    "share_document": _share_document,
    "differentiate": _differentiate,
    "integrate": _integrate,
    "factor_polynomial": _factor_polynomial,
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
