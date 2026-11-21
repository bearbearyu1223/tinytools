"""Parse ToolACE's call syntax: ``[Name(arg=value, ...), Other Name(...)]``.

ToolACE writes a tool call as Python-looking text in square brackets, but it
is not Python: tool names contain spaces (``Get Current Time by IP``), slashes
(``/email/valistring``) and dots, so ``ast.parse`` rejects most rows. This
module scans the text by hand: a name runs up to its ``(``, arguments are
``key=value`` separated by top-level commas, and each value is a Python
literal (``ast.literal_eval``) or, failing that, a JSON literal (``true``,
``null``). Commas, brackets and parentheses inside quoted strings, lists and
dicts are skipped over, so ``q="a, b (c)"`` is one argument.

Some names contain parentheses or apostrophes themselves (``Daily Forecast
(10 days)``, ``Today's Wordle Solution``), so when the tool list is known the
parser first matches the longest listed name followed by ``(``, and only falls
back to "everything up to the first ``(``" for names not on the list.

Model-free and pure, so it is unit-tested in ``tests/test_calls.py``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

__all__ = ["Call", "CallParseError", "looks_like_call", "parse_calls"]

_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")", "]", "}"}


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict[str, Any]


class CallParseError(ValueError):
    """Raised with a short, countable reason (``err.reason``)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def looks_like_call(text: str) -> bool:
    """Is this assistant turn written as a call? Starts with ``[`` and ends with ``]``.

    This is the rule the statistics use to split assistant turns into calls and
    not-calls; a turn that looks like a call but fails :func:`parse_calls` is
    counted as a parse failure, not as a not-call.
    """
    t = text.strip()
    return t.startswith("[") and t.endswith("]")


def _skip_string(s: str, i: int) -> int:
    """``s[i]`` opens a quoted string; return the index just past its close."""
    for q in ('"""', "'''"):
        if s.startswith(q, i):
            end = s.find(q, i + 3)
            if end < 0:
                raise CallParseError("unterminated string", s[i:i + 40])
            return end + 3
    quote = s[i]
    j = i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == quote:
            return j + 1
        j += 1
    raise CallParseError("unterminated string", s[i:i + 40])


def _scan_value(s: str, i: int) -> int:
    """Return the index of the top-level ``,`` or ``)`` that ends the value at ``i``."""
    stack: list[str] = []
    j = i
    while j < len(s):
        c = s[j]
        if c in "\"'":
            j = _skip_string(s, j)
            continue
        if c in _OPEN:
            stack.append(_OPEN[c])
        elif c in _CLOSE:
            if not stack:
                if c == ")":
                    return j
                raise CallParseError("unbalanced brackets", s[i:j + 1][:60])
            if stack.pop() != c:
                raise CallParseError("unbalanced brackets", s[i:j + 1][:60])
        elif c == "," and not stack:
            return j
        j += 1
    raise CallParseError("unbalanced brackets", s[i:][:60])


def _literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    try:
        return json.loads(text)
    except ValueError:
        raise CallParseError("value not a literal", text[:60]) from None


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _match_name(s: str, i: int, names: list[str]) -> tuple[str, int] | None:
    """The longest listed name at ``s[i:]`` followed by optional spaces and ``(``."""
    for name in names:  # sorted longest first
        if s.startswith(name, i):
            j = _skip_ws(s, i + len(name))
            if j < len(s) and s[j] == "(":
                return name, j
    return None


def parse_calls(text: str, names: list[str] | tuple[str, ...] = ()) -> list[Call]:
    """Parse one assistant turn into its calls, or raise :class:`CallParseError`.

    ``names`` is the tool list's names, used to read names that contain ``(``.
    """
    ordered = sorted(set(names), key=len, reverse=True)
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise CallParseError("not bracketed", s[:40])
    s = s[1:-1]
    calls: list[Call] = []
    i = _skip_ws(s, 0)
    while i < len(s):
        listed = _match_name(s, i, ordered)
        if listed:
            name, paren = listed
        else:
            paren = s.find("(", i)
            if paren < 0:
                raise CallParseError("no '(' after name", s[i:i + 40])
            name = s[i:paren].strip()
            if not name or any(c in name for c in "\"'[]{}()=,"):
                raise CallParseError("bad function name", name[:40])
        args: dict[str, Any] = {}
        i = _skip_ws(s, paren + 1)
        while i < len(s) and s[i] != ")":
            eq = s.find("=", i)
            if eq < 0:
                raise CallParseError("argument without '='", s[i:i + 40])
            key = s[i:eq].strip()
            if not key or not all(c.isalnum() or c in "_-. $@" for c in key):
                raise CallParseError("bad argument name", key[:40])
            start = _skip_ws(s, eq + 1)
            end = _scan_value(s, start)
            raw = s[start:end].strip()
            if not raw:
                raise CallParseError("empty argument value", key)
            if key in args:
                raise CallParseError("duplicate argument", key)
            args[key] = _literal(raw)
            i = _skip_ws(s, end)
            if i < len(s) and s[i] == ",":
                i = _skip_ws(s, i + 1)
        if i >= len(s):
            raise CallParseError("unbalanced brackets", name)
        calls.append(Call(name, args))
        i = _skip_ws(s, i + 1)  # past ')'
        if i < len(s):
            if s[i] != ",":
                raise CallParseError("text between calls", s[i:i + 40])
            i = _skip_ws(s, i + 1)
    if not calls:
        raise CallParseError("no calls in brackets")
    return calls
