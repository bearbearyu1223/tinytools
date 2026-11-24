"""The tag-free reading of a reply: calls written as bare JSON count as calls (plan 3f).

The strict reading (every report so far) sees a call only inside
``<tool_call>`` ... ``</tool_call>`` tags. The untrained Qwen2.5-1.5B base model
cannot emit those tags (their embedding rows are untrained), but often writes the
call itself as bare JSON, e.g. ``{"name": "calculator", "arguments": {...}}``
followed by unrelated text to the token cap (T9e review S1). The tag-free reading
counts such calls, by this rule:

1. If the reply contains a ``<tool_call>`` tag, the tag-free reading is the strict
   one: the same reply, the same calls.
2. Otherwise, the call is **the first call object at the start of the reply**
   (leading whitespace skipped): a JSON object whose keys are exactly ``name``
   and ``arguments``, ``name`` a string and ``arguments`` a JSON object. At most
   one call: ``arguments`` written as a string, even one that parses to an
   object, is not accepted, because the strict decoder does not accept it inside
   tags either (it reads ``arguments`` with ``.items()``).
3. Everything after that first object is ignored, whatever it is, including
   more call objects and repeats of the same call (the untrained base model
   often writes one call over and over to the token cap; Han, 2026-11-23). No
   call object at the start: no call.

The reading is applied by rewriting the reply: a reply that starts with a call
object becomes that object in ``<tool_call>`` tags as Qwen writes it
(:func:`tagfree_reply`), and every other reply is left as it is. The rewritten
reply then goes through the same decoder, classifier and checkers as the strict
reading (ours: :func:`tinytools.checkers.check_reply`; BFCL:
:func:`tinytools.bfcl.judge`), so the two readings differ only in what counts as
a call. A reply the rule leaves alone gets the strict outcome by construction.

Pure; tested in ``tests/test_tagfree.py``.
"""

from __future__ import annotations

import json

from .classify import TAG

__all__ = ["READINGS", "STRICT", "TAGFREE", "RULE", "is_call_object", "bare_calls",
           "tagfree_reply", "read"]

STRICT, TAGFREE = "strict", "tagfree"
READINGS = (STRICT, TAGFREE)
END_TAG = "</tool_call>"

# The rule in plain words, as the reports print it (one string per line, <= 70 columns).
RULE = (
    "strict: a call counts only inside <tool_call> tags.",
    "tag-free: a reply with a <tool_call> tag is read as strict. Any",
    "other reply's call is the JSON object at its very start, if it has",
    "exactly the keys name (a string) and arguments (an object); else",
    "no call. One call at most: everything after that first object is",
    "ignored, including more call objects and repeats of the same call.",
    "Arguments written as a string are not a call (strict rejects them",
    "too). The call then goes through the same checkers as strict.",
)

_DECODER = json.JSONDecoder()


def is_call_object(obj) -> bool:
    """``{"name": <str>, "arguments": <object>}``, with no other key."""
    return (isinstance(obj, dict) and set(obj) == {"name", "arguments"}
            and isinstance(obj["name"], str) and isinstance(obj["arguments"], dict))


def bare_calls(text: str) -> list[dict]:
    """The call at the start of ``text`` (rule 2 above): ``[call]``, or ``[]`` if the
    text (after leading whitespace) does not start with a call object. Anything after
    the first object is never read."""
    i = len(text) - len(text.lstrip())
    if not text.startswith("{", i):
        return []
    try:
        obj, _ = _DECODER.raw_decode(text, i)
    except ValueError:
        return []
    return [obj] if is_call_object(obj) else []


def tagfree_reply(raw: str) -> str:
    """The reply the strict checkers read on the tag-free reading.

    ``raw`` itself if it has a ``<tool_call>`` tag or no call object at its start;
    else that one call written ``<tool_call>\\n{json}\\n</tool_call>`` (Qwen's
    format, what the strict decoder reads).
    """
    if TAG in raw:
        return raw
    calls = bare_calls(raw)
    if not calls:
        return raw
    return "\n".join(f"{TAG}\n{_dump(c)}\n{END_TAG}" for c in calls)


def _dump(call: dict) -> str:
    """One call object as JSON. ``<`` can only occur inside a string, where ``\\u003c``
    is the same character, so a string holding a tag can't add one to the reply."""
    text = json.dumps({"name": call["name"], "arguments": call["arguments"]})
    return text.replace("<", "\\u003c")


def read(raw: str, reading: str) -> str:
    """The text the checkers see for ``raw`` on ``reading`` (``strict``: ``raw``)."""
    if reading == STRICT:
        return raw
    if reading == TAGFREE:
        return tagfree_reply(raw)
    raise ValueError(f"unknown reading {reading!r}; one of {READINGS}")
