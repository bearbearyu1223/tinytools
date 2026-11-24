"""The tag-free reading of a reply: calls written as bare JSON count as calls (plan 3f).

The strict reading (every report so far) sees a call only inside
``<tool_call>`` ... ``</tool_call>`` tags. The untrained Qwen2.5-1.5B base model
cannot emit those tags (their embedding rows are untrained), but often writes the
call itself as bare JSON, e.g. ``{"name": "calculator", "arguments": {...}}``
followed by unrelated text to the token cap (T9e review S1). The tag-free reading
counts such calls, by this rule:

1. If the reply contains a ``<tool_call>`` tag, the tag-free reading is the strict
   one: the same reply, the same calls.
2. Otherwise, the calls are the **call objects at the start of the reply**
   (leading whitespace skipped): JSON objects whose keys are exactly ``name`` and
   ``arguments``, ``name`` a string and ``arguments`` a JSON object, one after
   another with nothing but whitespace (or nothing) between them, taken until the
   first thing that is not one. ``arguments`` written as a string, even one that
   parses to an object, is not accepted, because the strict decoder does not
   accept it inside tags either (it reads ``arguments`` with ``.items()``).
3. Everything after the last call object is ignored, whatever it is: that is
   what "tag-free" means. No call object at the start: no call.

The reading is applied by rewriting the reply: a reply with call objects becomes
those objects, each in ``<tool_call>`` tags as Qwen writes them
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
    "other reply's calls are the JSON objects at its start with exactly",
    "the keys name (a string) and arguments (an object), one after",
    "another with only whitespace between, up to the first thing that",
    "is not one; everything after them is ignored. None: no call.",
    "Arguments written as a string are not a call (strict rejects them",
    "too). The calls then go through the same checkers as strict.",
)

_DECODER = json.JSONDecoder()


def is_call_object(obj) -> bool:
    """``{"name": <str>, "arguments": <object>}``, with no other key."""
    return (isinstance(obj, dict) and set(obj) == {"name", "arguments"}
            and isinstance(obj["name"], str) and isinstance(obj["arguments"], dict))


def bare_calls(text: str) -> list[dict]:
    """The call objects at the start of ``text`` (rule 2 above), in order."""
    calls: list[dict] = []
    i = 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            return calls
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except ValueError:
            return calls
        if not is_call_object(obj):
            return calls
        calls.append(obj)
        i = end


def tagfree_reply(raw: str) -> str:
    """The reply the strict checkers read on the tag-free reading.

    ``raw`` itself if it has a ``<tool_call>`` tag or no call object at its start;
    else its call objects, each written ``<tool_call>\\n{json}\\n</tool_call>``
    (Qwen's format, what the strict decoder reads), joined by newlines.
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
