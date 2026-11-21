"""Split every BFCL item into exactly one of five outcomes.

    no call          the reply has no ``<tool_call>`` tag at all
    unparseable      it has a tag, but some tagged block does not decode to a call
    wrong function   the called functions differ from the reference: a different
                     function, or a different number of calls (as multisets of
                     names); on irrelevance items, any decodable call
    wrong arguments  the right functions, but BFCL's AST checker rejects the
                     arguments
    right            BFCL's checker accepts it; on irrelevance items, no tag

"Right" is BFCL's own verdict on every AST category, so the "right" column
equals the official accuracy there. On irrelevance BFCL also counts a reply
whose ``<tool_call>`` blocks fail to decode as "no call", which this split
calls unparseable instead; the report prints BFCL's count beside ours so the
difference is visible.

Pure: the caller passes in BFCL's decode and verdict. Tested in
``tests/test_classify.py``.
"""

from __future__ import annotations

from collections import Counter

NO_CALL = "no call"
UNPARSEABLE = "unparseable"
WRONG_FUNCTION = "wrong function"
WRONG_ARGUMENTS = "wrong arguments"
RIGHT = "right"
OUTCOMES = (NO_CALL, UNPARSEABLE, WRONG_FUNCTION, WRONG_ARGUMENTS, RIGHT)

TAG = "<tool_call>"


def is_call_list(decoded) -> bool:
    """``[{name: {arg: value}}, ...]`` with at least one call (BFCL's call format)."""
    return (
        isinstance(decoded, list)
        and len(decoded) > 0
        and all(isinstance(d, dict) and len(d) == 1 and isinstance(next(iter(d.values())), dict)
                for d in decoded)
    )


def classify(
    *,
    raw: str,
    decoded: list | None,
    bfcl_valid: bool,
    irrelevance: bool,
    expected_names: list[str] = (),
) -> str:
    """One item's outcome.

    ``raw`` is the generated text, ``decoded`` the BFCL handler's ``decode_ast``
    of it (``None`` if that raised), ``bfcl_valid`` BFCL's verdict, and
    ``expected_names`` the reference call's function names (AST categories).
    """
    tags = raw.count(TAG)
    decodes = is_call_list(decoded) and len(decoded) >= tags
    if irrelevance:
        if tags == 0:
            return RIGHT
        return WRONG_FUNCTION if decodes else UNPARSEABLE
    if bfcl_valid:
        return RIGHT
    if tags == 0:
        return NO_CALL
    if not decodes:
        return UNPARSEABLE
    got = Counter(name for d in decoded for name in d)
    if got != Counter(expected_names):
        return WRONG_FUNCTION
    return WRONG_ARGUMENTS


def only_count_differs(decoded: list, expected_names: list[str]) -> bool:
    """Within "wrong function": the same set of names, called a different number of times.

    E.g. one ``spotify.play`` call where the reference has two. Reported as a
    sub-count so a missing parallel call isn't read as choosing the wrong tool.
    """
    got = [name for d in decoded or [] for name in d]
    return set(got) == set(expected_names) and Counter(got) != Counter(expected_names)


def mixed_share(right_counts: list[int], samples: int) -> tuple[int, int, int]:
    """(groups all wrong, groups mixed, groups all right) for per-task right counts."""
    none = sum(1 for r in right_counts if r == 0)
    every = sum(1 for r in right_counts if r == samples)
    return none, len(right_counts) - none - every, every
