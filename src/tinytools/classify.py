"""Split every reply into exactly one outcome, by what the task asks for.

Tasks that need a call (BFCL's AST categories; our choose+fill and tricky
arguments):

    no call              the reply has no ``<tool_call>`` tag at all
    unparseable          it has a tag, but some tagged block does not decode to
                         a call
    wrong function       the called function names differ from the reference's
                         as a set (a function missing, or one not in the
                         reference)
    wrong number of calls  the same set of names, but too many or too few calls
                         (one ``spotify.play`` where the reference has two)
    wrong arguments      the right calls, but the checker rejects the arguments
    right                the checker accepts it

Tasks where no tool fits (BFCL irrelevance; our "no tool fits"):

    declined             no ``<tool_call>`` tag (right)
    called anyway        any tag, whether or not it decodes

"Right" is the checker's own verdict on tasks that need a call, so on BFCL's
AST categories the "right" column equals the official accuracy. On irrelevance
BFCL also passes a reply whose ``<tool_call>`` blocks fail to decode; here that
reply called anyway, and the report prints BFCL's count beside ours so the
difference is visible.

Pure: the caller passes in the decode and the verdict. Tested in
``tests/test_classify.py``.
"""

from __future__ import annotations

from collections import Counter

NO_CALL = "no call"
UNPARSEABLE = "unparseable"
WRONG_FUNCTION = "wrong function"
WRONG_COUNT = "wrong number of calls"
WRONG_ARGUMENTS = "wrong arguments"
RIGHT = "right"
DECLINED = "declined"
CALLED_ANYWAY = "called anyway"
# Tasks that need a call, and tasks where no tool fits: each reply gets one
# label from its task's group, and the reports print the groups as separate tables.
CALL_OUTCOMES = (NO_CALL, UNPARSEABLE, WRONG_FUNCTION, WRONG_COUNT, WRONG_ARGUMENTS, RIGHT)
NO_FIT_OUTCOMES = (DECLINED, CALLED_ANYWAY)
OUTCOMES = CALL_OUTCOMES + NO_FIT_OUTCOMES
RIGHT_OUTCOMES = (RIGHT, DECLINED)
# Column headings for the 74-column tables. A new outcome added above without
# one here gets its full label as the heading.
SHORT = {NO_CALL: "no call", UNPARSEABLE: "unparse", WRONG_FUNCTION: "wrong fn",
         WRONG_COUNT: "wrong n", WRONG_ARGUMENTS: "wrong arg", RIGHT: "right",
         DECLINED: "declined", CALLED_ANYWAY: "called anyway"}


def short_label(outcome: str) -> str:
    return SHORT.get(outcome, outcome)


def is_right(outcome: str) -> bool:
    """Right in either group: the checker accepted the call, or a no-fit task declined."""
    return outcome in RIGHT_OUTCOMES


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
    """One reply's outcome.

    ``raw`` is the generated text, ``decoded`` the Qwen handler's ``decode_ast``
    of it (``None`` if that raised), ``bfcl_valid`` the checker's verdict (unused
    when no tool fits), ``irrelevance`` whether no tool fits, and
    ``expected_names`` the reference call's function names.
    """
    tags = raw.count(TAG)
    if irrelevance:
        return DECLINED if tags == 0 else CALLED_ANYWAY
    if bfcl_valid:
        return RIGHT
    if tags == 0:
        return NO_CALL
    if not (is_call_list(decoded) and len(decoded) >= tags):
        return UNPARSEABLE
    got = Counter(name for d in decoded for name in d)
    want = Counter(expected_names)
    if set(got) != set(want):
        return WRONG_FUNCTION
    if got != want:
        return WRONG_COUNT
    return WRONG_ARGUMENTS


def mixed_share(right_counts: list[int], samples: int) -> tuple[int, int, int]:
    """(groups all wrong, groups mixed, groups all right) for per-task right counts."""
    none = sum(1 for r in right_counts if r == 0)
    every = sum(1 for r in right_counts if r == samples)
    return none, len(right_counts) - none - every, every
