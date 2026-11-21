"""Exact match and execution match on hand-written replies, including the ways to game execution."""

import json

import pytest

from tinytools.checkers import check_reply, exact_match, execution_match, normalize_arguments
from tinytools.classify import NO_CALL, RIGHT, UNPARSEABLE, WRONG_ARGUMENTS, WRONG_FUNCTION


def reply(*calls, text=""):
    blocks = ['<tool_call>\n%s\n</tool_call>' % json.dumps({"name": n, "arguments": a})
              for n, a in calls]
    return text + "\n".join(blocks)


def ref(name, **args):
    return {"name": name, "arguments": args}


CALC = ref("calculator", expression="348 * 27")
CONTACT = ref("search_contacts", query="Siobhan O'Neill")
DOC = ref("search_documents", query="Q3 sales report")
CREATE = ref("create_calendar_event", title="Budget sync", date="2026-03-17",
             start_time="15:00", duration_minutes=90)


def both(raw, reference):
    c = check_reply(raw, reference)
    return c.exact, c.execution


def test_identical_call_passes_both():
    assert both(reply(("calculator", {"expression": "348 * 27"})), CALC) == (True, True)


def test_normalization_order_numbers_whitespace():
    swapped = {"duration_minutes": 90.0, "start_time": "15:00", "date": "2026-03-17",
               "title": "Budget  sync "}
    assert both(reply(("create_calendar_event", swapped)), CREATE) == (True, True)
    assert both(reply(("calculator", {"expression": "348*27"})), CALC) == (True, True)
    assert both(reply(("calculator", {"expression": "(348.0 * 27)"})), CALC) == (True, True)


def test_execution_forgives_what_exact_does_not():
    assert both(reply(("calculator", {"expression": "27 * 348"})), CALC) == (False, True)
    assert both(reply(("calculator", {"expression": "9396"})), CALC) == (False, True)
    assert both(reply(("search_contacts", {"query": "siobhan o'neill"})), CONTACT) == (False, True)
    assert both(reply(("search_contacts", {"query": "Siobhan"})), CONTACT) == (False, True)
    no_default = {"title": "budget sync", "date": "2026-03-17", "start_time": "15:00",
                  "duration_minutes": 90, "attendees": []}
    assert both(reply(("create_calendar_event", no_default)), CREATE) == (False, True)


def test_broad_or_narrow_searches_fail_execution():
    assert both(reply(("search_documents", {"query": "sales report"})), DOC) == (False, False)
    assert both(reply(("search_contacts", {"query": "O'Neill Siobhan Smith"})), CONTACT) \
        == (False, False)


def test_empty_results_never_match():
    # A query that finds nothing must not pass against a reference that finds nothing either:
    # the generator guarantees references find something; errors never match.
    bad = ref("search_contacts", query="Nobody Atall")
    assert not execution_match(reply(("search_contacts", {"query": "Also Nobody"})), None, bad)


def test_create_compared_by_resulting_state():
    wrong_time = dict(CREATE["arguments"], start_time="03:00")
    extra_guest = dict(CREATE["arguments"], attendees=["Maria Garcia"])
    default_len = {k: v for k, v in CREATE["arguments"].items() if k != "duration_minutes"}
    for args in (wrong_time, extra_guest, default_len):
        assert both(reply(("create_calendar_event", args)), CREATE) == (False, False)


def test_extra_calls_errors_and_other_tools_fail():
    right = ("calculator", {"expression": "348 * 27"})
    c = check_reply(reply(right, right), CALC)
    assert (c.exact, c.execution, c.outcome) == (False, False, WRONG_FUNCTION)
    c = check_reply(reply(right, ("search_contacts", {"query": "x"})), CALC)
    assert (c.execution, c.outcome) == (False, WRONG_FUNCTION)
    c = check_reply(reply(("calculator", {"expression": "348 * 27", "round": 2})), CALC)
    assert (c.execution, c.outcome) == (False, WRONG_ARGUMENTS)  # unknown argument: an error
    c = check_reply(reply(("search_documents", {"query": "348 * 27"})), CALC)
    assert c.outcome == WRONG_FUNCTION


def test_outcomes_use_classify():
    assert check_reply("It is 9396.", CALC).outcome == NO_CALL
    assert check_reply("<tool_call>\n{oops\n</tool_call>", CALC).outcome == UNPARSEABLE
    assert check_reply(reply(("calculator", {"expression": "348 * 28"})), CALC).outcome \
        == WRONG_ARGUMENTS
    assert check_reply(reply(("calculator", {"expression": "348 * 27"})), CALC).outcome == RIGHT


def test_no_tool_fits():
    assert both("I can't send email with these tools.", None) == (True, True)
    c = check_reply(reply(("search_contacts", {"query": "Maria Garcia"})), None)
    assert (c.exact, c.execution, c.outcome) == (False, False, WRONG_FUNCTION)
    c = check_reply("<tool_call>\n{oops\n</tool_call>", None)
    assert (c.execution, c.outcome) == (False, UNPARSEABLE)


def test_two_tags_one_broken_is_not_a_single_call():
    raw = reply(("calculator", {"expression": "348 * 27"})) + "\n<tool_call>\n{x\n</tool_call>"
    c = check_reply(raw, CALC)
    assert (c.exact, c.execution, c.outcome) == (False, False, UNPARSEABLE)


@pytest.mark.parametrize("a, b, same", [
    ({"x": 60}, {"x": 60.0}, True),
    ({"x": "60"}, {"x": 60}, False),
    ({"x": "a  b"}, {"x": " a b"}, True),
    ({"x": "A b"}, {"x": "a b"}, False),
    ({"x": ["a", "b"]}, {"x": ["b", "a"]}, False),
])
def test_normalize_arguments(a, b, same):
    assert (normalize_arguments("f", a) == normalize_arguments("f", b)) is same


def test_exact_match_needs_the_right_name():
    assert not exact_match(reply(("calc", {"expression": "348 * 27"})),
                           [{"calc": {"expression": "348 * 27"}}], CALC)
