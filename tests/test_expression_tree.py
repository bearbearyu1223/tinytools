"""Execution match on the calculator compares normalized trees, then numbers (code review 1, f1).

And blank optional arguments count as omitted (code review 1, f2).
"""

import json

import pytest

from tinytools.checkers import (check_reply, expression_tree, normalize_arguments,
                                same_percent_work)
from tinytools.tools import execute

CALC = {"name": "calculator", "arguments": {"expression": "348 * 27"}}


def reply(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def calc(expr, reference="348 * 27"):
    c = check_reply(reply("calculator", {"expression": expr}),
                    {"name": "calculator", "arguments": {"expression": reference}})
    return c.exact, c.execution


@pytest.mark.parametrize("a, b", [
    ("348 * 27", "27 * 348"),
    ("(348) * (27)", "348*27"),
    ("348.0 * 27", "348 * 27.00"),
    ("2 * 3 * 4", "4 * (3 * 2)"),
    ("1 + 2 + 3", "3 + (1 + 2)"),
    ("(3 + 4) * 2", "2 * (4 + 3)"),
    ("5 ** 2", "(5) ** 2"),
    ("348 * 27", "348 * 27 * 1"),
    ("348 * 27", "1 * 348 * 27 + 0"),
    ("10 - 4", "10 - 4 - 0"),
    ("10 / 4", "10 / 4 / 1"),
    ("-5", "-(5)"),
    ("--5", "5"),
    ("-(-(2 + 3))", "2 + 3"),
    ("+(2 + 3)", "2 + 3"),
])
def test_same_tree(a, b):
    assert expression_tree(a) == expression_tree(b) is not None


@pytest.mark.parametrize("a, b", [
    ("348 * 27", "9396"),
    ("10 - 4", "4 - 10"),  # - and / keep their order
    ("10 / 4", "4 / 10"),
    ("2 ** 3", "3 ** 2"),
    ("1 + 2 * 3", "(1 + 2) * 3"),
    ("3438 * 48 / 100", "3438 * 0.48"),  # no constant folding
    ("3438 * 48 / 100", "3438 * (48 / 100)"),
    ("10 - 4", "10 + -4"),  # subtraction is not rewritten as adding a negative
])
def test_different_tree(a, b):
    assert expression_tree(a) != expression_tree(b)


@pytest.mark.parametrize("expr", ["abc", "1 +", "f(2)", "1, 2", "'7'", 12, None, "2 ^ 3"])
def test_not_arithmetic(expr):
    assert expression_tree(expr) is None


@pytest.mark.parametrize("expr", ["9396", "--9396", "-(-9396)", "+(+9396)", "9396 + 0", "0 + 9396",
                                  "9396 * 1", "1 * 9396", "9396 - 0", "9396 / 1", "9396 // 1",
                                  "9396 ** 1", "(9396 * 1) + 0", "9396.0 * 1.0"])
def test_hand_worked_answer_fails_execution(expr):
    assert calc(expr) == (False, False)


def test_commuted_work_passes_execution_only():
    assert calc("27 * 348") == (False, True)
    assert calc("348 * 27") == (True, True)
    assert calc("12.5 + 3 + 4", "4 + 12.5 + 3") == (False, True)


def test_same_tree_wrong_number_still_fails():
    assert calc("348 * 28") == (False, False)


def test_bare_reference_matches_bare_reply():
    assert calc("-4.0", "-4") == (True, True)
    assert calc("--4", "4") == (False, True)


# -- the percent rule (Han, 2026-11-21) ------------------------------------------------------

PCT = "3438 * 48 / 100"


@pytest.mark.parametrize("expr", ["3438 * 0.48", "0.48 * 3438", "3438 * (48 / 100)",
                                  "(48 / 100) * 3438", "3438 * 0.480", "(3438) * .48"])
def test_percent_as_a_decimal_or_a_fraction_passes_execution(expr):
    # test-0437 in the 0.5B test capture wrote "3438 * 0.48"
    assert calc(expr, PCT) == (False, True)


def test_test_0964_passes_execution():
    assert calc("1062 * 0.68", "1062 * 68 / 100") == (False, True)


@pytest.mark.parametrize("expr", ["1650.24", "1650.24 * 1", "1650.24 * 100 / 100", "--1650.24",
                                  "1650.24 + 0"])
def test_hand_worked_percent_answer_still_fails(expr):
    assert calc(expr, PCT) == (False, False)


@pytest.mark.parametrize("expr", ["9396 * 1", "--9396", "9396"])
def test_percent_rule_leaves_other_hand_worked_answers_failing(expr):
    assert calc(expr) == (False, False)


@pytest.mark.parametrize("expr", ["3438 * 4.8", "3438 * 0.048", "3438 * 0.49", "3437 * 0.48",
                                  "3438 * 0.48 + 1", "3438 * (48 / 10)", "3438 * (48 // 100)",
                                  "3438 * 48 / 1000"])
def test_percent_rule_needs_exactly_100_and_the_rest_of_the_tree(expr):
    assert calc(expr, PCT)[1] is False


@pytest.mark.parametrize("expr, reference", [
    ("2.5", "250 / 100"),          # cm to m: a standalone n / 100 is not a decimal
    ("250 * 0.01", "250 / 100"),   # nor a product with 0.01 (the reference is no product)
    ("2500", "2.5 * 1000"),        # km/kg: no digits x 10^k rule
    ("25 * 100", "2.5 * 1000"),
    ("2.5 * 1000 / 100", "2.5 * 1000"),
])
def test_unit_traps_stay_failing(expr, reference):
    assert calc(expr, reference) == (False, False)


def test_percent_rule_is_symmetric_in_the_operands():
    # the rule doesn't know which operand of x * p is the percent: x / 100 passes too
    assert calc("48 * 34.38", PCT) == (False, True)


def test_percent_rule_trees():
    want = expression_tree(PCT)
    assert same_percent_work(expression_tree("3438 * 0.48"), want)
    assert not same_percent_work(expression_tree("3438 * 0.48"), expression_tree("3438 * 48"))
    assert not same_percent_work(None, want)
    assert not same_percent_work(expression_tree("3438 * 0.48"), None)
    assert expression_tree("3438 * 0.48") != want  # the trees still differ: the rule is separate


# -- blank optional arguments ---------------------------------------------------------------

DAY = {"name": "get_calendar_event", "arguments": {"date": "2026-02-23"}}


@pytest.mark.parametrize("title", ["", "   ", None, "meetings", "my events", "the meeting"])
def test_blank_or_filler_title_with_a_date_is_no_title(title):
    c = check_reply(reply("get_calendar_event", {"date": "2026-02-23", "title": title}), DAY)
    assert c.execution
    alone = execute("get_calendar_event", {"date": "2026-02-23"})
    assert c.reply_run.result == alone


@pytest.mark.parametrize("title", ["", None])
def test_blank_title_counts_as_omitted_in_exact_match_too(title):
    c = check_reply(reply("get_calendar_event", {"date": "2026-02-23", "title": title}), DAY)
    assert c.exact


def test_filler_title_is_not_exact():
    c = check_reply(reply("get_calendar_event", {"date": "2026-02-23", "title": "meetings"}), DAY)
    assert (c.exact, c.execution) == (False, True)


@pytest.mark.parametrize("args", [{"title": ""}, {"title": None}, {"title": "meetings"},
                                  {"date": "", "title": ""}, {"date": None}, {}])
def test_no_date_and_no_real_title_is_still_an_error(args):
    assert "error" in execute("get_calendar_event", args)


def test_a_real_title_with_a_date_still_narrows():
    both = execute("get_calendar_event", {"date": "2026-03-11", "title": "zzz-no-such"})
    assert both == {"events": []}


@pytest.mark.parametrize("args, err", [
    ({"title": "", "date": "2026-03-17", "start_time": "15:00"}, "title"),
    ({"title": None, "date": "2026-03-17", "start_time": "15:00"}, "title"),
    ({"title": "x", "date": " ", "start_time": "15:00"}, "date"),
])
def test_required_arguments_stay_strict(args, err):
    out = execute("create_calendar_event", args)
    assert "error" in out and err in out["error"]


def test_blank_optional_create_arguments_take_their_defaults():
    base = {"title": "x", "date": "2026-03-17", "start_time": "15:00"}
    plain = execute("create_calendar_event", base)
    for extra in ({"duration_minutes": None}, {"attendees": None}, {"duration_minutes": ""}):
        assert execute("create_calendar_event", {**base, **extra}) == plain


def test_unknown_arguments_stay_errors_even_when_blank():
    assert "error" in execute("get_calendar_event", {"date": "2026-02-23", "room": None})


def test_normalize_arguments_drops_blank_optional_only_for_known_tools():
    assert normalize_arguments("get_calendar_event", {"date": "d", "title": " "}) \
        == normalize_arguments("get_calendar_event", {"date": "d"})
    assert normalize_arguments("f", {"x": None}) != normalize_arguments("f", {})
