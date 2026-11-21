"""The outcome labels on hand-made replies."""

import pytest

from tinytools.classify import (
    CALL_OUTCOMES, CALLED_ANYWAY, DECLINED, NO_CALL, NO_FIT_OUTCOMES, OUTCOMES, RIGHT,
    UNPARSEABLE, WRONG_ARGUMENTS, WRONG_COUNT, WRONG_FUNCTION, classify, is_call_list, is_right,
    mixed_share, short_label,
)

CALL = '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>'


def ast(raw, decoded, valid=False, expected=("f",)):
    return classify(raw=raw, decoded=decoded, bfcl_valid=valid, irrelevance=False,
                    expected_names=list(expected))


def test_right_is_bfcls_verdict():
    assert ast(CALL % ("f", '{"x": 1}'), [{"f": {"x": 1}}], valid=True) == RIGHT


def test_no_tag_is_no_call_even_if_it_looks_like_python():
    assert ast("I would call f(x=1).", [], valid=False) == NO_CALL


@pytest.mark.parametrize("raw, decoded", [
    ('<tool_call>\n{"name": "f", "arguments": {"x": 1\n</tool_call>', []),     # bad JSON
    ('<tool_call>{"name": "f", "arguments": {}}</tool_call>', []),              # no newlines
    ('<tool_call>\n{"name": "f", "arguments": "x=1"}\n</tool_call>', None),     # decode raised
    ('<tool_call>\n{"name": "f", "arguments": {"x": 1', []),                    # truncated
])
def test_tag_without_a_decodable_call_is_unparseable(raw, decoded):
    assert ast(raw, decoded) == UNPARSEABLE


def test_one_of_two_blocks_failing_is_unparseable():
    raw = CALL % ("f", '{"x": 1}') + "\n<tool_call>\n{broken\n</tool_call>"
    assert ast(raw, [{"f": {"x": 1}}], expected=("f", "f")) == UNPARSEABLE


def test_wrong_name_is_wrong_function():
    assert ast(CALL % ("g", "{}"), [{"g": {}}]) == WRONG_FUNCTION
    # a right call plus one to a tool the reference doesn't call: the sets differ
    raw = CALL % ("f", "{}") + "\n" + CALL % ("g", "{}")
    assert ast(raw, [{"f": {}}, {"g": {}}], expected=("f",)) == WRONG_FUNCTION
    # one of two reference functions missing
    assert ast(CALL % ("f", "{}"), [{"f": {}}], expected=("f", "g")) == WRONG_FUNCTION


def test_right_names_wrong_count_is_its_own_outcome():
    raw = CALL % ("f", "{}")
    assert ast(raw, [{"f": {}}], expected=("f", "f")) == WRONG_COUNT  # too few
    assert ast(raw + "\n" + raw, [{"f": {}}, {"f": {}}], expected=("f",)) == WRONG_COUNT
    two = raw + "\n" + CALL % ("g", "{}")
    assert ast(two + "\n" + raw, [{"f": {}}, {"g": {}}, {"f": {}}],
               expected=("g", "f")) == WRONG_COUNT


def test_right_names_wrong_values_is_wrong_arguments():
    raw = CALL % ("f", '{"x": 2}') + "\n" + CALL % ("g", "{}")
    assert ast(raw, [{"f": {"x": 2}}, {"g": {}}], expected=("g", "f")) == WRONG_ARGUMENTS


def irr(raw, decoded, valid=True):
    return classify(raw=raw, decoded=decoded, bfcl_valid=valid, irrelevance=True)


def test_no_tool_fits_declined_or_called_anyway():
    assert irr("None of the functions can do that.", [], valid=False) == DECLINED
    assert irr(CALL % ("f", "{}"), [{"f": {}}]) == CALLED_ANYWAY
    # BFCL would count this one right (nothing decodes); here it called anyway.
    assert irr("<tool_call>\n{oops\n</tool_call>", []) == CALLED_ANYWAY


def test_groups_and_right():
    assert set(CALL_OUTCOMES) & set(NO_FIT_OUTCOMES) == set()
    assert OUTCOMES == CALL_OUTCOMES + NO_FIT_OUTCOMES
    assert [o for o in OUTCOMES if is_right(o)] == [RIGHT, DECLINED]
    # the call table fits 74 columns: an 11-character first column (choose+fill,
    # tricky args), a 4-digit n, six outcome columns of up to 4 digits each
    width = 2 + 11 + 4 + sum(max(4, len(short_label(o))) for o in CALL_OUTCOMES) + 2 * 7
    assert width <= 74


def test_is_call_list():
    assert is_call_list([{"f": {}}])
    assert not is_call_list([])
    assert not is_call_list([{}])
    assert not is_call_list([{"f": "x"}])
    assert not is_call_list("f()")


def test_mixed_share():
    assert mixed_share([0, 3, 8, 8, 1, 0], 8) == (2, 2, 2)
