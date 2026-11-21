"""The five-way split on hand-made replies."""

import pytest

from tinytools.classify import (
    NO_CALL, RIGHT, UNPARSEABLE, WRONG_ARGUMENTS, WRONG_FUNCTION, classify, is_call_list,
    mixed_share,
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


def test_wrong_name_or_wrong_count_is_wrong_function():
    assert ast(CALL % ("g", "{}"), [{"g": {}}]) == WRONG_FUNCTION
    raw = CALL % ("f", "{}")
    assert ast(raw, [{"f": {}}], expected=("f", "f")) == WRONG_FUNCTION
    assert ast(raw + "\n" + raw, [{"f": {}}, {"f": {}}], expected=("f",)) == WRONG_FUNCTION


def test_right_names_wrong_values_is_wrong_arguments():
    raw = CALL % ("f", '{"x": 2}') + "\n" + CALL % ("g", "{}")
    assert ast(raw, [{"f": {"x": 2}}, {"g": {}}], expected=("g", "f")) == WRONG_ARGUMENTS


def irr(raw, decoded):
    return classify(raw=raw, decoded=decoded, bfcl_valid=True, irrelevance=True)


def test_irrelevance():
    assert irr("None of the functions can do that.", []) == RIGHT
    assert irr(CALL % ("f", "{}"), [{"f": {}}]) == WRONG_FUNCTION
    # BFCL would count this one right (nothing decodes); the split says unparseable.
    assert irr("<tool_call>\n{oops\n</tool_call>", []) == UNPARSEABLE


def test_is_call_list():
    assert is_call_list([{"f": {}}])
    assert not is_call_list([])
    assert not is_call_list([{}])
    assert not is_call_list([{"f": "x"}])
    assert not is_call_list("f()")


def test_mixed_share():
    assert mixed_share([0, 3, 8, 8, 1, 0], 8) == (2, 2, 2)


def test_only_count_differs():
    from tinytools.classify import only_count_differs
    assert only_count_differs([{"f": {}}], ["f", "f"])
    assert not only_count_differs([{"g": {}}], ["f"])
    assert not only_count_differs([{"f": {}}, {"g": {}}], ["f", "f"])
