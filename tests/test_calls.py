"""The ToolACE call parser, on the shapes the real file contains."""

import pytest

from tinytools.calls import Call, CallParseError, looks_like_call, parse_calls


def test_name_with_spaces():
    assert parse_calls('[Get Current Time by IP(ipv4="192.168.1.1")]') == [
        Call("Get Current Time by IP", {"ipv4": "192.168.1.1"})]


def test_names_with_slashes_dots_and_no_arguments():
    assert parse_calls("[/email/valistring(email=\"a@b.co\"), Math.pi()]") == [
        Call("/email/valistring", {"email": "a@b.co"}), Call("Math.pi", {})]


def test_multiple_calls_in_one_turn():
    calls = parse_calls('[SEC Filings(identifier="AAPL"), Mobility API(string="2025-05-17")]')
    assert [c.name for c in calls] == ["SEC Filings", "Mobility API"]
    assert calls[1].arguments == {"string": "2025-05-17"}


def test_nested_values():
    calls = parse_calls(
        '[plan(stops=[{"city": "Paris", "days": 2}, {"city": "Rome", "days": [1, 2]}], '
        'opts={"a": (1, 2)}, n=-3.5, flag=True, none=None)]')
    assert calls == [Call("plan", {
        "stops": [{"city": "Paris", "days": 2}, {"city": "Rome", "days": [1, 2]}],
        "opts": {"a": (1, 2)}, "n": -3.5, "flag": True, "none": None})]


def test_strings_with_commas_brackets_and_quotes():
    calls = parse_calls(
        "[search(q=\"a, b (c) [d] {e}\", note='it\\'s \"fine\"', eq=\"x=1\"), next(k=1)]")
    assert calls[0].arguments == {"q": "a, b (c) [d] {e}", "note": 'it\'s "fine"', "eq": "x=1"}
    assert calls[1] == Call("next", {"k": 1})


def test_json_literals_are_accepted():
    assert parse_calls("[f(a=true, b=null, c=[true, false])]")[0].arguments == {
        "a": True, "b": None, "c": [True, False]}


def test_listed_names_containing_parentheses_and_apostrophes():
    names = ["Daily Forecast (10 days)", "Today's Wordle Solution", "Daily Forecast"]
    calls = parse_calls(
        "[Daily Forecast (10 days)(latitude=\"34.05\"), Today's Wordle Solution()]", names)
    assert calls == [Call("Daily Forecast (10 days)", {"latitude": "34.05"}),
                     Call("Today's Wordle Solution", {})]


def test_unlisted_names_fall_back_to_first_paren():
    # Without the tool list, a parenthesised name can't be read: the reason says so.
    with pytest.raises(CallParseError) as err:
        parse_calls('[Daily Forecast (10 days)(latitude="34.05")]')
    assert err.value.reason == "bad argument name"


@pytest.mark.parametrize("text, reason", [
    ("Here you go", "not bracketed"),
    ("[]", "no calls in brackets"),
    ('[f(a="x)]', "unterminated string"),
    ("[f(a=[1, 2)]", "unbalanced brackets"),
    ("[f(a=unquoted)]", "value not a literal"),
    ("[f(a=1, a=2)]", "duplicate argument"),
    ("[f(a=1) g(b=2)]", "text between calls"),
    ("[f(a=)]", "empty argument value"),
])
def test_failures_carry_a_countable_reason(text, reason):
    with pytest.raises(CallParseError) as err:
        parse_calls(text)
    assert err.value.reason == reason


def test_looks_like_call():
    assert looks_like_call('  [f(a=1)]\n')
    assert not looks_like_call("The function [f] needs an argument.")
