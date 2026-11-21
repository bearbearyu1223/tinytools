"""The harness against bfcl-eval's real data and checker, with hand-written replies (no model)."""

import json

import pytest

from tinytools import bfcl
from tinytools.classify import (
    CALLED_ANYWAY, DECLINED, NO_CALL, RIGHT, UNPARSEABLE, WRONG_ARGUMENTS, WRONG_COUNT,
    WRONG_FUNCTION, is_right,
)


@pytest.fixture(scope="module")
def h():
    return bfcl.handler()


@pytest.mark.parametrize("category", list(bfcl.CATEGORIES))
def test_reference_reply_is_right(h, category):
    for item in bfcl.load_items(category, None):
        v = bfcl.judge(h, item, bfcl.oracle_reply(item))
        want = DECLINED if item.irrelevance else RIGHT
        assert v.outcome == want and is_right(v.outcome) and v.bfcl_valid, item.id


def test_prompt_docs_carry_the_hint_and_checker_docs_do_not():
    item = bfcl.load_items("simple", 1)[0]
    assert item.prompt_functions[0]["description"].endswith("Python 3 syntax.")
    assert not item.checker_entry["function"][0]["description"].endswith("Python 3 syntax.")


def test_each_failure_kind_on_a_real_item(h):
    item = bfcl.load_items("simple", 1)[0]  # calculate_triangle_area(base=10, height=5)
    (name, params), = item.ground_truth[0].items()

    def call(n, args):
        return '<tool_call>\n{"name": %s, "arguments": %s}\n</tool_call>' % (
            json.dumps(n), json.dumps(args))

    good = {k: v[0] for k, v in params.items() if v[0] != ""}
    bad = dict(good, base=999)
    assert bfcl.judge(h, item, "The area is 25.").outcome == NO_CALL
    assert bfcl.judge(h, item, "<tool_call>\n{not json}\n</tool_call>").outcome == UNPARSEABLE
    assert bfcl.judge(h, item, call("triangle_area", good)).outcome == WRONG_FUNCTION
    assert bfcl.judge(h, item, call(name, bad)).outcome == WRONG_ARGUMENTS
    assert bfcl.judge(h, item, call(name, good)).outcome == RIGHT


def test_irrelevance_split_vs_bfcl(h):
    item = bfcl.load_items("irrelevance", 1)[0]
    broken = bfcl.judge(h, item, "<tool_call>\n{oops\n</tool_call>")
    assert broken.outcome == CALLED_ANYWAY and broken.bfcl_valid  # BFCL counts it right
    fname = item.prompt_functions[0]["name"]
    called = bfcl.judge(h, item, '<tool_call>\n{"name": "%s", "arguments": {}}\n</tool_call>' % fname)
    assert called.outcome == CALLED_ANYWAY and not called.bfcl_valid


def test_a_missing_parallel_call_is_wrong_count(h):
    # The first parallel item whose reference calls one function 2+ times.
    item = next(i for i in bfcl.load_items("parallel", None)
                if len(set(i.expected_names())) == 1 and len(i.expected_names()) > 1)
    blocks = bfcl.oracle_reply(item).split("\n<tool_call>")
    assert bfcl.judge(h, item, blocks[0]).outcome == WRONG_COUNT
    assert bfcl.judge(h, item, bfcl.oracle_reply(item)).outcome == RIGHT


def test_item_seed_depends_only_on_seed_and_id():
    assert bfcl.item_seed(0, "simple_python_3") == bfcl.item_seed(0, "simple_python_3")
    assert bfcl.item_seed(0, "simple_python_3") != bfcl.item_seed(1, "simple_python_3")


def test_reference_reply_unwraps_nested_dict_values(h):
    # multiple_8's budget is {"min": [300000], "max": [400000]}: allowed values all the way down.
    item = next(i for i in bfcl.load_items("multiple", 20) if i.id == "multiple_8")
    assert '"budget": {"min": 300000, "max": 400000}' in bfcl.oracle_reply(item)
    assert bfcl.judge(h, item, bfcl.oracle_reply(item)).outcome == RIGHT
