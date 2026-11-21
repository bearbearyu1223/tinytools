"""The model-free pieces of scoring our tasks: limits, token spans, wrapping, group summaries."""

import json

from tinytools import ours, tasks
from tinytools.checkers import check_reply
from tinytools.classify import NO_CALL, RIGHT, WRONG_FUNCTION


def test_limit_per_kind_takes_the_first_n_of_each_kind_by_id():
    items = [{"id": f"test-{i:04d}", "kind": k} for i, k in
             enumerate([tasks.NO_FIT, tasks.CHOOSE, tasks.CHOOSE, tasks.TRICKY, tasks.CHOOSE])]
    got = ours.limit_per_kind(list(reversed(items)), 2)
    assert [t["id"] for t in got] == ["test-0000", "test-0001", "test-0002", "test-0003"]
    assert len(ours.limit_per_kind(items, None)) == 5


def test_tokens_in_span():
    offsets = [(0, 3), (3, 7), (7, 10), (10, 12)]
    assert ours.tokens_in_span(offsets, 3, 10) == (2, 0)
    assert ours.tokens_in_span(offsets, 4, 10) == (1, 1)


def test_wrap_lines_fits_and_is_lossless():
    text = '{"name": "create_calendar_event", "arguments": {"title": "Budget sync"}}\n\nshort'
    lines = ours.wrap_lines(text, 30)
    assert all(len(line) <= 30 for line in lines)
    joined, buf = [], ""
    for line in lines:
        if line.endswith("\\"):
            buf += line[:-1]
        else:
            joined.append(buf + line)
            buf = ""
    assert "\n".join(joined) == text
    assert ours.wrap_lines("x" * 25, 10) == ["x" * 9 + "\\", "x" * 9 + "\\", "x" * 7]


def test_group_summaries():
    assert ours.histogram([0, 8, 3, 3, 8], 8) == [1, 0, 0, 2, 0, 0, 0, 0, 2]
    assert ours.pass_at_k([0, 8, 3, 0]) == 2
    # post 8's unit: (1000 / G) x mixed share; 3 mixed of 10 groups of 8 -> 37.5
    assert ours.per_thousand(3, 10, 8) == 37.5
    assert ours.pct(57, 120) == "57 of 120 (47.5%)"


def _scored(task, replies):
    return ours.Scored(task, replies, [check_reply(r, task["reference"]) for r in replies],
                       [False] * len(replies), 0, 0.0)


def test_outcome_rows_and_disagreements():
    ref = {"name": "calculator", "arguments": {"expression": "348 * 27"}}
    calc = {"id": "t1", "kind": tasks.CHOOSE, "tool": "calculator", "reference": ref}
    nofit = {"id": "t2", "kind": tasks.NO_FIT, "tool": "get_weather", "reference": None}

    def call(expr):
        return "<tool_call>\n%s\n</tool_call>" % json.dumps(
            {"name": "calculator", "arguments": {"expression": expr}})

    results = [_scored(calc, [call("348 * 27"), call("27 * 348"), "9396"]),
               _scored(nofit, ["No weather tool here.", call("1 + 1")])]
    rows = {r.kind: r for r in ours.outcome_rows(results)}
    assert rows[tasks.CHOOSE].n == 3 and rows[tasks.CHOOSE].outcomes[RIGHT] == 2
    assert rows[tasks.CHOOSE].outcomes[NO_CALL] == 1 and rows[tasks.CHOOSE].exact == 1
    assert rows[tasks.NO_FIT].outcomes[RIGHT] == 1
    assert rows[tasks.NO_FIT].outcomes[WRONG_FUNCTION] == 1
    assert rows["total"].n == 5 and rows["total"].execution == 3
    exec_only, exact_only = ours.disagreements(results)
    assert [(s.task["id"], i) for s, i in exec_only] == [("t1", 1)]
    assert exact_only == []


def test_r1_task_is_a_tricky_create():
    t = ours.r1_task(tasks.generate(tasks.DEFAULT_SEED)["test"])
    assert t["kind"] == tasks.TRICKY and t["tool"] == "create_calendar_event"


def test_prompt_census_counts_the_tools_block():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    task = ours.r1_task(tasks.generate(tasks.DEFAULT_SEED)["test"])
    text = ours.prompt(tok, task)
    c = ours.prompt_census(tok, text)
    a, b = text.index("\n<tools>\n") + 1, text.index("\n</tools>\n") + len("\n</tools>")
    alone = len(tok(text[a:b], add_special_tokens=False)["input_ids"])
    assert abs(c.tool_list_tokens - alone) <= 2  # the same block, tokenized on its own
    assert c.tool_list_tokens < c.tools_section_tokens < c.tokens
