"""The tag-free reading (plan 3f, T9e review S1): the extractor, scoring through the
strict checkers, and sft-report's strict output unchanged."""

import dataclasses
import json
from pathlib import Path

import pytest

from tinytools import tagfree as TF
from tinytools.checkers import check_reply, decode
from tinytools.classify import (CALLED_ANYWAY, DECLINED, NO_CALL, RIGHT, UNPARSEABLE,
                                WRONG_COUNT)

ROOT = Path(__file__).resolve().parents[1]
M0_BASE = ROOT / "runs" / "2026-11-22-milestone0" / "outputs" / "milestone0-Qwen2.5-1.5B-ours-replies.jsonl"

CALC = {"name": "calculator", "arguments": {"expression": "8326 / 574.9"}}
CALC_JSON = json.dumps(CALC)

# Real shapes, cut from Milestone 0's saved base replies (Qwen2.5-1.5B, greedy, dev).
DEV_0000 = (CALC_JSON + "\nçãeste\n" + CALC_JSON + "\nçãeste\n"
            + '{"name": "calculator", "arguments": {"expression":')  # cut by the cap
STOCK = '{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}'
DEV_0004 = "\n".join([STOCK] * 3) + '\n{"name": "get_stock_price",'  # repeats to the cap
DEV_0022 = '{"name": "get_calendar_event", "arguments": {"date": "2026-03-13", "title": "Meeting with John"}}'
DEV_0026 = ('{"type": "function", "function": {"name": "get_calendar_event", "description": '
            '"Look up events"}}')  # the tool list echoed: not a call
DEV_0006 = 'Set up "Design critique" on April 5th, starting at 17:30, for 75 minutes.acey\nacey'


# -- the extractor -----------------------------------------------------------------------


def test_one_call_then_garbage_is_one_call():
    assert TF.bare_calls(DEV_0000) == [CALC]
    assert TF.tagfree_reply(DEV_0000) == f"<tool_call>\n{CALC_JSON}\n</tool_call>"


def test_repeated_calls_are_all_taken():
    assert TF.bare_calls(DEV_0004) == [json.loads(STOCK)] * 3


def test_a_whole_reply_that_is_one_call():
    assert TF.bare_calls(DEV_0022) == [json.loads(DEV_0022)]


def test_an_echoed_schema_and_prose_are_no_call():
    for raw in (DEV_0026, DEV_0006, "", "   ", "{", "{not json}"):
        assert TF.bare_calls(raw) == []
        assert TF.tagfree_reply(raw) == raw


def test_json_in_prose_is_not_a_call():
    raw = f"Sure, here is the call: {CALC_JSON}"
    assert TF.bare_calls(raw) == [] and TF.tagfree_reply(raw) == raw


def test_two_objects_with_whitespace_or_nothing_between():
    b = {"name": "b", "arguments": {}}
    assert TF.bare_calls(f"  {CALC_JSON} \n\t {json.dumps(b)}  tail") == [CALC, b]
    assert TF.bare_calls(f"{CALC_JSON}{json.dumps(b)}") == [CALC, b]


def test_stops_at_the_first_thing_that_is_not_a_call():
    b = {"name": "b", "arguments": {}}
    raw = f"{CALC_JSON} x {json.dumps(b)}"
    assert TF.bare_calls(raw) == [CALC]
    raw = f'{CALC_JSON}\n{{"name": "b"}}\n{json.dumps(b)}'  # a non-call object ends the run
    assert TF.bare_calls(raw) == [CALC]


def test_nested_braces_and_quotes_inside_strings():
    call = {"name": "search_documents", "arguments": {"query": 'a } b { "c" \\n}'}}
    raw = json.dumps(call) + "}}}garbage{"
    assert TF.bare_calls(raw) == [call]
    assert decode(TF.tagfree_reply(raw)) == [{"search_documents": call["arguments"]}]


def test_keys_must_be_exactly_name_and_arguments():
    for obj in ({"name": "a"}, {"arguments": {}}, {"name": "a", "arguments": {}, "id": 1},
                {"name": 5, "arguments": {}}, {"name": "a", "arguments": [1]},
                {"name": "a", "arguments": '{"x": 1}'}):
        assert TF.bare_calls(json.dumps(obj)) == []


def test_string_arguments_are_rejected_as_strict_rejects_them():
    # the premise of the rule: strict's decoder does not take arguments as a string
    raw = '<tool_call>\n{"name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}\n</tool_call>'
    assert decode(raw) is None
    assert check_reply(raw, {"name": "calculator", "arguments": {"expression": "1+1"}}).outcome \
        == UNPARSEABLE


def test_a_tagged_reply_is_read_as_strict():
    tagged = f"{CALC_JSON}\n<tool_call>\n{CALC_JSON}\n</tool_call>"
    assert TF.tagfree_reply(tagged) == tagged
    broken = "<tool_call>\nnot json\n</tool_call>"
    assert TF.tagfree_reply(broken) == broken


def test_a_tag_inside_a_string_does_not_add_a_tag():
    call = {"name": "search_documents", "arguments": {"query": "<tool_call> x"}}
    raw = json.dumps(call).replace("<", "\\u003c")  # no literal tag in the reply
    out = TF.tagfree_reply(raw)
    assert out.count("<tool_call>") == 1
    assert decode(out) == [{"search_documents": call["arguments"]}]


def test_read_dispatches_and_rejects_unknown_readings():
    assert TF.read(DEV_0000, TF.STRICT) == DEV_0000
    assert TF.read(DEV_0000, TF.TAGFREE) == TF.tagfree_reply(DEV_0000)
    with pytest.raises(ValueError):
        TF.read("x", "loose")


def test_rule_lines_fit_the_report():
    assert all(len(line) <= 70 for line in TF.RULE)


# -- scoring through the strict checkers --------------------------------------------------


def test_ours_outcomes_on_both_readings():
    assert check_reply(DEV_0000, CALC).outcome == NO_CALL
    assert check_reply(TF.tagfree_reply(DEV_0000), CALC).outcome == RIGHT
    stock = json.loads(STOCK)
    assert check_reply(TF.tagfree_reply(DEV_0004), stock).outcome == WRONG_COUNT
    assert check_reply(TF.tagfree_reply(DEV_0006), CALC).outcome == NO_CALL


def test_no_tool_fits_a_bare_call_is_called_anyway():
    assert check_reply(DEV_0000, None).outcome == DECLINED
    assert check_reply(TF.tagfree_reply(DEV_0000), None).outcome == CALLED_ANYWAY
    assert check_reply(TF.tagfree_reply(DEV_0006), None).outcome == DECLINED


def test_bfcl_judge_on_the_tag_free_reading():
    from tinytools import bfcl

    h = bfcl.handler()
    item = bfcl.load_items("simple", 1, "eval")[0]
    tagged = bfcl.oracle_reply(item)
    bare = "\n".join(line for line in tagged.splitlines() if "tool_call>" not in line)
    assert bfcl.judge(h, item, tagged).outcome == RIGHT
    assert bfcl.judge(h, item, bare).outcome == NO_CALL
    assert bfcl.judge(h, item, TF.tagfree_reply(bare + "\nqwerty")).outcome == RIGHT
    irr = bfcl.load_items("irrelevance", 1, "eval")[0]
    assert bfcl.judge(h, irr, bare).outcome == DECLINED
    assert bfcl.judge(h, irr, TF.tagfree_reply(bare)).outcome == CALLED_ANYWAY


@pytest.mark.skipif(not M0_BASE.exists(), reason="Milestone 0's run record is not here")
def test_milestone0_base_replies():
    """Every saved base reply: tagless, and the reading never turns a non-call into
    anything but a call that starts the reply. 32 of 160 dev call tasks' first line is
    call JSON (T9e S1); the extractor finds a call on at least those."""
    rows = [json.loads(x) for x in M0_BASE.read_text(encoding="utf-8").split("\n") if x]
    greedy = [r for r in rows if r["pass"] == "greedy"]
    assert all("<tool_call>" not in r["reply"] for r in rows)
    found = [r for r in greedy if r["kind"] != "no tool fits" and TF.bare_calls(r["reply"])]
    first_line = [r for r in greedy if r["kind"] != "no tool fits"
                  and TF.bare_calls(r["reply"].split("\n")[0])
                  and TF.bare_calls(r["reply"].split("\n")[0])[0]["name"] == r["tool"]]
    assert len(first_line) == 32
    assert {r["id"] for r in first_line} <= {r["id"] for r in found}
    for r in found:
        assert r["reply"].lstrip().startswith("{")


# -- sft-report: strict unchanged, tag-free end to end -------------------------------------


def _report(monkeypatch, tmp_path, src: Path, quick: bool, profile: str, reading: str) -> str:
    import tinytools.report as R
    import tinytools.sftrun as S

    dest = tmp_path / "out.txt"

    class Redirect(R.Report):
        def __init__(self, title, out=None):
            super().__init__(title, out=dest)

    monkeypatch.setattr(R, "Report", Redirect)
    orig = S.mode
    monkeypatch.setattr(S, "mode", lambda q, root, settings=None, profile="main":
                        dataclasses.replace(orig(q, root, settings, profile), out=src))
    assert S.run_report(ROOT, quick, profile, reading) == 0
    return dest.read_text()


def _body(text: str) -> list[str]:
    lines = text.splitlines()
    return lines[:3] + lines[7:]  # the four hardware lines differ by machine


@pytest.mark.parametrize("src,profile,capture", [
    ("outputs/main-quick", "main", "outputs/sft-quick.txt"),
    ("outputs/base-quick", "base", "outputs/sft-base-quick.txt"),
])
def test_strict_report_matches_its_capture(monkeypatch, tmp_path, src, profile, capture):
    if not (ROOT / capture).exists() or not (ROOT / src / "untrained").exists():
        pytest.skip(f"{capture} not here")
    got = _report(monkeypatch, tmp_path, ROOT / src, True, profile, "strict")
    assert _body(got) == _body((ROOT / capture).read_text())


def test_tagfree_report_quick_base(monkeypatch, tmp_path):
    src = ROOT / "outputs" / "base-quick"
    if not (src / "untrained" / "score.json").exists():
        pytest.skip("outputs/base-quick not here")
    text = _report(monkeypatch, tmp_path, src, True, "base", "tagfree")
    assert all(len(x) <= 74 for x in text.splitlines())
    for want in ("Strict vs tag-free, right, per arm", "(r5a)", "(r5b)", "(r5c)", "(r5d)",
                 "base-arm rule (plan 3f)", "(r6a)", *TF.RULE):
        assert want in text
    assert "(r4c)" not in text


def test_base_rule_is_r5a_paired_on_call_tasks():
    from fractions import Fraction

    from tinytools.sftrun import Scores, base_rule

    ids = ["a", "b", "c", "d"]
    un = Scores("u", dict(zip(ids, [RIGHT, NO_CALL, NO_CALL, NO_CALL], strict=True)), {}, {}, {}, {})
    s1 = Scores("s1", dict(zip(ids, [RIGHT, RIGHT, NO_CALL, RIGHT], strict=True)), {}, {}, {}, {})
    s2 = Scores("s2", dict(zip(ids, [RIGHT, RIGHT, RIGHT, NO_CALL], strict=True)), {}, {}, {}, {})
    p = base_rule(un, [s1, s2], ids)
    assert p.share_b == Fraction(1, 4) and p.share_a == Fraction(6, 8)


def test_reply_files_with_unicode_line_breaks_load(tmp_path):
    """Replies are written with ensure_ascii=False, so U+0085 / U+2028 stay raw in the
    file (9 of Milestone 0's 1,800 base replies have U+0085); str.splitlines() would
    split a row there. The readers split on newlines only."""
    from tinytools.sftrun import _rows

    rows = [{"reply": "a\x85b c d"}, {"reply": "e"}]
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    assert len(path.read_text(encoding="utf-8").splitlines()) > 2  # the trap
    assert _rows(path) == rows
