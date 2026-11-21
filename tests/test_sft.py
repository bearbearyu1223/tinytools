"""The SFT sets: truncation, the random-call control, the counts (hand-made rows)."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from tinytools import sft


def tool(name, props=None, required=None):
    return {"name": name, "parameters": {"type": "dict", "properties": props or {},
                                         "required": required or []}}


def call(*pairs):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": n, "arguments": a}} for n, a in pairs]}


def user(text="hi"):
    return {"role": "user", "content": text}


def say(text):
    return {"role": "assistant", "content": text}


def toolmsg(text="{}"):
    return {"role": "tool", "content": text}


def record(rid, tools, messages):
    return {"id": rid, "tools": [{"type": "function", "function": t} for t in tools],
            "messages": messages}


A = tool("A", {"q": {"type": "string"}}, ["q"])
B = tool("B", {"n": {"type": "int"}}, ["n"])
C = tool("C", {"unit": {"type": "string", "enum": ["C", "F"]}}, ["unit"])


# -- truncation -----------------------------------------------------------------


def test_truncate_after_first_call():
    msgs = [user(), call(("A", {"q": "x"})), toolmsg(), say("done"), user(), call(("B", {}))]
    assert sft.truncate(msgs) == msgs[:2]


def test_truncate_keeps_not_call_before_the_call():
    msgs = [user(), say("which city?"), user("Paris"), call(("A", {"q": "Paris"})),
            toolmsg(), say("sunny")]
    assert sft.truncate(msgs) == msgs[:4]


def test_truncate_keeps_system_and_call_only_row():
    msgs = [{"role": "system", "content": "s"}, user(), call(("A", {"q": "x"}))]
    assert sft.truncate(msgs) == msgs


def test_row_without_a_call_is_kept_whole():
    msgs = [user(), say("no function fits"), user(), say("still no")]
    assert sft.truncate(msgs) == msgs
    assert sft.first_call_index(msgs) is None


def test_truncate_does_not_mutate():
    msgs = [user(), call(("A", {"q": "x"})), toolmsg()]
    sft.truncate(msgs)
    assert len(msgs) == 3


def test_cut_counts():
    before = [user(), call(("A", {"q": "x"})), toolmsg(), say("done")]
    after = sft.truncate(before)
    c = sft.Cut()
    c.add(before, after, (100, 70), (30, 10))
    c.add(after, after, (70, 70), (10, 10))
    assert (c.rows, c.rows_cut, c.turns_removed, c.assistant_turns_removed) == (2, 1, 2, 1)
    assert (c.tokens_removed, c.trained_removed) == (30, 20)


def test_shape_counts():
    s = sft.Shape()
    s.add([user(), say("which?"), user(), call(("A", {}), ("B", {})), toolmsg(), say("ok")],
          tokens=50, trained=20)
    assert (s.rows, s.assistant_turns, s.call_turns, s.parallel_turns, s.calls) == (1, 3, 1, 1, 2)
    assert (s.not_calls_after_user, s.not_calls_after_tool, s.tool_results) == (1, 1, 1)
    assert (s.tokens, s.trained) == (50, 20)


# -- schemas --------------------------------------------------------------------


@pytest.mark.parametrize("value, declared, ok", [
    ("x", "string", True), (3, "int", True), (3.5, "int", False), (True, "int", False),
    (3, "float", True), (True, "boolean", True), ([1], "array", True), ({}, "dict", True),
    (None, ["integer", "null"], True), ("x", ["integer", "null"], False), ("x", "weird", True),
])
def test_type_ok(value, declared, ok):
    assert sft.type_ok(value, declared) is ok


def test_fits():
    assert sft.fits({"q": "x"}, A)
    assert not sft.fits({}, A)  # required missing
    assert not sft.fits({"q": "x", "z": 1}, A)  # unlisted key
    assert not sft.fits({"q": 3}, A)  # wrong type


# -- argument sources -------------------------------------------------------------


def test_another_row_that_fits_is_preferred_and_never_this_row():
    pool = sft.Pool.build([
        record("r1", [A], [user(), call(("A", {"q": "mine"}))]),
        record("r2", [A], [user(), call(("A", {"q": 7}))]),  # doesn't fit A's schema
        record("r3", [A], [user(), call(("A", {"q": "theirs"}))]),
    ])
    for i in range(50):
        ch = sft.choose_arguments(A, "r1", pool, random.Random(i), replaced={"q": "mine"})
        assert ch.source == sft.ANOTHER_ROW
        assert (ch.copied_from, ch.arguments) == ("r3", {"q": "theirs"})


def test_other_schema_when_nothing_fits():
    pool = sft.Pool.build([record("r2", [A], [user(), call(("A", {"q": 7}))])])
    ch = sft.choose_arguments(A, "r1", pool, random.Random(0))
    assert (ch.source, ch.copied_from, ch.arguments) == (sft.OTHER_SCHEMA, "r2", {"q": 7})


def test_this_row_only_for_the_replaced_calls_tool():
    pool = sft.Pool.build([record("r1", [A, B], [user(), call(("A", {"q": "x"}), ("B", {"n": 2}))])])
    right = sft.choose_arguments(A, "r1", pool, random.Random(0), replaced={"q": "x"})
    assert (right.source, right.arguments, right.copied_from) == (sft.THIS_ROW, {"q": "x"}, "r1")
    # B is called in this row too (a parallel sibling), but not by the call replaced:
    # its real arguments are not used.
    wrong = sft.choose_arguments(B, "r1", pool, random.Random(0), replaced=None)
    assert wrong.source == sft.BUILT and wrong.copied_from is None


def test_built_parameters_each_source():
    t = tool("T", {"unit": {"type": "string", "enum": ["C", "F"]},
                   "n": {"type": "int"},
                   "lang": {"type": "string", "default": "en"},
                   "blob": {"type": "array"},
                   "opt": {"type": "string"}},
             ["unit", "n", "lang", "blob"])
    pool = sft.Pool.build([
        record("r1", [B], [user(), call(("B", {"n": 99}))]),  # this row: not used
        record("r2", [B], [user(), call(("B", {"n": 5}))]),
        record("r3", [A], [user(), call(("A", {"q": "x"}))]),
    ])
    pool.values["lang"].append(("r3", 12))  # wrong type for lang: skipped
    ch = sft.build_arguments(t, "r1", pool, random.Random(0))
    assert ch.source == sft.BUILT
    assert ch.arguments["unit"] in ("C", "F")
    assert ch.arguments["n"] == 5
    assert ch.arguments["lang"] == "en"
    assert ch.arguments["blob"] == []
    assert "opt" not in ch.arguments  # optional parameters left out
    assert ch.param_sources == (sft.P_ENUM, sft.P_COPIED, sft.P_DEFAULT, sft.P_PLACEHOLDER)


# -- the random-call twin ---------------------------------------------------------


def two_tool_pool():
    return sft.Pool.build([
        record("d1", [A], [user(), call(("A", {"q": "donor"}))]),
        record("d2", [B], [user(), call(("B", {"n": 1}))]),
    ])


def test_two_tool_rows_draw_the_right_tool_about_half_the_time():
    pool = two_tool_pool()
    msgs = [user(), call(("A", {"q": "x"}))]
    hits = sum(sft.random_calls(f"row-{i}", [A, B], msgs, pool, seed=0)[1][0]["drawn"] == "A"
               for i in range(4000))
    # binomial(4000, 1/2): sd ~ 32; 5 sd either side
    assert 1840 < hits < 2160


def test_uniform_draw_includes_the_right_tool_among_k():
    pool = sft.Pool.build([])
    msgs = [user(), call(("C", {"unit": "F"}))]
    drawn = Counter(sft.random_calls(f"row-{i}", [A, B, C], msgs, pool, seed=3)[1][0]["drawn"]
                    for i in range(6000))
    assert set(drawn) == {"A", "B", "C"}
    for name in "ABC":
        assert 1850 < drawn[name] < 2150  # 2000 expected, sd ~ 37


def test_parallel_turn_replaced_call_by_call():
    pool = two_tool_pool()
    msgs = [user(), call(("A", {"q": "1"}), ("A", {"q": "2"}), ("B", {"n": 3}))]
    out, log = sft.random_calls("r", [A, B], msgs, pool, seed=0)
    assert len(out[1]["tool_calls"]) == 3
    assert [(e["turn"], e["call"]) for e in log] == [(1, 0), (1, 1), (1, 2)]
    assert [e["original"] for e in log] == ["A", "A", "B"]
    assert [tc["function"]["name"] for tc in out[1]["tool_calls"]] == [e["drawn"] for e in log]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"q": "1"}  # input untouched


def test_not_call_turns_and_others_untouched():
    pool = two_tool_pool()
    sysm = {"role": "system", "content": "date"}
    msgs = [sysm, user("u1"), say("which one?"), user("u2"), call(("A", {"q": "x"}))]
    out, log = sft.random_calls("r", [A, B], msgs, pool, seed=0)
    for i in range(4):
        assert out[i] is msgs[i]
    assert len(log) == 1 and out[4] is not msgs[4]


def test_row_without_call_has_no_replacements():
    msgs = [user(), say("no function fits")]
    out, log = sft.random_calls("r", [A, B], msgs, two_tool_pool(), seed=0)
    assert out == msgs and log == []


def test_deterministic_per_seed_and_row():
    pool = two_tool_pool()
    msgs = [user(), call(("A", {"q": "x"}), ("B", {"n": 2}))]
    a = sft.random_calls("row-7", [A, B, C], msgs, pool, seed=1)
    assert a == sft.random_calls("row-7", [A, B, C], msgs, pool, seed=1)
    seeds = {json.dumps(sft.random_calls("row-7", [A, B, C], msgs, pool, seed=s)[0])
             for s in range(20)}
    assert len(seeds) > 1


def test_identical_flag_and_control_stats():
    pool = sft.Pool.build([])
    msgs = [user(), call(("A", {"q": "x"}))]
    st = sft.ControlStats()
    for i in range(200):
        _, log = sft.random_calls(f"r{i}", [A, B], msgs, pool, seed=0)
        e = log[0]
        # A is never called elsewhere: a hit keeps the real arguments (this row)
        assert e["identical"] == (e["drawn"] == "A")
        assert e["source"] == (sft.THIS_ROW if e["drawn"] == "A" else sft.BUILT)
        st.add(2, log)
    assert st.calls == 200 and st.expected == pytest.approx(100.0)
    assert st.identical == st.hits == st.source_hit[sft.THIS_ROW]
    assert st.rows_by_k[2] == 200


# -- tokens ---------------------------------------------------------------------


def test_turn_tokens_by_block():
    text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n" \
           "<|im_start|>assistant\nAB<|im_end|>\n"
    a0 = text.index("<|im_start|>assistant")
    body = a0 + len("<|im_start|>assistant\n")
    end = text.index("<|im_end|>", body) + len("<|im_end|>")
    # one token per block header, one per character of content, one per <|im_end|>
    offsets = []
    for start in (0, text.index("<|im_start|>user"), a0):
        nl = text.index("\n", start)
        offsets.append((start, nl + 1))
        k = nl + 1
        close = text.index("<|im_end|>", k)
        offsets += [(j, j + 1) for j in range(k, close)]
        offsets.append((close, close + len("<|im_end|>")))
    turns = sft.turn_tokens(text, offsets, [(body, end)])
    assert turns == [("system", 3, 0), ("user", 3, 0), ("assistant", 4, 3)]


def test_count_tokens_rejects_straddling():
    with pytest.raises(ValueError):
        sft.count_tokens([(0, 5)], [(3, 10)])
    assert sft.count_tokens([(0, 3), (3, 5)], [(3, 5)]) == (2, 1)


# -- the files, pinned ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA256 = {
    "real": "8238570af984a0c0ab00343eb3e9d540a5691a63cf99c1e59da32af0db78122a",
    "control": "808f1aea46b23d5bdc6657c3bcedb9f29acde141be63a4bad93adb5a5bd367af",
}


def test_sets_regenerate_byte_for_byte():
    src = ROOT / sft.CONVERTED
    if not src.exists():
        pytest.skip("run `tinytools toolace-convert` first")
    raw = src.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sft.CONVERTED_SHA256
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001 - no cached tokenizer: nothing to pin against
        pytest.skip("Qwen2.5 tokenizer not cached")
    converted = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    s = sft.build_sets(converted, sft.DEFAULT_SEED, tok)
    assert next(r for r in s.real if r["id"] == sft.EXAMPLE_ID)["n_tools"] == 2
    for name, records in (("real", s.real), ("control", s.control)):
        assert hashlib.sha256(sft.jsonl_bytes(records)).hexdigest() == PINNED_SHA256[name], name
