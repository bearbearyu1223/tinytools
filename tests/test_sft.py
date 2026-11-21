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


# -- which rows train ---------------------------------------------------------------


def test_row_type():
    assert sft.row_type(2, [user(), call(("A", {}))]) == sft.R_CALL
    assert sft.row_type(1, [user(), call(("A", {}))]) == sft.R_OUT
    assert sft.row_type(3, [user(), say("no")]) == sft.R_NO_CALL_2
    assert sft.row_type(1, [user(), say("no")]) == sft.R_NO_CALL_1
    assert sft.row_type(0, [user(), say("hi")]) == sft.R_NO_CALL_0
    assert set(sft.ROW_TYPES) == {sft.R_CALL, sft.R_NO_CALL_2, sft.R_NO_CALL_1,
                                  sft.R_NO_CALL_0, sft.R_OUT}


@pytest.mark.parametrize("value, empty", [
    ("", True), ("  ", True), ([], True), ({}, True), (None, True),
    ("x", False), (0, False), (False, False), ([0], False), ({"a": 1}, False),
])
def test_is_empty(value, empty):
    assert sft.is_empty(value) is empty


# -- argument sources -------------------------------------------------------------


def test_another_row_that_fits_is_preferred_and_never_this_row():
    pool = sft.Pool.build([
        record("r1", [A], [user(), call(("A", {"q": "mine"}))]),
        record("r2", [A], [user(), call(("A", {"q": 7}))]),  # doesn't fit A's schema
        record("r3", [A], [user(), call(("A", {"q": "theirs"}))]),
    ])
    for i in range(50):
        ch = sft.choose_arguments(A, "r1", pool, random.Random(i))
        assert ch.source == sft.ANOTHER_ROW
        assert (ch.copied_from, ch.arguments) == ("r3", {"q": "theirs"})


def test_other_schema_when_nothing_fits():
    pool = sft.Pool.build([record("r2", [A], [user(), call(("A", {"q": 7}))])])
    ch = sft.choose_arguments(A, "r1", pool, random.Random(0))
    assert (ch.source, ch.copied_from, ch.arguments) == (sft.OTHER_SCHEMA, "r2", {"q": 7})


def test_right_tool_called_only_in_this_row_is_built_not_copied():
    pool = sft.Pool.build([
        record("r1", [A, B], [user(), call(("A", {"q": "x"}), ("B", {"n": 2}))]),
        record("r2", [C], [user(), call(("C", {"unit": "F"}), ("Z", {"s": "other"}))]),
    ])
    # A is the replaced call's own tool, and only this row calls it: built, from r2's string.
    right = sft.choose_arguments(A, "r1", pool, random.Random(0))
    assert (right.source, right.copied_from) == (sft.BUILT, None)
    assert right.param_sources == (sft.P_TYPED,) and right.param_donors == ("r2",)
    assert right.arguments["q"] in ("F", "other")
    # B is called in this row too (a parallel sibling): its real arguments are not used,
    # and no other row has an integer, so there is nothing to build it from.
    with pytest.raises(ValueError, match="no value of type"):
        sft.choose_arguments(B, "r1", pool, random.Random(0))


def test_built_parameters_each_source_and_never_empty():
    t = tool("T", {"unit": {"type": "string", "enum": ["", "C", "F"]},
                   "n": {"type": "int"},
                   "lang": {"type": "string", "default": "en"},
                   "blob": {"type": "array"},
                   "note": {"type": "string", "default": ""},
                   "opt": {"type": "string"}},
             ["unit", "n", "lang", "blob", "note"])
    pool = sft.Pool.build([
        record("r1", [B], [user(), call(("B", {"n": 99}), ("Y", {"blob": [7]}))]),  # this row
        record("r2", [B], [user(), call(("B", {"n": 5}), ("Y", {"note": "", "blob": []}))]),
        record("r3", [A], [user(), call(("A", {"q": "x"}), ("Y", {"list": ["a"]}))]),
    ])
    pool.values["lang"].append(("r3", 12))  # wrong type for lang: skipped
    for i in range(30):
        ch = sft.build_arguments(t, "r1", pool, random.Random(i))
        assert ch.source == sft.BUILT
        assert ch.arguments["unit"] in ("C", "F")  # the empty enum value is never drawn
        assert ch.arguments["n"] == 5
        assert ch.arguments["lang"] == "en"
        assert ch.arguments["blob"] == ["a"]  # r2's [] is empty, r1's is this row's
        assert ch.arguments["note"] == "x"  # the only non-empty string in another row
        assert "opt" not in ch.arguments  # optional parameters left out
        assert ch.param_sources == (sft.P_ENUM, sft.P_COPIED, sft.P_DEFAULT, sft.P_TYPED,
                                    sft.P_TYPED)
        assert ch.param_donors == (None, "r2", None, "r3", "r3")
        assert not any(sft.is_empty(v) for v in ch.arguments.values())


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
    pool = two_tool_pool()
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


def test_no_control_argument_comes_from_its_own_row():
    # Every row calls A and B with its own values; the control must never reuse them.
    rows = [record(f"r{i}", [A, B], [user(), call(("A", {"q": f"q{i}"}), ("B", {"n": i}))])
            for i in range(12)]
    pool = sft.Pool.build(rows)
    st = sft.ControlStats()
    for r in rows:
        own = {json.dumps(tc["function"]["arguments"], sort_keys=True)
               for tc in r["messages"][1]["tool_calls"]}
        for seed in range(5):
            out, log = sft.random_calls(r["id"], [A, B], r["messages"], pool, seed)
            st.add(r["id"], 2, log)
            for e, tc in zip(log, out[1]["tool_calls"], strict=True):
                assert e["copied_from"] != r["id"] and r["id"] not in e["param_donors"]
                assert json.dumps(tc["function"]["arguments"], sort_keys=True) not in own
    assert st.from_own_row == 0 and st.identical == 0


def test_identical_only_by_coincidence_and_control_stats():
    # d1 asks the same as the rows below, so a draw of A copies {"q": "x"} by coincidence.
    pool = sft.Pool.build([record("d1", [A], [user(), call(("A", {"q": "x"}))]),
                           record("d2", [B], [user(), call(("B", {"n": 1}))])])
    msgs = [user(), call(("A", {"q": "x"}))]
    st = sft.ControlStats()
    for i in range(200):
        out, log = sft.random_calls(f"r{i}", [A, B], msgs, pool, seed=0)
        e = log[0]
        assert e["source"] == sft.ANOTHER_ROW
        assert e["identical"] == (e["drawn"] == "A")
        st.add(f"r{i}", 2, log, [dict(tc["function"], source=e["source"])
                                 for tc in out[1]["tool_calls"]])
    assert st.calls == 200 and st.expected == pytest.approx(100.0)
    assert st.identical == st.hits == st.identical_by_source[sft.ANOTHER_ROW]
    assert st.rows_by_k[2] == 200 and st.from_own_row == 0 and st.empty_built == 0
    assert st.identical_no_args == 0


def test_a_tool_with_no_arguments_reproduces_the_real_call():
    N = tool("N")  # no parameters: one possible call
    pool = two_tool_pool()
    st = sft.ControlStats()
    for i in range(100):
        _, log = sft.random_calls(f"r{i}", [N, B], [user(), call(("N", {}))], pool, seed=0)
        st.add(f"r{i}", 2, log)
    assert st.identical == st.hits == st.identical_no_args == st.source_hit[sft.BUILT] > 0


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


# -- both arms, with the real tokenizer ------------------------------------------------


def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001 - no cached tokenizer
        pytest.skip("Qwen2.5 tokenizer not cached")


def _converted(tok, rid, tools, messages):
    r = sft.render(rid, tools, messages, tok)
    return {k: r[k] for k in ("id", "tools", "messages", "text", "train_spans")}


def test_no_call_rows_go_into_both_arms_unchanged():
    tok = _tokenizer()
    rows = [
        _converted(tok, "call2", [A, B], [user("find x"), call(("A", {"q": "x"})), toolmsg(),
                                          say("done")]),
        _converted(tok, "call1", [B], [user("count"), call(("B", {"n": 9}))]),  # left out
        _converted(tok, "decline2", [A, B], [user("weather?"), say("No function fits.")]),
        _converted(tok, "decline1", [B], [user("weather?"), say("I can't do that.")]),
        _converted(tok, "chat0", [], [user("hi"), say("Hello!")]),
        _converted(tok, "donor", [A, B], [user("z"), call(("A", {"q": "z"}), ("B", {"n": 4}))]),
    ]
    s = sft.build_sets(rows, (1, 2), tok)
    assert [r["id"] for r in s.real] == ["call2", "decline2", "decline1", "chat0", "donor"]
    assert s.rows_converted[sft.R_OUT] == 1 and s.rows_trained[sft.R_OUT] == 0
    for seed in (1, 2):
        twins = {r["id"]: r for r in s.controls[seed].records}
        assert list(twins) == [r["id"] for r in s.real]
        for real in s.real:
            twin = dict(twins[real["id"]])
            log = twin.pop("random_calls")
            if real["id"] in ("decline2", "decline1", "chat0"):
                assert twin == real and log == []
            else:
                assert len(log) == sum(len(m.get("tool_calls") or []) for m in real["messages"])
        assert sft.same_structure(s.controls[seed].shape, s.shape_real)
    assert s.real[0]["messages"][-1]["role"] == "assistant"  # truncated after the call


def test_drop_reason():
    tools = [A, B]
    assert sft.drop_reason(tools, [user("x"), call(("A", {"q": "x"}))]) is None
    assert sft.drop_reason(tools, [user("x"), say(""), call(("A", {"q": "x"}))]) == sft.D_EMPTY
    assert sft.drop_reason(tools, [user("x"), say("  \n")]) == sft.D_EMPTY
    bracket = say('I will call two.\n\n[A(q="x"), A(q="y")]')
    assert sft.drop_reason(tools, [user("x"), bracket]) == sft.D_BRACKET
    # brackets around anything but a listed tool's call are prose
    assert sft.drop_reason(tools, [user("x"), say("See [1] and [C(q=1)].")]) is None
    # an empty turn wins over a bracket call
    assert sft.drop_reason(tools, [user("x"), bracket, say("")]) == sft.D_EMPTY


def test_dropped_rows_leave_both_arms():
    tok = _tokenizer()
    rows = [
        _converted(tok, "keep", [A, B], [user("find x"), call(("A", {"q": "x"}), ("B", {"n": 2}))]),
        _converted(tok, "empty", [A, B], [user("find y"), say(""), call(("A", {"q": "y"}))]),
        _converted(tok, "prose", [A, B], [user("weather?"), say('Try [B(n=2)] later.')]),
        _converted(tok, "donor", [A, B], [user("z"), call(("A", {"q": "z"}), ("B", {"n": 4}))]),
    ]
    s = sft.build_sets(rows, (1,), tok)
    assert [r["id"] for r in s.real] == ["keep", "donor"]
    assert [r["id"] for r in s.controls[1].records] == ["keep", "donor"]
    assert s.dropped == {sft.D_EMPTY: ["empty"], sft.D_BRACKET: ["prose"]}
    assert s.rows_dropped == {sft.R_CALL: 1, sft.R_NO_CALL_2: 1}
    assert s.rows_trained[sft.R_CALL] == 2 and s.rows_converted[sft.R_CALL] == 3


# -- the files, pinned ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
# Re-pinned after the review round 1 drops (fd99c4190): 14 rows out of both arms.
PINNED_SHA256 = {
    "real": "2cf7f25a4545d4d01dfa3c2f5d678c5d9194e20724ca417d2464b31cbdf1cdb8",
    1: "1a8b98c21361edc3dcc3d0e37e238b74c97e43d2dfc37c7a4f58ea83f1ec5f7f",
    2: "08d89acedda694f5ead4b943f42c866c245a5423e1e9964abc3ea406411a4fcb",
    3: "f5968d457419f1a89afa7582dc54acb1cb204b468c05fc2f878d09db4f2dea68",
}
DROPPED = {
    sft.D_EMPTY: ["toolace-27", "toolace-30", "toolace-162", "toolace-163", "toolace-263"],
    sft.D_BRACKET: ["toolace-1383", "toolace-1929", "toolace-4716", "toolace-5826",
                    "toolace-6774", "toolace-7224", "toolace-7605", "toolace-9833",
                    "toolace-10539"],
}


def test_sets_regenerate_byte_for_byte():
    src = ROOT / sft.CONVERTED
    if not src.exists():
        pytest.skip("run `tinytools toolace-convert` first")
    raw = src.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sft.CONVERTED_SHA256
    tok = _tokenizer()
    converted = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    s = sft.build_sets(converted, sft.DEFAULT_SEEDS, tok)
    assert hashlib.sha256(sft.jsonl_bytes(s.real)).hexdigest() == PINNED_SHA256["real"]
    assert s.dropped == DROPPED
    assert not {r["id"] for r in s.real} & {i for ids in DROPPED.values() for i in ids}
    for seed in sft.DEFAULT_SEEDS:
        cs = s.controls[seed]
        assert hashlib.sha256(sft.jsonl_bytes(cs.records)).hexdigest() == PINNED_SHA256[seed]
        # no real arguments, no placeholders, on the real data
        assert cs.stats.from_own_row == 0 and cs.stats.empty_built == 0
        for r in cs.records:
            for e in r["random_calls"]:
                assert e["copied_from"] != r["id"] and r["id"] not in e["param_donors"]
    # the r4b example: 2 tools, and the first seed draws the other tool, copied from another row
    assert next(r for r in s.real if r["id"] == sft.EXAMPLE_ID)["n_tools"] == 2
    e = next(r for r in s.controls[sft.DEFAULT_SEEDS[0]].records
             if r["id"] == sft.EXAMPLE_ID)["random_calls"][0]
    assert e["source"] == sft.ANOTHER_ROW and e["drawn"] != e["original"]
