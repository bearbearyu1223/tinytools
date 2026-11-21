"""ToolACE conversion and masking, on hand-made rows (no download, no tokenizer)."""

import pytest

from tinytools import toolace

SYSTEM = (
    "You are an expert in composing functions.\n"
    "Here is a list of functions in JSON format that you can invoke:\n"
    '[{"name": "Get Time (UTC)", "parameters": {"type": "dict", "properties": '
    '{"zone": {"type": "string"}}}}, {"name": "ping", "parameters": {}}]. \n'
    "Should you decide to return the function call(s). \nPut it in the format of [func1(...)]"
)


def row(*turns, system=SYSTEM):
    return {"system": system, "conversations": [{"from": f, "value": v} for f, v in turns]}


def test_convert_call_tool_and_answer():
    conv = toolace.convert_row(7, row(
        ("user", "time?"),
        ("assistant", '[Get Time (UTC)(zone="CET"), ping()]'),
        ("tool", '[{"name": "ping", "results": {}}]'),
        ("assistant", "It is noon."),
    ))
    assert [t["name"] for t in conv.tools] == ["Get Time (UTC)", "ping"]
    assert conv.messages[1]["tool_calls"] == [
        {"type": "function", "function": {"name": "Get Time (UTC)", "arguments": {"zone": "CET"}}},
        {"type": "function", "function": {"name": "ping", "arguments": {}}},
    ]
    assert conv.messages[2] == {"role": "tool", "content": '[{"name": "ping", "results": {}}]'}
    assert conv.messages[3] == {"role": "assistant", "content": "It is noon."}


@pytest.mark.parametrize("r, reason", [
    (row(("user", "x"), system="Here are the tools you can use:\n- **tool_name**: f"),
     "tool list not JSON (format-variation row)"),
    (row(("user", "x"), ("assistant", "[nope(a=1)]")), "call names a tool not in the list"),
    (row(("user", "x"), ("assistant", "[ping(a=bare)]")), "call does not parse: value not a literal"),
    (row(("assistant", "hi")), "conversation does not start with a user turn"),
])
def test_conversion_failures_have_reasons(r, reason):
    with pytest.raises(toolace.ConvertError) as err:
        toolace.convert_row(0, r)
    assert err.value.reason == reason


def test_empty_tool_list_converts():
    sys = SYSTEM.split("invoke:")[0] + "invoke:\n[]. \nShould you decide"
    conv = toolace.convert_row(0, row(("user", "x"), ("assistant", "No function fits."), system=sys))
    assert conv.tools == []


def test_context_line_is_kept_after_qwens_default_sentence():
    sys = SYSTEM.replace("functions.\n", "functions.\nThe current time is 2023-05-01 10:00:00.\n", 1)
    conv = toolace.convert_row(0, row(("user", "x"), ("assistant", "ok"), system=sys))
    assert conv.messages[0] == {"role": "system", "content": toolace.QWEN_DEFAULT_SYSTEM
                                + "\nThe current time is 2023-05-01 10:00:00."}
    today = SYSTEM.replace("functions.\nHere is", "functions.\nToday is 2022-03-20, Sunday..Here is", 1)
    assert toolace.context_lines(today) == ["Today is 2022-03-20, Sunday."]
    no_context = toolace.convert_row(0, row(("user", "x"), ("assistant", "ok")))
    assert no_context.messages[0]["role"] == "user"


def test_language_rule():
    assert toolace.is_chinese(row(("user", "x"), system="你是一个工具专家。" + SYSTEM))
    english_about_kanji = row(("user", "What does 水 mean?"))
    assert not toolace.is_chinese(english_about_kanji)
    assert toolace.has_any_cjk(english_about_kanji)


def fake_render(messages, add_generation_prompt):
    """A prefix-stable stand-in for Qwen's template."""
    out = "<sys>\n"
    for m in messages:
        body = m["content"] or "".join(f"<call {c['function']['name']}>" for c in m["tool_calls"])
        out += f"<{m['role']}>\n{body}<end>\n"
    return out + ("<assistant>\n" if add_generation_prompt else "")


def test_assistant_spans_cover_assistant_turns_only():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}]},
        {"role": "tool", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    text, spans = toolace.assistant_spans(msgs, fake_render)
    assert [text[s:e] for s, e in spans] == ["<call f><end>", "done<end>"]


def test_tokens_in_spans_counts_inside_and_straddling():
    spans = [(10, 20)]
    offsets = [(0, 10), (10, 15), (15, 20), (18, 22), (20, 25), (5, 5)]
    assert toolace.tokens_in_spans(offsets, spans) == (2, 1)


def test_census_counts_hand_made_rows():
    rows = [
        row(("user", "x"), ("assistant", '[ping()]'), ("tool", "[]"), ("assistant", "ok")),
        row(("user", "x"), ("assistant", "Which zone?")),
        row(("user", "x"), system="你是一个工具专家。" + SYSTEM),
        row(("user", "x"), system="Here are the tools you can use:\nf"),
    ]
    c, out = toolace.census_and_convert(rows)
    assert (c.total, c.chinese, c.english, c.converted) == (4, 1, 3, 2)
    assert c.failures == {"tool list not JSON (format-variation row)": 1}
    assert (c.call_turns, c.calls, c.text_after_tool, c.text_after_user) == (1, 1, 1, 1)
    assert c.by_assistant_turns == {2: 1, 1: 1}
    assert (c.two_plus_tools, c.two_plus_tools_with_call) == (2, 1)
    assert [o.index for o in out] == [0, 1]
