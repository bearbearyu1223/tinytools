"""The figure parsers, on small captures printed by the real ``Report``."""

import json

import pytest

from tinytools import figures as F
from tinytools.classify import CALL_OUTCOMES, NO_FIT_OUTCOMES, short_label
from tinytools.ours import wrap_lines
from tinytools.report import WIDTH, Report


def printed(capsys, build) -> str:
    rep = Report("tinytools test")
    rep.print("=" * WIDTH)
    build(rep)
    return capsys.readouterr().out


def block(rep, text, indent=4):  # cli._block
    for line in wrap_lines(text, WIDTH - indent):
        rep.print((" " * indent + line).rstrip())


# -- outcome tables -----------------------------------------------------------------

CALL_HEAD = [short_label(o) for o in CALL_OUTCOMES]
NO_FIT_HEAD = [short_label(o) for o in NO_FIT_OUTCOMES]


def score_capture(rep, *, choose=(1, 0, 0, 0, 2, 7), tricky=(0, 1, 2, 0, 3, 4),
                  total=None, no_fit=(3, 7), decoding="greedy (argmax)", last=True):
    rep.section("Setup")
    rep.kv("model", "Qwen/Qwen2.5-0.5B-Instruct")
    rep.kv("decoding", decoding)
    rep.section(F.CALL_SECTION)
    total = total or [a + b for a, b in zip(choose, tricky, strict=True)]
    rep.table(["kind", "n", *CALL_HEAD],
              [["choose+fill", sum(choose), *choose], ["tricky args", sum(tricky), *tricky],
               ["total", sum(choose) + sum(tricky), *total]])
    rep.note("wrong fn: another tool; wrong n: the right tool, more than one")
    rep.section(F.NO_FIT_SECTION)
    rep.table(["kind", "n", *NO_FIT_HEAD], [["no tool fits", sum(no_fit), *no_fit]])
    rep.note("declined: no <tool_call> tag (right); called anyway: any tag.")
    if last:
        rep.section(F.LAST_SECTION)


def test_score_outcomes(capsys):
    out = F.parse_score_outcomes(printed(capsys, score_capture))
    assert out.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert [(r.label, r.needs_call, r.n) for r in out.rows] == [
        ("choose+fill", True, 10), ("tricky arguments", True, 10), ("no tool fits", False, 10)]
    assert out.rows[0].counts["right"] == 7 and out.rows[0].counts["no call"] == 1
    assert out.rows[2].counts == {"declined": 3, "called anyway": 7}


@pytest.mark.parametrize("kwargs, msg", [
    (dict(total=[1, 1, 2, 0, 5, 12]), "total row"),
    (dict(decoding="8 samples, temperature 1"), "not a greedy"),
    (dict(last=False), "still being written"),
])
def test_score_outcomes_raise(capsys, kwargs, msg):
    text = printed(capsys, lambda rep: score_capture(rep, **kwargs))
    with pytest.raises(F.CaptureError, match=msg):
        F.parse_score_outcomes(text)


def test_outcomes_must_sum_to_n(capsys):
    def build(rep):
        rep.section("Setup")
        rep.kv("model", "m")
        rep.kv("decoding", "greedy (argmax)")
        rep.section(F.CALL_SECTION)
        rep.table(["kind", "n", *CALL_HEAD], [["choose+fill", 9, 1, 0, 0, 0, 2, 7],
                                             ["tricky args", 1, 0, 0, 0, 0, 0, 1],
                                             ["total", 10, 1, 0, 0, 0, 2, 8]])
        rep.section(F.LAST_SECTION)
    with pytest.raises(F.CaptureError, match="sum to 10, n = 9"):
        F.parse_score_outcomes(printed(capsys, build))


def test_bfcl_outcomes(capsys):
    def build(rep):
        rep.section("Setup")
        rep.kv("model", "Qwen/Qwen2.5-0.5B-Instruct")
        rep.kv("decoding", "greedy")
        rep.section(F.CALL_SECTION)
        rows = [[c, 2, 0, 0, 0, 0, 1, 1] for c in ("simple", "multiple", "parallel", "par_multi")]
        rep.table(["category", "n", *CALL_HEAD], rows + [["total", 8, 0, 0, 0, 0, 4, 4]])
        rep.note("wrong fn: names differ as a set; wrong n: the right names, too")
        rep.section(F.NO_FIT_SECTION)
        rep.table(["category", "n", *NO_FIT_HEAD, "bfcl right"], [["irrelevance", 2, 1, 1, 1]])
        rep.section(F.LAST_SECTION)
    out = F.parse_bfcl_outcomes(printed(capsys, build))
    assert [r.label for r in out.rows] == ["simple", "multiple", "parallel", "parallel_multiple",
                                         "irrelevance"]
    assert out.rows[-1].counts == {"declined": 1, "called anyway": 1}


def test_bfcl_missing_category_raises(capsys):
    def build(rep):
        rep.section("Setup")
        rep.kv("model", "m")
        rep.kv("decoding", "greedy")
        rep.section(F.CALL_SECTION)
        rep.table(["category", "n", *CALL_HEAD],
                  [["simple", 2, 0, 0, 0, 0, 1, 1], ["total", 2, 0, 0, 0, 0, 1, 1]])
        rep.section(F.LAST_SECTION)
    with pytest.raises(F.CaptureError, match="expected"):
        F.parse_bfcl_outcomes(printed(capsys, build))


# -- check-demo ----------------------------------------------------------------------

def check_capture(rep, verdicts=((True, True), (False, True), (False, False))):
    calls = [("create_calendar_event", {"title": "Budget sync", "duration_minutes": 90},
              {"duration_minutes": 90.0, "title": "Budget  sync"}),
             ("calculator", {"expression": "348 * 27"}, {"expression": "27 * 348"}),
             ("search_documents", {"query": "Q3 sales report"}, {"query": "sales report"})]
    for k, ((name, ref, got), (ex, run)) in enumerate(zip(calls, verdicts, strict=True), start=1):
        rep.section(f"Call {k}: title {k}")
        rep.note("request:")
        block(rep, f"request {k}")
        rep.note("reference call:")
        block(rep, json.dumps({"name": name, "arguments": ref}))
        rep.note("the call checked:")
        block(rep, json.dumps({"name": name, "arguments": got}))
        rep.kv("exact match", ex)
        rep.kv("execution match", run)
        rep.kv("execution match compared", "9396 vs 9396")


def test_check_demo(capsys):
    calls = F.parse_check_demo(printed(capsys, check_capture))
    assert [c.title for c in calls] == ["title 1", "title 2", "title 3"]
    assert calls[0].checked[1]["title"] == "Budget  sync"   # the double space survives
    assert F.differing_arguments(calls[0].reference[1], calls[0].checked[1]) == [
        "title", "duration_minutes"]
    F.two_checks_claims(calls)


def test_two_checks_claims_raise_on_other_verdicts(capsys):
    text = printed(capsys, lambda rep: check_capture(rep, ((True, True),) * 3))
    with pytest.raises(F.CaptureError, match="not pass both"):
        F.two_checks_claims(F.parse_check_demo(text))


# -- r1 -------------------------------------------------------------------------------

PROMPT = ('<|im_start|>system\nYou are Qwen.\n\n# Tools\n<tools>\n'
          + json.dumps({"type": "function", "function": {"name": "calculator", "description":
                        "Evaluate an arithmetic expression and return the number. " * 2}})
          + '\n</tools>\n\nCall a tool.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n'
          '<|im_start|>assistant\n')


def r1_capture(rep, prompt=PROMPT):
    call = json.dumps({"name": "calculator", "arguments": {"expression": "2 + 2"}})
    rep.section("One task, whole: test-0001 (choose+fill, calculator)")
    rep.note("in outputs/r1-prompt.txt):")
    rep.blank()
    block(rep, prompt)
    rep.blank()
    rep.kv("prompt tokens", 120)
    rep.kv("  the tool list (<tools> block)", 80)
    rep.section(F.R1_REPLY)
    rep.note("raw reply (up to the end-of-turn token):")
    block(rep, f"<tool_call>\n{call}\n</tool_call>")
    rep.note("parsed call:")
    block(rep, call)
    rep.note("the tool's result, fed back as the next turn:")
    block(rep, '<|im_start|>user\n<tool_response>\n{"result": 4}\n</tool_response><|im_end|>')
    rep.note("reference call:")
    block(rep, call)
    rep.kv("outcome", "right")
    rep.kv("exact match / execution match", "yes / yes")


def test_round_trip(capsys):
    rt = F.parse_round_trip(printed(capsys, r1_capture), PROMPT)
    assert (rt.task_id, rt.kind, rt.tool) == ("test-0001", "choose+fill", "calculator")
    assert (rt.prompt_tokens, rt.tool_list_tokens) == (120, 80)
    assert rt.call == ("calculator", {"expression": "2 + 2"})
    assert rt.exact and rt.execution


def test_round_trip_prompt_must_match_the_file(capsys):
    text = printed(capsys, r1_capture)
    with pytest.raises(F.CaptureError, match="not r1-prompt.txt"):
        F.parse_round_trip(text, PROMPT.replace("2 + 2", "3 + 3"))


def test_prompt_excerpt_spans():
    ex, tools, req = F.prompt_excerpt(PROMPT, 70)
    assert ex[tools[0]:tools[1]].startswith("<tools>\n")
    assert ex[tools[0]:tools[1]].endswith("\n</tools>")
    assert ex[tools[0]:tools[1]].splitlines()[1] == (
        '{"type": "function", "function": {"name": "calculator", ...}}')
    assert ex[req[0]:req[1]] == "What is 2 + 2?"
    assert all(len(line) <= 70 for line in ex.splitlines() if "What is" not in line)


# -- r4b ------------------------------------------------------------------------------

def sft_capture(rep, control_user=19):
    rep.section("One training row and its control twin (toolace-4231, seed 1)")
    rep.kv("tools offered", "getMedication, getDiagnosis")
    for label, user, asst in (("real arm", 19, (27, 23, 4)), ("control", control_user,
                                                              (29, 25, 4))):
        rep.blank()
        rep.note(f"{label}, turn by turn:")
        rows = [["system", 286, 0, 286], ["user", user, 0, user], ["assistant", *asst]]
        rows.append(["total", sum(r[1] for r in rows), sum(r[2] for r in rows),
                     sum(r[3] for r in rows)])
        rep.table(["turn", "tokens", "scored", "masked"], rows)
    rep.note("user:")
    block(rep, "I've been diagnosed with hypertension.")
    rep.note("real call:")
    block(rep, json.dumps({"name": "getMedication", "arguments": {"condition": "x"}}))
    rep.note("control call:")
    block(rep, json.dumps({"name": "getDiagnosis", "arguments": {"patient_id": "3056"}}))
    rep.kv("control's tool", "getDiagnosis (replaced getMedication)")
    rep.kv("arguments", "another row: copied from toolace-723")


def test_sft_example(capsys):
    ex = F.parse_sft_example(printed(capsys, sft_capture))
    assert (ex.row_id, ex.seed, ex.tools) == ("toolace-4231", 1, ("getMedication", "getDiagnosis"))
    assert [(t.role, t.tokens, t.scored) for t in ex.real] == [
        ("system", 286, 0), ("user", 19, 0), ("assistant", 27, 23)]
    assert (ex.drawn, ex.replaced, ex.source, ex.copied_from) == (
        "getDiagnosis", "getMedication", "another row", "toolace-723")


# -- helpers --------------------------------------------------------------------------

def test_unwrap_inverts_block():
    text = '{"name": "f", "arguments": {"title": "Budget  sync", "x": "' + "y" * 90 + '"}}'
    lines = ["    " + line for line in wrap_lines(text, WIDTH - 4)]
    assert F.unwrap(lines, 4, "t") == text


def test_wrap_offsets_round_trip():
    text = "alpha beta gamma delta\nepsilon " + "z" * 30
    pieces = F.wrap_offsets(text, 10)
    assert all(len(p) <= 10 for _, p in pieces)
    for off, p in pieces:
        assert text[off:off + len(p)] == p


def test_argument_span_and_join():
    t = json.dumps({"name": "f", "arguments": {"a": 90.0, "b": "x"}})
    a, b = F.argument_span(t, "a", 90.0)
    assert t[a:b] == '"a": 90.0'
    assert F.join_names(["a"]) == "a" and F.join_names(["a", "b", "c"]) == "a, b and c"


def test_kv_needs_the_exact_label(capsys):
    text = printed(capsys, lambda rep: (rep.kv("execution match", "yes"),
                                        rep.kv("execution match compared", "1 vs 1")))
    assert F.kv(text.splitlines(), "execution match", "t") == "yes"
    with pytest.raises(F.CaptureError):
        F.kv(text.splitlines(), "exact match", "t")
