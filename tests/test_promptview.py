"""The readable view of r1's prompt: parts reassemble, token rows sum, the tool table is the JSON."""

import json
import re
from pathlib import Path

import pytest

from tinytools import promptview as pv

R1 = Path(__file__).resolve().parents[1] / "outputs" / "r1-prompt.txt"


def fake_tokens(text: str) -> list[tuple[int, int]]:
    """Offsets of a stand-in tokenizer: special markers, runs of word characters, one
    punctuation mark plus the newlines after it (as BPE joins '>' to '\\n\\n'), or a char."""
    pat = re.compile(r"<\|im_(?:start|end)\|>|\w+|[^\w\s]\n+|\s+|.", re.S)
    return [m.span() for m in pat.finditer(text)]


@pytest.fixture(scope="module")
def prompt() -> str:
    if not R1.exists():
        pytest.skip("outputs/r1-prompt.txt not present")
    return R1.read_text()


def test_parts_reassemble_to_the_exact_prompt(prompt):
    segs = pv.split_prompt(prompt)
    assert "".join(s.text for s in segs) == prompt
    assert [s.name for s in segs if s.kind == "part"] == list(pv.PARTS)
    # contiguous, in order, covering every character
    assert segs[0].start == 0 and segs[-1].end == len(prompt)
    assert all(a.end == b.start for a, b in zip(segs[:-1], segs[1:], strict=True))
    markers = "".join(s.text for s in segs if s.kind == "marker")
    assert markers.count(pv.START) == 3 and markers.count(pv.END) == 2
    assert pv.part(segs, "assistant").text == ""
    assert pv.part(segs, "user").text.startswith('Add "Coffee chat"')
    assert pv.part(segs, "tools").text.startswith("# Tools")
    assert pv.part(segs, "format").text.endswith("</tool_call>")


def test_split_prompt_refuses_text_that_is_not_turns():
    with pytest.raises(ValueError):
        pv.split_prompt("stray" + pv.START + "user\nhi" + pv.END + "\n")


SMALL = (f"{pv.START}system\nSys.\n\n# Tools\n\nPre:\n<tools>\n{{\"a\": 1}}\n</tools>\n\n"
         f"Fmt<tool_call></tool_call>{pv.END}\n{pv.START}user\nHi{pv.END}\n"
         f"{pv.START}assistant\n")


def test_split_prompt_on_a_small_prompt():
    segs = pv.split_prompt(SMALL)
    parts = {s.name: s.text for s in segs if s.kind == "part"}
    assert parts == {"system": "Sys.\n\n", "tools": "# Tools\n\nPre:\n<tools>\n{\"a\": 1}\n"
                     "</tools>\n\n", "format": "Fmt<tool_call></tool_call>", "user": "Hi",
                     "assistant": ""}
    assert pv.tools_in(SMALL) == [{"a": 1}]


def test_a_token_across_a_part_edge_is_its_own_row():
    offsets = fake_tokens(SMALL)
    i = SMALL.index("Hi")
    k = next(j for j, (a, _) in enumerate(offsets) if a == i)
    merged = offsets[:k] + [(i, offsets[k + 1][1])] + offsets[k + 2:]  # "Hi" + "<|im_end|>"
    c = pv.part_tokens(SMALL, merged)
    assert c.across_parts == 1 and c.parts["user"] == 0
    assert c.markers + sum(c.parts.values()) + c.across_parts == len(merged)


def test_token_rows_sum_to_the_total(prompt):
    offsets = fake_tokens(prompt)
    c = pv.part_tokens(prompt, offsets)
    assert c.total == len(offsets)
    assert c.markers + sum(c.parts.values()) + c.across_parts == c.total
    assert c.preamble + c.tool_list + c.across_list_end + c.after_list == c.parts["tools"]
    assert c.across_list_end == 1  # the stand-in joins '>' of </tools> to '\n\n' too
    assert c.across_parts == 0


def test_param_table_matches_the_json(prompt):
    tool = next(t for t in pv.tools_in(prompt)
                if t["function"]["name"] == "create_calendar_event")
    rows = pv.params(tool)
    props = tool["function"]["parameters"]["properties"]
    required = tool["function"]["parameters"]["required"]
    assert [r.name for r in rows] == list(props)
    for r in rows:
        assert r.required == (r.name in required)
        assert r.description == props[r.name]["description"]
        assert r.type.split(" ")[-1] == props[r.name]["type"]
    att = next(r for r in rows if r.name == "attendees")
    assert att.type == props["attendees"]["items"]["type"] + " array"
    lines = pv.param_table(rows, 72)
    assert all(len(x) <= 72 and x == x.rstrip() for x in lines)
    # every description's words appear, in order, in the table
    body = " ".join(" ".join(lines[2:]).split())
    for r in rows:
        assert " ".join(r.description.split()) in body


def test_tool_lines_are_the_prompts_own_json(prompt):
    lines = pv.tool_lines(prompt)
    assert len(lines) == 3
    for line, tool in zip(lines, pv.tools_in(prompt), strict=True):
        assert line in prompt.split("\n")
        assert json.dumps(tool, ensure_ascii=False) == line


def test_wrap_breaks_at_spaces_only():
    lines = pv.wrap("The day, as 2026-03-11 YYYY-MM-DD.\n\nend", 12)
    assert lines == ["The day, as", "2026-03-11", "YYYY-MM-DD.", "", "end"]
    assert not any(x.endswith("\\") for x in lines)


def test_command_on_r1_rows_sum_to_540_and_388(prompt, tmp_path, monkeypatch, capsys):
    """The command itself, with Qwen's tokenizer: r1a's 540 and 388, every line <= 74."""
    from tinytools import cli

    monkeypatch.setattr(cli, "_prompt_view_path", lambda *a: tmp_path / "view.txt")
    cli.cmd_prompt_view(cli.build_parser().parse_args(["prompt-view"]))
    out = (tmp_path / "view.txt").read_text()
    assert out == capsys.readouterr().out
    assert all(len(x) <= 74 and x == x.rstrip() and not x.endswith("\\")
               for x in out.split("\n"))
    rows = dict(re.findall(r"^  (\S.*?)\s{2,}(\d+)$", out, re.M))
    assert rows["prompt tokens (indented rows aside)"] == "540"
    assert re.search(r"^    the tool list itself\s+388$", out, re.M)
    top = ["turn markers", *[f"[{p}]" for p in pv.PARTS], "tokens across a part's edge"]
    assert sum(int(rows[k]) for k in top) == 540
