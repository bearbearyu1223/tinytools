"""``rescore``'s pure parts: carrying a section over from an earlier capture."""

from tinytools.cli import WALL_CLOCK, section_lines
from tinytools.report import Report


def capture(tmp_path):
    out = tmp_path / "c.txt"
    rep = Report("t", out=out)
    rep.section("First")
    rep.kv("a", 1)
    rep.section(WALL_CLOCK)
    rep.table(["pass", "tasks"], [["greedy", 3]])
    rep.close()
    return out.read_text()


def test_section_lines_takes_one_section_body(tmp_path):
    text = capture(tmp_path)
    assert [line.split() for line in section_lines(text, "First")] == [["a", "1"]]
    body = section_lines(text, WALL_CLOCK)
    assert body[0].split() == ["pass", "tasks"] and body[-1].split() == ["greedy", "3"]


def test_section_lines_missing_section(tmp_path):
    assert section_lines(capture(tmp_path), "Nope") is None
