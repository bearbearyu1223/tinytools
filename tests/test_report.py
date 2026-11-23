"""The 74-column rule (T8 review S1): Report.print refuses a wider line; kv and note wrap
on request."""

import pytest

from tinytools.device import determinism_label
from tinytools.report import WIDTH, Report


def _lines(tmp_path, fn):
    rep = Report("t", out=tmp_path / "r.txt")
    fn(rep)
    rep.close()
    return (tmp_path / "r.txt").read_text().splitlines()


def test_a_line_wider_than_the_report_raises_before_it_is_written(tmp_path, capsys):
    rep = Report("t", out=tmp_path / "r.txt")
    rep.print("x" * WIDTH)
    with pytest.raises(ValueError, match="75 columns"):
        rep.print("x" * (WIDTH + 1))
    with pytest.raises(ValueError):
        rep.print("ok\n" + "y" * (WIDTH + 1))
    rep.close()
    assert (tmp_path / "r.txt").read_text() == "x" * WIDTH + "\n"
    assert "y" * WIDTH not in capsys.readouterr().out


def test_the_cuda_determinism_line_wraps_at_the_value_column(tmp_path):
    lines = _lines(tmp_path, lambda r: r.kv("determinism", determinism_label("cuda"), wrap=True))
    assert lines == ["  determinism                        deterministic algorithms on,",
                     "                                     CUBLAS_WORKSPACE_CONFIG=:4096:8"]
    short = _lines(tmp_path, lambda r: r.kv("determinism", determinism_label("mps"), wrap=True))
    assert short == ["  determinism                        not set on mps (the CUDA flags only)"]


def test_a_wrapped_note_keeps_its_indent(tmp_path):
    text = "  " + "word " * 40
    lines = _lines(tmp_path, lambda r: r.note(text, wrap=True))
    assert len(lines) > 1 and all(len(x) <= WIDTH and x.startswith("    word") for x in lines)
    assert " ".join(x.strip() for x in lines) == text.strip()
