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


def test_a_real_run_folds_a_wide_line_instead_of_dying(monkeypatch, capsys):
    from tinytools import report as R

    monkeypatch.delenv(R.STRICT_ENV, raising=False)
    rep = R.Report("t")
    rep.kv("copied for rescore", "score-Qwen2.5-1.5B-Instruct-test-greedy-limitall-replies.jsonl")
    out = capsys.readouterr()
    assert all(len(x) <= R.WIDTH for x in out.out.splitlines())
    assert "".join(x.strip() for x in out.out.splitlines()).endswith("limitall-replies.jsonl")
    assert "folded" in out.err


def test_the_rescore_copies_print_within_the_width_with_the_real_names(capsys):
    from pathlib import Path

    from tinytools import report as R
    from tinytools import sftrun

    rep = R.Report("t")  # strict in tests: raises if a line is too wide
    name = "Qwen2.5-1.5B-Instruct"
    sftrun.print_copies(rep, [Path(f"score-{name}-test-greedy-limitall-replies.jsonl"),
                              Path(f"bfcl-{name}-eval-greedy-limitall-replies.jsonl"),
                              Path(f"score-{name}-train-sample-samples8-t1-limitall-replies.jsonl")])
    joined = "".join(x.strip() for x in capsys.readouterr().out.splitlines())
    assert "train-sample-samples8-t1-limitall-replies.jsonl" in joined


def test_a_full_width_table_keeps_its_rule_within_the_width(tmp_path):
    # Rows exactly WIDTH wide: the rule used to be WIDTH + 2 and fold into a stray "    --"
    # line (round-3 review). Under the strict env a fold would raise, so this also proves none.
    head = ["name", "a", "b"]
    rows = [["x" * (WIDTH - 2 - 4 - 2 * 2 - 4 - 4) + "abcd", "1234", "5678"]]
    lines = _lines(tmp_path, lambda rep: rep.table(head, rows))
    assert max(len(x) for x in lines) == WIDTH  # the row fills the width
    rule = lines[1]
    assert set(rule.strip()) == {"-"} and len(rule) <= WIDTH
    assert len(lines) == 3  # header, rule, row: nothing folded


def test_a_narrow_table_keeps_its_two_column_overhang(tmp_path):
    # Tables that never folded print exactly as before (captures stay byte-identical).
    lines = _lines(tmp_path, lambda rep: rep.table(["name", "n"], [["ab", 12]]))
    assert lines[1] == "  " + "-" * (4 + 2 + 2 * 2)
