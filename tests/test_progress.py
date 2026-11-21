"""Progress must stay on stderr: stdout is the receipt document."""

from tinytools import progress


def test_training_reports_on_stderr_only(capsys, monkeypatch):
    monkeypatch.delenv("TINYTOOLS_QUIET", raising=False)
    on_step = progress.training("unit test run", 8)
    for step in range(8):
        on_step(step, 1.0 / (step + 1))
    out, err = capsys.readouterr()
    assert out == ""
    assert "unit test run, 8 steps" in err
    # Not a terminal under pytest, so one line per quarter of the run.
    assert [line.split()[1] for line in err.splitlines() if line.startswith("    step")] == [
        "2/8", "4/8", "6/8", "8/8"]


def test_quiet_silences_everything(capsys, monkeypatch):
    monkeypatch.setenv("TINYTOOLS_QUIET", "1")
    on_step = progress.training("silent", 4)
    for step in range(4):
        on_step(step, 0.5)
    progress.status("nothing")
    assert capsys.readouterr() == ("", "")


def test_items_reports_on_stderr_only(capsys, monkeypatch):
    monkeypatch.delenv("TINYTOOLS_QUIET", raising=False)
    on_item = progress.items("unit test pass", 4)
    for done in range(1, 5):
        on_item(done)
    out, err = capsys.readouterr()
    assert out == ""
    assert "unit test pass, 4 items" in err
    assert sum(line.startswith("    item") for line in err.splitlines()) == 4
