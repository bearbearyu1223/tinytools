"""``rescore`` and ``bfcl-rescore``'s pure parts: carrying a section over from an
earlier capture, reading a replies file's name and setup, and the Setup labels."""

import pytest

from tinytools import cli
from tinytools.cli import WALL_CLOCK, section_lines
from tinytools.device import determinism_label, dtype_name
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


# -- bfcl-rescore and the Setup fields ------------------------------------------------


def test_bfcl_replies_name():
    p = cli.parse_bfcl_replies_name
    assert p("bfcl-Qwen2.5-0.5B-Instruct-greedy-limit400-replies.jsonl") == {
        "model": "Qwen2.5-0.5B-Instruct", "split": "all", "mode": "greedy", "samples": 1,
        "temperature": 0.0, "limit": 400}
    got = p("bfcl-Qwen2.5-0.5B-Instruct-eval-samples8-t1-limitall-replies.jsonl")
    assert (got["split"], got["samples"], got["temperature"], got["limit"]) == ("eval", 8, 1.0, None)
    assert p("score-Qwen2.5-0.5B-Instruct-test-greedy-limitall-replies.jsonl") is None
    # the name a capture gets and the name parsed back agree
    for split, mode, limit in [("all", "greedy", 400), ("dev", "samples8-t1", None)]:
        name = f"bfcl-{cli._bfcl_tag('Qwen/Qwen2.5-0.5B-Instruct', split, mode, limit)}-replies.jsonl"
        got = p(name)
        assert (got["split"], got["mode"], got["limit"]) == (split, mode, limit)


def test_recorded_setup():
    rows = [{"device": "mps", "dtype": "float32"}] * 2
    assert cli.recorded_setup(rows, None) == ("mps", "float32")
    header = "device        : mps (Apple arm64)\ntorch         : 2.13.0\n"
    assert cli.recorded_setup([{}], header) == ("mps", "float32")
    with pytest.raises(ValueError, match="CUDA"):
        cli.recorded_setup([{}], "device        : cuda (NVIDIA A100)\n")
    with pytest.raises(ValueError, match="no earlier capture"):
        cli.recorded_setup([{}], None)
    with pytest.raises(ValueError, match="mixed"):
        cli.recorded_setup([{"device": "mps", "dtype": "float32"},
                            {"device": "cuda", "dtype": "bfloat16"}], None)


@pytest.mark.parametrize("samples, temperature, ok", [
    (1, 0.0, True), (8, 1.0, True), (8, 0.0, False), (1, 1.0, False)])
def test_greedy_or_sampled_never_mislabelled(samples, temperature, ok):
    if ok:
        cli._check_sampling(samples, temperature)
    else:
        with pytest.raises(SystemExit):
            cli._check_sampling(samples, temperature)


def test_setup_labels():
    import torch

    assert dtype_name(torch.float32) == "float32" and dtype_name(torch.bfloat16) == "bfloat16"
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in determinism_label("cuda")
    assert determinism_label("mps").startswith("not set on mps")
