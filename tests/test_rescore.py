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


# -- bfcl-rescore --split eval: filter an all-split file (needs bfcl-eval's data, no model) ----


def _all_split_irrelevance(tmp_path, monkeypatch):
    import json

    from tinytools import bfcl

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs")
    (tmp_path / "outputs").mkdir()
    ids = [it.id for it in bfcl.load_items("irrelevance", 400, "all")]
    src = tmp_path / "outputs" / "bfcl-Qwen2.5-0.5B-Instruct-greedy-limit400-replies.jsonl"
    src.write_text("".join(json.dumps({"id": i, "category": "irrelevance", "reply": "No tool fits.",
                                       "hit_cap": False, "sample": 0, "dtype": "float32",
                                       "device": "mps"}) + "\n" for i in ids))
    return src, ids


def _args(path, split):
    import argparse

    return argparse.Namespace(replies=path, model="Qwen/Qwen2.5-0.5B-Instruct", seed=0, split=split)


def test_bfcl_rescore_split_eval_filters_to_eval(tmp_path, monkeypatch):
    import json

    from tinytools import bfcl

    src, ids = _all_split_irrelevance(tmp_path, monkeypatch)
    before = src.read_bytes()
    cli.cmd_bfcl_rescore(_args(src, "eval"))
    assert src.read_bytes() == before  # the all-split file is not touched
    out = tmp_path / "outputs"
    rows = [json.loads(x) for x in
            (out / "bfcl-Qwen2.5-0.5B-Instruct-eval-greedy-limit400-replies.jsonl").read_text().splitlines()]
    dev = set(bfcl.dev_slice("irrelevance"))
    assert [r["id"] for r in rows] == [i for i in ids if i not in dev] and len(rows) == 216
    text = (out / "bfcl-Qwen2.5-0.5B-Instruct-eval-greedy-limit400.txt").read_text()
    assert "eval: every item minus the dev slice" in text and "WARNING" not in text
    assert any(line.split()[:2] == ["irrelevance", "216"] for line in text.splitlines())
    # the eval file it wrote rescores to the same capture
    first = text
    cli.cmd_bfcl_rescore(_args(out / "bfcl-Qwen2.5-0.5B-Instruct-eval-greedy-limit400-replies.jsonl",
                               None))
    assert (out / "bfcl-Qwen2.5-0.5B-Instruct-eval-greedy-limit400.txt").read_text() \
        .replace("  (the eval items of a --split all replies file)\n", "") == first \
        .replace("  (the eval items of a --split all replies file)\n", "")


def test_bfcl_rescore_split_eval_refuses_a_non_all_file(tmp_path, monkeypatch):
    src, _ = _all_split_irrelevance(tmp_path, monkeypatch)
    dev = src.with_name("bfcl-Qwen2.5-0.5B-Instruct-dev-greedy-limit400-replies.jsonl")
    src.rename(dev)
    with pytest.raises(SystemExit, match="filters an all-split file"):
        cli.cmd_bfcl_rescore(_args(dev, "eval"))
