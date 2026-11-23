"""Post 9's main run, model-free: the experiment file, which pairs are whole, the training
determinism comparison, and sft-check's pure checks (<|im_end|> scored, the system prompt's
shape, the worst micro-batch)."""

import json
import shlex
from pathlib import Path

import pytest
import torch

from tinytools import lambda_run as lr
from tinytools import sftrun as R
from tinytools import train as T
from tinytools.cli import build_parser
from tinytools.lora import save_adapter

ROOT = Path(__file__).resolve().parents[1]


def _main():
    return lr.parse_experiment((ROOT / "experiments" / "main.toml").read_text(), "main")


def test_the_main_experiment_parses_and_every_command_is_real():
    e = _main()
    assert (e.post, e.instance_type, e.max_hours, e.cap_usd) == (9, "gpu_1x_a100_sxm4", 8.0, 16.0)
    assert e.expected_hours <= e.max_hours and e.setup_record == "outputs/main/setup.json"
    parser = build_parser()
    commands = []
    for c in e.commands:
        if c == f"echo {lr.CHECKPOINT_MARK}":
            commands.append("FETCH")
            continue
        words = shlex.split(c.replace('"$TINYTOOLS_DEADLINE_EPOCH"', "1700000000"))
        assert words[:3] == ["uv", "run", "tinytools"], c
        args = parser.parse_args(words[3:])  # unknown flags would exit
        commands.append(words[3])
        if "--deadline-epoch" in c:
            assert args.deadline_epoch == 1700000000
        # no sft command runs in quick mode (engine-check --quick is the preflight)
        assert not (words[3].startswith("sft-") and getattr(args, "quick", False))
    # the stages, in the spec's order: data, checks, engine, untrained, the pairs, the report
    assert commands == ["toolace-convert", "sft-data", "sft-check", "engine-check", "sft-score",
                        "FETCH", "sft-pair", "FETCH", "sft-pair", "FETCH", "sft-pair", "FETCH",
                        "sft-train", "sft-report", "rescore", "rescore", "bfcl-rescore"]
    pairs = [c for c in e.commands if "sft-pair" in c]
    assert [shlex.split(c)[shlex.split(c).index("--seed") + 1] for c in pairs] == ["1", "2", "3"]
    assert "--seeds 1,2,3" in e.commands[1]
    assert "--repeat" in e.commands[12] and "--seed 1" in e.commands[12]
    # the batch size is part of every scoring recipe (plan 3c)
    for c in e.commands:
        if c.split()[3:4] in (["sft-check"], ["sft-score"], ["sft-pair"]):
            assert "--batch-size 64" in c, c


def test_the_main_experiment_fetches_what_the_run_writes():
    e = _main()
    want = ["outputs/main/checks.json", "outputs/main/setup.json",
            "outputs/main/untrained/score.json", "outputs/main/real-seed2/train.json",
            "outputs/main/control-seed3/losses.jsonl", "outputs/main/real-seed1/adapter.safetensors",
            "outputs/main/real-seed1-repeat/adapter.safetensors",
            f"outputs/main/untrained/{R.OURS_FILE}", f"outputs/main/control-seed1/{R.BFCL_FILE}",
            f"outputs/main/real-seed3/{R.SAMPLE_FILE}", "outputs/sft.txt", "outputs/sft-check.txt",
            "outputs/sft-pair-seed2.txt", "outputs/sft-score-untrained.txt",
            "outputs/score-Qwen2.5-1.5B-Instruct-test-greedy-limitall-replies.jsonl",
            "outputs/bfcl-Qwen2.5-1.5B-Instruct-eval-greedy-limitall.txt"]
    for f in want:
        assert any(Path(f).match(p) for p in e.fetch), f
    # the merge check's B = 0 adapter and temporary files are not fetched
    for f in ["outputs/main/b0-adapter.safetensors"]:
        assert not any(Path(f).match(p) for p in e.fetch), f


def _run(d: Path, train=True, score=True):
    d.mkdir(parents=True)
    if train:
        (d / "train.json").write_text("{}")
    if score:
        (d / "score.json").write_text("{}")


def test_only_whole_pairs_are_reported(tmp_path):
    _run(tmp_path / "untrained", train=False)
    for s in (1, 2):
        _run(tmp_path / f"real-seed{s}")
        _run(tmp_path / f"control-seed{s}")
    _run(tmp_path / "real-seed3")
    _run(tmp_path / "control-seed3", score=False)  # cut off while scoring
    _run(tmp_path / "real-seed1-repeat", score=False)
    seeds, partial = R.complete_seeds(tmp_path)
    assert seeds == [1, 2] and partial == ["real-seed3", "control-seed3"]
    assert R.complete_seeds(tmp_path / "missing") == ([], [])


def test_determinism_check_compares_losses_init_rows_and_tensors(tmp_path):
    a = {"A": torch.ones(2, 3), "B": torch.zeros(4, 2)}
    b = {"A": torch.ones(2, 3), "B": torch.full((4, 2), 1e-3)}
    save_adapter(tmp_path / "a.safetensors", a, {})
    save_adapter(tmp_path / "b.safetensors", b, {})
    first = {"losses": [1.0, 0.5], "init_sha256": "i", "first_step_rows": ["r1", "r2"],
             "adapter_sha256": T.tensors_sha256(a)}
    same = R.determinism_check(first, dict(first), tmp_path / "a.safetensors",
                               tmp_path / "a.safetensors")
    assert same == {"same LoRA init (A)": True, "same first-step rows": True,
                    "losses identical, step by step": "2 of 2",
                    "largest loss difference": "0.000e+00", "adapters bitwise identical": True}
    other = dict(first, losses=[1.0, 0.5001], adapter_sha256=T.tensors_sha256(b))
    d = R.determinism_check(first, other, tmp_path / "a.safetensors", tmp_path / "b.safetensors")
    assert d["losses identical, step by step"] == "1 of 2"
    assert d["adapters bitwise identical"] is False
    assert d["largest adapter difference"] == "1.000e-03"


def test_im_end_is_counted_only_when_scored_as_one_token():
    text = "<|im_start|>assistant\nhi<|im_end|>\n"
    s = text.index("hi")
    e = text.index("<|im_end|>") + len("<|im_end|>")
    offsets = [(0, 12), (12, 22), (22, s), (s, s + 2), (s + 2, e), (e, e + 1)]
    assert R.im_end_scored(text, offsets, [False, False, False, True, True, False],
                           [(s, e)]) == (1, 1)
    assert R.im_end_scored(text, offsets, [False, False, False, True, False, False],
                           [(s, e)]) == (1, 0)  # masked
    split = offsets[:4] + [(s + 2, s + 6), (s + 6, e), (e, e + 1)]
    assert R.im_end_scored(text, split, [False] * 3 + [True] * 3 + [False], [(s, e)]) == (1, 0)


def test_system_shape():
    with_tools = ("<|im_start|>system\nYou are Qwen.\nToday is 2022-03-20, Sunday.\n\n# Tools\n\n"
                  "...<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n")
    assert R.system_shape(with_tools) == ["You are Qwen.", "Today is 2022-03-20, Sunday."]
    bare = "<|im_start|>system\nYou are Qwen.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    assert R.system_shape(bare) == ["You are Qwen."]


def test_the_worst_micro_batch_has_the_most_token_slots():
    s = T.Settings(batch_rows=4, epochs=1, token_budget=4000)
    lengths = [3000, 100, 900, 950, 1000, 990, 200, 300]
    mb = R.worst_micro_batch(lengths, s)
    slots = T.padded_tokens([mb], lengths)
    for st in T.plan_steps(len(lengths), 1, 1, 4):
        for other in T.micro_batches(st.rows, lengths, 4000):
            assert T.padded_tokens([other], lengths) <= slots


def test_lengths_census():
    c = R.lengths_census([100, 2000, 3100], 3072)
    assert (c["rows"], c["longest"], c["over 1024"], c["over 3072"]) == (3, 3100, 2, 1)


@pytest.mark.parametrize("quick", [False, True])
def test_modes(tmp_path, quick):
    m = R.mode(quick, tmp_path)
    assert m.out.name == ("main-quick" if quick else "main")
    assert (m.settings is T.QUICK) == quick and (m.ours_limit is None) != quick
    assert R.run_label("real", 1, True) == "real-seed1-repeat"
    assert json.dumps(m.suffix)
