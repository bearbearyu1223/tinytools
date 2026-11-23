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
    # max_hours 7.5 since 29aeae0: the $16 cap less the failed first attempt's $0.87
    assert (e.post, e.instance_type, e.max_hours, e.cap_usd) == (9, "gpu_1x_a100_sxm4", 7.5, 16.0)
    assert e.expected_hours <= e.max_hours and e.setup_record == "outputs/main/setup.json"
    parser = build_parser()
    commands = []
    # the CUDA allocator setting first, for every command after it (T8 review B1)
    assert e.commands[0] == f"export PYTORCH_CUDA_ALLOC_CONF={R.ALLOC_CONF}"
    for c in e.commands[1:]:
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
                        "sft-score", "FETCH", "sft-pair", "FETCH", "sft-pair", "FETCH", "sft-pair", "FETCH",
                        "sft-train", "sft-report", "rescore", "rescore", "bfcl-rescore"]
    pairs = [c for c in e.commands if "sft-pair" in c]
    assert [shlex.split(c)[shlex.split(c).index("--seed") + 1] for c in pairs] == ["1", "2", "3"]
    assert "--seeds 1,2,3" in e.commands[2]
    assert "--repeat" in e.commands[14] and "--seed 1" in e.commands[14]
    # the noise floor (r6a, T8 review B2): untrained, before the first pair
    assert "--untrained --noise-floor" in e.commands[6]
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
            f"outputs/main/{R.FLOOR}/score.json", f"outputs/main/{R.FLOOR}/{R.SAMPLE_FILE}",
            f"outputs/sft-score-{R.FLOOR}.txt",
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


def test_the_two_largest_micro_batches_have_the_most_token_slots():
    s = T.Settings(batch_rows=4, epochs=1, token_budget=4000)
    lengths = [3000, 100, 900, 950, 1000, 990, 200, 300, 2900, 50]
    two = R.largest_micro_batches(lengths, s)
    assert len(two) == 2 and sorted(two[0]) != sorted(two[1])
    every = {}  # each distinct micro-batch of seeds 1-3's plans -> its token slots
    for seed in (1, 2, 3):
        for st in T.plan_steps(len(lengths), seed, 1, 4):
            for mb in T.micro_batches(st.rows, lengths, 4000):
                every[tuple(sorted(mb))] = T.padded_tokens([mb], lengths)
    top = sorted(every.values(), reverse=True)[:2]
    assert [T.padded_tokens([mb], lengths) for mb in two] == top
    assert all(lengths[mb[0]] == max(lengths[i] for i in mb) for mb in two)  # longest first


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


# -- T8 review fixes: pinned sets (S4), the merge check (S3), the memory step (B1),
# -- r6a's noise floor and per-seed rows (B2, S7) ----------------------------------------


def test_the_sets_are_checked_against_the_pinned_sha256(tmp_path):
    import hashlib

    from tinytools import sft

    assert sorted(sft.SET_SHA256) == ["control-seed1.jsonl", "control-seed2.jsonl",
                                      "control-seed3.jsonl", "real.jsonl"]
    (tmp_path / "real.jsonl").write_text("not the Mac's\n")
    pins = R.set_pins(tmp_path)
    assert pins["real.jsonl"] == (hashlib.sha256(b"not the Mac's\n").hexdigest(),
                                  sft.SET_SHA256["real.jsonl"])
    assert pins["control-seed1.jsonl"][0] is None  # missing
    if (ROOT / "data" / "sft" / "real.jsonl").exists():  # this Mac's sets are the pinned ones
        assert all(h == want for h, want in R.set_pins(ROOT / "data" / "sft").values())


def test_logit_gap_is_the_error_as_a_share_of_the_adapters_effect():
    base = torch.zeros(3, 5)
    lora = torch.zeros(3, 5)
    lora[:, 1] = 2.0  # the adapter's effect: |effect| = sqrt(12)
    g = R.LogitGap()
    g.add(lora, lora, base)
    assert g.relative == 0.0 and g.argmax_same == 3 and g.positions == 3
    g = R.LogitGap()
    g.add(base, lora, base)  # nothing merged
    assert g.relative == pytest.approx(1.0)
    g = R.LogitGap()
    g.add(1.25 * lora, lora, base)  # a quarter too much
    assert g.relative == pytest.approx(0.25) and g.max_abs == pytest.approx(0.5)


def _tiny(tmp_path):
    from test_train import _tiny_model, _tok

    return _tok(), _tiny_model(tmp_path, _tok())


def test_the_random_b_merge_check_passes_a_right_merge_and_fails_a_wrong_one(tmp_path,
                                                                              monkeypatch):
    from tinytools import lora

    tok, model = _tiny(tmp_path)
    s = T.Settings(model=model)
    texts = ["What is 2 plus 3?", "Turn on the lights in the kitchen, please."]
    ok = R.merge_check_random_b(s, model, texts, torch.device("cpu"))
    rows = ok["rows"]
    assert ok["passes"] and ok["modules"] == 2 * 7
    assert rows["merged"]["relative"] < 1e-4  # float32 rounding
    assert rows[f"merged, {R.MERGE_CONTRAST:g} x the scaling"]["relative"] > R.MERGE_TOLERANCE
    assert rows["base (nothing merged)"]["relative"] == pytest.approx(1.0)
    real = lora.merge_into
    monkeypatch.setattr(lora, "merge_into", lambda m, st, scaling: real(m, st, scaling * 1.05))
    assert not R.merge_check_random_b(s, model, texts, torch.device("cpu"))["passes"]


def test_the_memory_step_runs_both_micro_batches_and_adamw(tmp_path):
    """On the CPU there is no peak counter: the step ran, checkpointing stays off; the
    record names both micro-batches and the 85% rule (CUDA only)."""
    from tinytools import sft

    tok, model = _tiny(tmp_path)
    tools = [{"name": "add", "description": "Add.", "parameters": {
        "type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}]
    recs = [sft.render(f"r{i}", tools, [{"role": "user", "content": "add " * (5 * i + 1)},
                                        {"role": "assistant", "content": "ok"}], tok)
            for i in range(8)]
    s = T.Settings(model=model, batch_rows=4, epochs=1,
                   token_budget=2 * max(r["n_tokens"] for r in recs))
    torch.use_deterministic_algorithms(True)
    try:
        mem = R.memory_probe(s, recs, torch.device("cpu"))
        assert torch.are_deterministic_algorithms_enabled()  # restored
    finally:
        torch.use_deterministic_algorithms(False)
    assert len(mem["micro_batches"]) == 2 and mem["headroom"] == 0.85
    assert not mem["peak_rule"] and mem["allocator"] is None
    assert mem["fits"] and not mem["needs_checkpointing"]
    assert [a["checkpointing"] for a in mem["attempts"]] == [False]


def _scores(label, rights, greedy=None):
    return R.Scores(label, {}, {}, greedy or {i: False for i in rights}, rights, {})


def test_r6a_reads_movements_beyond_the_noise_floor_and_prints_per_seed_call_rows(tmp_path):
    """A fabricated full-size train-sample (500 tasks): the floor's table comes first,
    each arm's table has per-seed rows for both denominators, the movements are read
    beyond the floor, and every line fits 74 columns (Report.print raises otherwise)."""
    import random

    from tinytools import tasks
    from tinytools.cli import _load_or_make_tasks
    from tinytools.ours import limit_per_kind
    from tinytools.report import Report

    sample = limit_per_kind(_load_or_make_tasks(tasks.DEFAULT_SEED)["train-sample"], None)
    rng = random.Random(0)
    ids = [t["id"] for t in sample]

    def draw():
        return {i: rng.choice([0, 0, 3, 5, 8, 8]) for i in ids}

    un, floor = _scores("untrained", draw()), draw()
    sft_ = [_scores(f"real-seed{s}", draw()) for s in (1, 2, 3)]
    ctl = [_scores(f"control-seed{s}", draw()) for s in (1, 2, 3)]
    rep = Report("t", out=tmp_path / "r6a.txt")
    R._r6a(rep, un, sft_, ctl, [1, 2, 3], sample, floor)
    rep.close()
    text = (tmp_path / "r6a.txt").read_text()
    assert text.index("untrained -> untrained again") < text.index("untrained -> SFT")
    for s in (1, 2, 3):
        assert f"seed {s}, need a call" in text and f"seed {s}, all kinds" in text
    assert text.count("largest beyond the floor") == 2
    assert "beyond" in text and "largest, tasks that need a call" not in text
    rep = Report("t", out=tmp_path / "nofloor.txt")
    R._r6a(rep, un, sft_, ctl, [1, 2, 3], sample, None)
    rep.close()
    assert "Not measured" in (tmp_path / "nofloor.txt").read_text()
