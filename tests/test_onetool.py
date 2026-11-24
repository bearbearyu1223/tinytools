"""The one-tool rerun (plan 3g): the sft-data variant, the onetool profile, its report's
arithmetic, and experiments/onetool.toml."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import pytest
from test_sft import A, B, _converted, _tokenizer, call, say, toolmsg, user

from tinytools import effects as E
from tinytools import lambda_run as lr
from tinytools import onetool as O
from tinytools import sft
from tinytools import sftrun as R
from tinytools import train as T
from tinytools.cli import PROFILE_NAMES, build_parser

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data" / "sft"


# -- the data variant --------------------------------------------------------------------------


def _rec(rid: str, n_tools: int, messages: list[dict]) -> dict:
    return {"id": rid, "n_tools": n_tools, "messages": messages}


def test_without_rows_keeps_every_other_line_byte_for_byte():
    recs = [_rec("a", 2, [user("x"), call(("A", {"q": "x"}))]),
            _rec("b", 1, [user("w?"), say("No.")]),  # no call, 1 tool: out
            _rec("c", 0, [user("hi"), say("Hello!")]),
            _rec("d", 1, [user("w?"), say("Ünïcode, still no.")]),  # out
            _rec("e", 2, [user("w?"), say("None fits.")])]
    data = sft.jsonl_bytes(recs)
    cut, removed = sft.without_rows(data, (sft.R_NO_CALL_1,))
    assert removed == 2
    assert cut == sft.jsonl_bytes([recs[0], recs[2], recs[4]])
    assert sft.without_rows(data, ()) == (data, 0)


def test_the_variant_leaves_out_its_rows_after_the_drop_check():
    tok = _tokenizer()
    rows = [
        _converted(tok, "call2", [A, B], [user("find x"), call(("A", {"q": "x"})), toolmsg(),
                                          say("done")]),
        _converted(tok, "call1", [B], [user("count"), call(("B", {"n": 9}))]),  # R_OUT
        _converted(tok, "decline2", [A, B], [user("weather?"), say("No function fits.")]),
        _converted(tok, "decline1", [B], [user("weather?"), say("I can't do that.")]),
        _converted(tok, "empty1", [B], [user("weather?"), say("")]),  # dropped first
        _converted(tok, "chat0", [], [user("hi"), say("Hello!")]),
    ]
    full = sft.build_sets(rows, (), tok)
    var = sft.build_sets(rows, (), tok, leave_out=sft.VARIANTS["onetool"])
    assert [r["id"] for r in full.real] == ["call2", "decline2", "decline1", "chat0"]
    assert [r["id"] for r in var.real] == ["call2", "decline2", "chat0"]
    # the kept records are the full set's, byte for byte
    assert sft.jsonl_bytes(var.real) == sft.without_rows(sft.jsonl_bytes(full.real),
                                                         (sft.R_NO_CALL_1,))[0]
    assert var.rows_left_out == {sft.R_NO_CALL_1: 1} and full.rows_left_out == {}
    assert var.rows_dropped == full.rows_dropped == {sft.R_NO_CALL_1: 1}
    assert var.dropped == full.dropped and var.rows_converted == full.rows_converted
    assert var.rows_trained[sft.R_NO_CALL_1] == 0 and full.rows_trained[sft.R_NO_CALL_1] == 1
    assert var.controls == {}


def test_variant_names():
    assert sft.variant_file("onetool") == "real-onetool.jsonl" == T.PROFILES["onetool"].real_set
    assert set(sft.VARIANT_SHA256) == {sft.variant_file(v) for v in sft.VARIANTS}
    assert sft.VARIANTS["onetool"] == (sft.R_NO_CALL_1,)
    with pytest.raises(ValueError):
        sft.variant_file("other")
    # the four sets' pins are not touched by the variant
    assert list(sft.SET_SHA256) == ["real.jsonl", "control-seed1.jsonl", "control-seed2.jsonl",
                                    "control-seed3.jsonl"]


def test_the_variant_file_is_real_jsonl_without_the_514_rows():
    var, real = SFT_DIR / "real-onetool.jsonl", SFT_DIR / "real.jsonl"
    if not (var.exists() and real.exists()):
        pytest.skip("run `tinytools sft-data` and `sft-data --variant onetool` first")
    data = var.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sft.VARIANT_SHA256["real-onetool.jsonl"]
    raw = real.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sft.SET_SHA256["real.jsonl"]
    cut, removed = sft.without_rows(raw, (sft.R_NO_CALL_1,))
    assert (cut, removed) == (data, 514)
    assert raw.count(b"\n") == 8504 and data.count(b"\n") == 7990
    kinds = {sft.row_type(r["n_tools"], r["messages"])
             for r in map(json.loads, data.decode("utf-8").split("\n")[:-1])}
    assert sft.R_NO_CALL_1 not in kinds and all(r["n_tools"] != 1 for r in map(
        json.loads, data.decode("utf-8").split("\n")[:-1]))


def test_the_variant_regenerates_byte_for_byte():
    src = ROOT / sft.CONVERTED
    if not src.exists():
        pytest.skip("run `tinytools toolace-convert` first")
    raw = src.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sft.CONVERTED_SHA256
    tok = _tokenizer()
    converted = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    s = sft.build_sets(converted, (), tok, leave_out=sft.VARIANTS["onetool"])
    data = sft.jsonl_bytes(s.real)
    assert hashlib.sha256(data).hexdigest() == sft.VARIANT_SHA256["real-onetool.jsonl"]
    assert len(s.real) == 7990 and s.rows_left_out == {sft.R_NO_CALL_1: 514}
    assert sum(s.rows_trained.values()) + 514 == 8504


# -- the profile -------------------------------------------------------------------------------


def test_the_onetool_profile_is_main_with_its_set_and_one_arm():
    p = T.PROFILES["onetool"]
    assert p.full is T.MAIN and p.quick is T.QUICK
    assert (p.tokenizer, p.quick_tokenizer, p.embed_rows, p.stop_tokens) == (None, None, (), ())
    assert (p.variant, p.arms, p.real_set, p.data_capture) == (
        "onetool", ("real",), "real-onetool.jsonl", "sft-data-onetool.txt")
    for quick in (False, True):
        m = R.mode(quick, ROOT, profile="onetool")
        assert m.out.name == ("onetool-quick" if quick else "onetool")
        assert m.suffix == ("-onetool-quick" if quick else "-onetool")
        assert m.title == " --profile onetool"
        assert m.settings is (T.QUICK if quick else T.MAIN)
        assert (m.real_set, m.arms, m.tokenizer) == ("real-onetool.jsonl", ("real",), None)
    assert "onetool" in PROFILE_NAMES and set(PROFILE_NAMES) == set(T.PROFILES)


def test_main_and_base_keep_their_sets_and_arms():
    for name in ("main", "base"):
        p = T.PROFILES[name]
        assert (p.variant, p.arms, p.real_set, p.data_capture) == (
            None, T.ARMS, "real.jsonl", "sft-data.txt")
        m = R.mode(False, ROOT, profile=name)
        assert (m.real_set, m.arms) == ("real.jsonl", T.ARMS)
    assert R.profile_pins("real.jsonl") == sft.SET_SHA256
    assert R.profile_pins("real-onetool.jsonl") == {
        "real-onetool.jsonl": sft.VARIANT_SHA256["real-onetool.jsonl"]}
    assert R.set_path(ROOT, "real", 2).name == "real.jsonl"
    assert R.set_path(ROOT, "control", 2).name == "control-seed2.jsonl"
    assert R.set_path(ROOT, "real", 2, "real-onetool.jsonl").name == "real-onetool.jsonl"


def test_set_pins_reads_the_profiles_sets(tmp_path):
    (tmp_path / "real-onetool.jsonl").write_bytes(b"x\n")
    pins = R.set_pins(tmp_path, R.profile_pins("real-onetool.jsonl"))
    assert list(pins) == ["real-onetool.jsonl"]
    assert pins["real-onetool.jsonl"] == (hashlib.sha256(b"x\n").hexdigest(),
                                          sft.VARIANT_SHA256["real-onetool.jsonl"])


def test_seed_n_sets_init_and_shuffle_but_the_order_is_not_mains():
    var, real = SFT_DIR / "real-onetool.jsonl", SFT_DIR / "real.jsonl"
    if not (var.exists() and real.exists()):
        pytest.skip("the SFT sets are not here")
    ids = {}
    for name, path in (("main", real), ("onetool", var)):
        rows = [json.loads(x)["id"] for x in path.read_text(encoding="utf-8").split("\n") if x]
        ids[name] = [rows[i] for i in T.plan_steps(len(rows), 1, 2, 16)[0].rows]
    again = [json.loads(x)["id"] for x in var.read_text(encoding="utf-8").split("\n") if x]
    assert [again[i] for i in T.plan_steps(len(again), 1, 2, 16)[0].rows] == ids["onetool"]
    assert ids["onetool"] != ids["main"]


def test_onetool_scores_no_untrained_model_and_trains_no_control(tmp_path):
    with pytest.raises(SystemExit, match="main run"):
        R.run_score(tmp_path, True, None, None, profile="onetool")
    with pytest.raises(SystemExit, match="real arm only"):
        R.run_train(tmp_path, True, "control", 1, False, None, None, False, profile="onetool")
    with pytest.raises(SystemExit, match="strict reading only"):
        R.run_report(tmp_path, True, "onetool", "tagfree")


def test_the_cli_takes_the_variant_and_the_main_tree():
    p = build_parser()
    assert p.parse_args(["sft-data", "--variant", "onetool"]).variant == "onetool"
    assert p.parse_args(["sft-data"]).variant is None
    a = p.parse_args(["sft-report", "--profile", "onetool", "--main-from", "x/y"])
    assert (a.profile, a.main_from) == ("onetool", "x/y")
    for cmd in (["sft-check"], ["sft-pair", "--seed", "1"], ["sft-train", "--arm", "real",
                                                             "--seed", "2"]):
        assert p.parse_args([*cmd, "--profile", "onetool"]).profile == "onetool"


# -- the report's arithmetic -------------------------------------------------------------------


def test_arm_seeds(tmp_path):
    for name, files in (("real-seed1", ("train.json", "score.json")),
                        ("real-seed3", ("train.json", "score.json")),
                        ("real-seed2", ("train.json",)),  # not scored yet
                        ("real-seed1-repeat", ("train.json", "score.json")),
                        ("control-seed1", ("train.json", "score.json"))):
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_text("{}")
    assert O.arm_seeds(tmp_path) == ([1, 3], ["real-seed2"])
    assert O.arm_seeds(tmp_path, "control") == ([1], [])
    assert O.arm_seeds(tmp_path / "missing") == ([], [])


def test_the_rule_reads_the_interval_upward_only():
    def p(lo, hi):
        return E.Paired("onetool real", "main real", 540, E.Fraction(1, 2), E.Fraction(1, 3),
                        lo, hi)

    assert O.verdict(p(0.01, 0.2)) == "confirmed"
    for lo, hi in ((-0.01, 0.2), (0.0, 0.2), (-0.3, -0.1)):  # a drop excludes 0 too: not up
        assert O.verdict(p(lo, hi)) == "not confirmed"
    line = O.primary_line(p(0.1234, 0.2104))
    assert line == ("primary: right on BFCL's one-function call items (simple + "
                    "parallel, 540), onetool real - main real: +16.7 points [+12.3, +21.0], "
                    "excludes 0 upward: yes -> confirmed")
    assert O.primary_line(p(0.1234, 0.2104), [1, 2, 3], [1, 2, 3]) == line


def test_the_verdict_is_withheld_unless_both_sides_have_seeds_1_to_3():
    # T9h review S2: a deadline-skipped seed must not yield a verdict
    p = E.Paired("onetool real", "main real", 540, E.Fraction(1, 2), E.Fraction(1, 3),
                 0.1234, 0.2104)
    for seeds, main_seeds in (([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 3]), ([2, 3], [2, 3])):
        line = O.primary_line(p, seeds, main_seeds)
        assert "confirmed" not in line and "excludes 0 upward" not in line
        assert line.endswith(f"verdict withheld: the rule needs seeds 1-3 on both sides, and "
                             f"onetool has {', '.join(map(str, seeds))}, main real "
                             f"{', '.join(map(str, main_seeds))}")


def test_check_inputs_names_every_run_off_its_pin():
    # T9h review S1: the report stops unless every run trained on its side's pinned set
    one, main = "a" * 64, "b" * 64
    assert O.check_inputs({"real-seed1": one}, {"real-seed1": main}, one, main) == []
    swapped = O.check_inputs({"real-seed1": main, "real-seed2": main},
                             {"real-seed1": one}, one, main)
    assert len(swapped) == 3 and swapped[0].startswith("onetool real-seed1")
    assert swapped[2].startswith("main real-seed1")
    assert O.check_inputs({"real-seed1": None}, {}, one, main) == [
        f"onetool real-seed1: trained on ?, not the variant's {one[:16]}"]


def test_the_report_refuses_main_runs_as_onetool_runs(monkeypatch, tmp_path):
    # T9h review S1: --from pointed at the main run's tree prints nothing and exits
    main = ROOT / O.MAIN_FROM
    if not (main / "real-seed1" / "train.json").exists():
        pytest.skip("the main run's tree is not here")
    with pytest.raises(SystemExit, match="did not train on its pinned set"):
        R.run_report(tmp_path, False, "onetool", "strict", main, main)
    assert not (tmp_path / "outputs").exists()


def test_compare_is_seed_means_paired_by_task():
    one = [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
    main = [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
    p = O.compare(one, main)
    assert p.n == 4 and p.share_a == E.Fraction(8, 12) and p.share_b == E.Fraction(7, 12)
    assert p.diff == pytest.approx(1 / 12)
    d = [x - y for x, y in zip(E.seed_mean(one), E.seed_mean(main), strict=True)]
    assert (p.lo, p.hi) == E.bootstrap_interval(d)


def test_mean_count():
    assert O.mean_count([["right", "no call", "right"]], "right") == "2"
    assert O.mean_count([["right", "right"], ["no call", "right"], ["right", "right"]],
                        "right") == "1.7"
    assert O.differing({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}, ("a", "b", "c")) == ["b", "c"]


def test_the_report_sets():
    from types import SimpleNamespace

    items = {c: [SimpleNamespace(id=f"{c}-{i}") for i in range(2)] for c in
             ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")}
    test = [{"id": "o1", "kind": "single"}, {"id": "o2", "kind": "no_fit"}]
    from tinytools import tasks

    test[1]["kind"] = tasks.NO_FIT
    s = O._sets(test, items)
    assert s["one-function"] == ("bfcl", ["simple-0", "simple-1", "parallel-0", "parallel-1"])
    assert s["multi-function"][1] == ["multiple-0", "multiple-1", "parallel_multiple-0",
                                      "parallel_multiple-1"]
    assert s["ours, call tasks"] == ("ours", ["o1"]) and s["ours, no fit"] == ("ours", ["o2"])
    assert s["BFCL irrelevance"][1] == ["irrelevance-0", "irrelevance-1"]


# -- onetool.toml ------------------------------------------------------------------------------


def _exp():
    return lr.parse_experiment((ROOT / "experiments" / "onetool.toml").read_text(), "onetool")


def test_the_onetool_experiment_runs_the_real_arm_three_times_and_the_report():
    e, parser = _exp(), build_parser()
    stages = []
    for c in e.commands:
        if c.startswith(("echo", "export")):
            stages.append(c.split()[0])
            continue
        words = shlex.split(c.replace('"$TINYTOOLS_DEADLINE_EPOCH"', "1700000000"))
        args = parser.parse_args(words[3:])
        stages.append((words[3], getattr(args, "profile", None), getattr(args, "seed", None)))
        assert "--quick" not in c
    assert stages == [
        "export", ("toolace-convert", None, None), ("sft-data", None, None),
        ("sft-check", "onetool", None), "echo",
        ("sft-pair", "onetool", 1), "echo", ("sft-pair", "onetool", 2), "echo",
        ("sft-pair", "onetool", 3), "echo", ("sft-report", "onetool", None)]
    assert "sft-data --variant onetool" in e.commands[2]
    assert not any("--untrained" in c or "--repeat" in c or "control" in c for c in e.commands)
    assert (e.post, e.instance_type, e.cap_usd, e.setup_record) == (
        9, "gpu_1x_a100_sxm4", 8.0, "outputs/onetool/setup.json")
    assert e.expected_hours < e.max_hours


def test_the_onetool_experiment_fetches_its_runs_and_captures_only():
    e = _exp()
    for f in ["outputs/onetool/checks.json", "outputs/onetool/setup.json",
              "outputs/onetool/real-seed1/train.json", "outputs/onetool/real-seed2/score.json",
              "outputs/onetool/real-seed3/losses.jsonl",
              f"outputs/onetool/real-seed1/{R.OURS_FILE}", f"outputs/onetool/real-seed2/{R.BFCL_FILE}",
              f"outputs/onetool/real-seed3/{R.SAMPLE_FILE}",
              "outputs/onetool/real-seed1/adapter.safetensors",
              "outputs/sft-onetool.txt", "outputs/sft-check-onetool.txt",
              "outputs/sft-data-onetool.txt", "outputs/sft-pair-seed3-onetool.txt",
              "outputs/toolace-convert.txt"]:
        assert any(Path(f).match(p) for p in e.fetch), f
    for f in ["outputs/onetool/b0-adapter.safetensors", "outputs/sft.txt", "outputs/sft-data.txt",
              "outputs/main/untrained/score.json", "outputs/sft-check.txt",
              "outputs/sft-base.txt", "outputs/base/real-seed1/train.json"]:
        assert not any(Path(f).match(p) for p in e.fetch), f


TYPES = {"gpu_1x_a100_sxm4": {"instance_type": {"price_cents_per_hour": 199},
                              "regions_with_capacity_available": [{"name": "us-east-1"}]}}


def test_the_onetool_experiment_fits_both_caps_before_its_launch():
    """The pre-launch check against the ledger as it stood before any onetool run (so the
    test stays true after the run is recorded); and whatever onetool has spent, its cap."""
    e = _exp()
    ledger = lr.read_ledger(ROOT / "runs" / "ledger.csv")
    before = [r for r in ledger if r["experiment"] != "onetool"]
    est = lr.make_estimate(e, TYPES, before)
    assert est.worst == 7.96 and est.spent == 0 and est.post == 9
    assert est.post_spent == pytest.approx(sum(float(r["cost_usd"]) for r in before
                                               if r["post"] == "9"))
    assert est.within_cap and est.expected <= est.worst
    assert sum(float(r["cost_usd"]) for r in ledger if r["experiment"] == "onetool") <= e.cap_usd
