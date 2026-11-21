"""Milestone 0's deadline (review round 2, R2-5): the estimate that decides whether a model
can start, and the run loop stopping at the deadline or on SIGTERM with the rule printed
from what was measured. Scoring is stubbed: no model loads."""

import json
import os
import signal
import time

import pytest

from tinytools import milestone0 as m0

CAP = 512


def timing(model, wall=100.0, passes=524, gen=90.0, greedy_s=20.0, steps=10_000):
    return m0.Timing(model, wall, passes, gen, greedy_s, steps)


def test_item_passes():
    assert m0.item_passes(m0.I3, 200, 124, m0.I15) == 524
    assert m0.item_passes(m0.I15, 200, 124, m0.I15) == 724  # the greedy pass run twice


def test_no_measurement_no_estimate():
    assert m0.estimate_seconds(m0.I15, 724, [], CAP) is None


def test_same_kind_uses_the_slowest_rate_times_the_safety_factor():
    ts = [timing(m0.I15, wall=724.0, passes=724), timing(m0.I3, wall=1048.0, passes=524)]
    assert m0.estimate_seconds(m0.I05, 524, ts, CAP) == pytest.approx(1.5 * 524 * 2.0)


def test_a_base_model_after_instruct_ones_is_costed_at_the_token_cap():
    ts = [timing(m0.I15, wall=100.0, gen=90.0, greedy_s=20.0, steps=1000),  # 20 ms/step
          timing(m0.I3, wall=130.0, gen=100.0, greedy_s=25.0, steps=1000)]  # 25 ms/step
    # 524 items x 512 steps x 25 ms, plus the largest loading/checking time (30 s), x 1.5
    assert m0.estimate_seconds(m0.B15, 524, ts, CAP) == pytest.approx(
        1.5 * (524 * 512 * 0.025 + 30.0))
    # once a base model is measured, its own rate is used
    tb = timing(m0.B15, wall=5000.0, passes=524)
    assert m0.estimate_seconds(m0.B3, 524, [*ts, tb], CAP) == pytest.approx(1.5 * 5000.0)


def test_batch_count():
    # 200 tasks at 64: 4 greedy batches; 8 per batch sampled: 25; BFCL 40/20/20/20/24: 5
    assert m0.batch_count(200, [40, 20, 20, 20, 24], 64, 8, again=False) == 34
    assert m0.batch_count(200, [40, 20, 20, 20, 24], 64, 8, again=True) == 38
    # batch 1: one batch per item-pass, as item_passes counts them
    assert m0.batch_count(200, [124], 1, 8, again=False) == m0.item_passes(
        m0.I3, 200, 124, m0.I15)


def test_a_base_model_is_costed_per_batch_with_the_batched_engine():
    ts = [timing(m0.I15, wall=100.0, gen=90.0, greedy_s=20.0, steps=1000)]  # 20 ms/step
    assert m0.estimate_seconds(m0.B15, 524, ts, CAP, batches=34) == pytest.approx(
        1.5 * (34 * 512 * 0.020 + 10.0))


def test_fits():
    assert m0.fits(0, None, 10**9)  # no deadline
    assert m0.fits(0, 1000, None)  # nothing measured: the first model starts
    assert not m0.fits(900, 1000, None)  # ...unless the margin is already gone
    assert m0.fits(0, 1000, 880) and not m0.fits(0, 1000, 881)  # 120 s margin


# -- the loop, with scoring stubbed --------------------------------------------------------


def measured(model, ours_right):
    return m0.Measured(model, ours_right, 200, 70, 100, 25, 160, 90, 160)


def stub(monkeypatch, seconds_per_model, on_score=None):
    """_score_model returns at once with a Timing that says it took ``seconds_per_model``
    (I15 and I3 fail (a); bases would qualify)."""
    scored = []

    def score(model, s, device, dev_tasks, setup, out_dir, repeat, batch):
        scored.append(model)
        if on_score:
            on_score(model)
        right = 190 if model.kind == "instruct" else 100
        return m0.ModelResult(model, [], [], [], measured(model, right),
                              timing(model, wall=seconds_per_model, passes=524))

    monkeypatch.setattr(m0, "_score_model", score)
    monkeypatch.setattr(m0, "_print_model", lambda rep, r: None)
    monkeypatch.setattr(m0, "_print_repeat", lambda rep, repeat: None)
    monkeypatch.setattr(m0, "_probe", lambda rep, s, device, root: [])
    monkeypatch.setattr(m0, "_versions", lambda: {"torch": "x"})
    return scored


def test_the_deadline_stops_before_a_model_that_cant_finish(monkeypatch, tmp_path, capsys):
    scored = stub(monkeypatch, seconds_per_model=600.0)
    # 19 min left (the stubs take no real time): 1.5B instruct starts unestimated; 3B
    # instruct needs 1.5 x 400 item-passes x 600/524 s = ~11.5 min, + 2 min margin: starts;
    # base 1.5B is costed at the token cap: 1.5 x (400 x 512 x 2 ms + 510 s) = ~23 min: not
    # (at batch size 1, every one of the 400 item-passes is its own batch)
    deadline = time.time() + 19 * 60
    code = m0.run(False, None, tmp_path, base_when_needed=True, deadline=deadline, gen_batch=1)
    out = capsys.readouterr().out
    assert scored == [m0.I15, m0.I3]
    assert code == 1  # the rule reaches base 1.5B, unscored
    rec = json.loads((tmp_path / "outputs" / "milestone0-setup.json").read_text())
    assert rec["skipped"]["Qwen/Qwen2.5-1.5B"].startswith("not started: the deadline (needs")
    assert rec["stopped"].startswith("the deadline: 1.5B base would need")
    assert "verdict" in out and "incomplete" in out
    assert "Scoring stopped early, by the deadline" in out
    assert rec["verdict"].startswith("no verdict: Qwen/Qwen2.5-1.5B was not scored")


def test_no_deadline_scores_everything(monkeypatch, tmp_path, capsys):
    scored = stub(monkeypatch, seconds_per_model=600.0)
    assert m0.run(False, None, tmp_path, base_when_needed=True) == 0
    assert scored == [m0.I15, m0.I3, m0.B15, m0.I05]  # base 1.5B qualifies: 3B base skipped
    assert "pick Qwen/Qwen2.5-1.5B" in capsys.readouterr().out


def test_sigterm_prints_the_rule_with_what_was_measured(monkeypatch, tmp_path, capsys):
    def on_score(model):
        if model == m0.I3:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # interrupted by the handler

    scored = stub(monkeypatch, seconds_per_model=10.0, on_score=on_score)
    before = signal.getsignal(signal.SIGTERM)
    t0 = time.time()
    code = m0.run(False, None, tmp_path, base_when_needed=True)
    assert time.time() - t0 < 4
    assert code == 128 + signal.SIGTERM
    assert scored == [m0.I15, m0.I3]
    out = capsys.readouterr().out
    assert "The rule (fixed before these numbers)" in out and "incomplete" in out
    assert "Scoring stopped early, by SIGTERM" in out
    rec = json.loads((tmp_path / "outputs" / "milestone0-setup.json").read_text())
    assert list(rec["measured"]) == ["Qwen/Qwen2.5-1.5B-Instruct"]
    assert rec["skipped"]["Qwen/Qwen2.5-3B-Instruct"] == "stopped by SIGTERM while being scored"
    assert rec["skipped"]["Qwen/Qwen2.5-1.5B"] == "not started: SIGTERM"
    assert signal.getsignal(signal.SIGTERM) == before  # restored
