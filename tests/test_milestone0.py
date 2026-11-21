"""Milestone 0's model-free pieces: the rule and its verdict, the run-twice diff, the
probe's token windows and epoch arithmetic; plus the stop ids and the LoRA port."""

import pytest
import torch

from tinytools import milestone0 as m0
from tinytools.bfcl import stop_ids
from tinytools.lora import ATTENTION_MLP, inject_lora, trainable_parameters


def measured(model, ours=(130, 200), bfcl=(70, 100), mixed=(25, 200)):
    return m0.Measured(model, *ours, *bfcl, *mixed)


def test_thresholds_are_inclusive_and_exact():
    assert m0.at_most(140, 200, 70) and not m0.at_most(141, 200, 70)
    assert m0.at_most(80, 100, 80) and not m0.at_most(81, 100, 80)
    assert m0.at_least(20, 200, 10) and not m0.at_least(19, 200, 10)
    assert not m0.at_most(0, 0, 70) and not m0.at_least(0, 0, 10)  # nothing scored: not met


def test_each_criterion():
    m = measured(m0.I15)
    assert (m.a, m.b, m.c, m.qualifies) == (True, True, True, True)
    assert not measured(m0.I15, ours=(141, 200)).a
    assert not measured(m0.I15, bfcl=(81, 100)).b
    assert not measured(m0.I15, mixed=(19, 200)).c
    assert m0.failed_criteria(measured(m0.I15, ours=(150, 200), mixed=(5, 200))) == ["a", "c"]


def test_the_smaller_qualifying_instruct_is_picked():
    v = m0.verdict({m0.I15: measured(m0.I15), m0.I3: measured(m0.I3)})
    assert (v.outcome, v.model) == ("pick", m0.I15)
    v = m0.verdict({m0.I15: measured(m0.I15, ours=(180, 200)), m0.I3: measured(m0.I3)})
    assert (v.outcome, v.model) == ("pick", m0.I3)


def test_fallbacks_in_order_then_stop():
    fail = dict(ours=(190, 200))
    ms = {m: measured(m, **fail) for m in (m0.I15, m0.I3, m0.B3)}
    ms[m0.B15] = measured(m0.B15, mixed=(20, 200))  # exactly 10%
    v = m0.verdict(ms)
    assert (v.outcome, v.model) == ("pick", m0.B15)
    assert "fallback" in v.text
    ms[m0.B15] = measured(m0.B15, **fail)
    assert m0.verdict(ms).outcome == "stop"
    assert "bring it to the author" in m0.verdict(ms).text


def test_0p5b_never_decides_and_a_missing_model_the_rule_reaches_is_incomplete():
    assert m0.verdict({m0.I05: measured(m0.I05)}).outcome == "incomplete"
    v = m0.verdict({m0.I15: measured(m0.I15, ours=(190, 200)), m0.I05: measured(m0.I05)})
    assert v.outcome == "incomplete" and "3B-Instruct" in v.text
    # a model the rule never reaches may be missing
    assert m0.verdict({m0.I15: measured(m0.I15)}).outcome == "pick"


def test_takeaway_is_built_from_the_numbers():
    ms = {m0.I15: measured(m0.I15, ours=(150, 200)), m0.I3: measured(m0.I3)}
    t = m0.takeaway(ms, m0.verdict(ms))
    assert t == ("1.5B instruct fails (a: 75.0% right on our dev tasks, 70.0% on BFCL's dev "
                 "calls, 12.5% mixed groups); 3B instruct qualifies (65.0% right on our dev "
                 "tasks, 70.0% on BFCL's dev calls, 12.5% mixed groups).")


def test_parse_models_keeps_the_run_order():
    assert m0.parse_models("0.5B-base,1.5B-instruct") == (m0.I15, m0.B05)
    with pytest.raises(ValueError):
        m0.parse_models("7B-instruct")
    assert [m.hf_id for m in m0.RUN_ORDER][:2] == ["Qwen/Qwen2.5-1.5B-Instruct",
                                                   "Qwen/Qwen2.5-3B-Instruct"]


def test_differing():
    a = [("t1", "x"), ("t2", "y"), ("t3", "z")]
    assert m0.differing(a, [("t1", "x"), ("t2", "y "), ("t3", "z")]) == ["t2"]
    with pytest.raises(ValueError):
        m0.differing(a, a[:2])


def test_token_mask_and_last_window():
    offsets = [(0, 5), (5, 10), (10, 14), (14, 20)]
    assert m0.token_mask(offsets, [(10, 20)]) == [False, False, True, True]
    assert m0.token_mask(offsets, [(8, 20)]) == [False, False, True, True]  # straddler masked
    assert m0.last_window([1, 2, 3, 4], [False, False, True, True], 3) == ([2, 3, 4],
                                                                          [False, True, True])


def test_epoch_arithmetic():
    assert m0.epoch_steps(8504, 4) == 2126 and m0.epoch_steps(8505, 4) == 2127
    assert m0.epoch_hours(0.5, 8504, 4) == pytest.approx(2126 * 0.5 / 3600)


def test_stop_ids_add_end_of_turn_for_base_models_only():
    assert stop_ids([151645, 151643], 151645) == [151645, 151643]  # instruct: unchanged
    assert stop_ids(151643, 151645) == [151643, 151645]  # base
    assert stop_ids(None, 151645) == [151645]
    assert stop_ids([1, 2], None) == [1, 2]


def test_lora_starts_as_the_base_model_and_trains_only_adapters():
    torch.manual_seed(0)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 8)
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (torch.nn.Linear(8, 8)
                                                                  for _ in range(4))
            self.gate_proj, self.up_proj = torch.nn.Linear(8, 16), torch.nn.Linear(8, 16)
            self.down_proj = torch.nn.Linear(16, 8)

        def forward(self, ids):
            x = self.embed(ids)
            x = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
            return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    model = Block()
    ids = torch.tensor([[1, 2, 3]])
    before = model(ids)
    wrapped = inject_lora(model, ATTENTION_MLP, rank=2, alpha=4)
    assert len(wrapped) == 7
    assert torch.equal(model(ids), before)  # B starts at zero
    params = trainable_parameters(model)
    expected = 2 * (4 * (8 + 8) + 2 * (8 + 16) + (16 + 8))
    assert sum(p.numel() for p in params) == expected
    assert not model.embed.weight.requires_grad
    with pytest.raises(ValueError):
        inject_lora(Block(), ["missing_proj"], rank=2, alpha=4)


def test_base_when_needed_skips_only_what_the_rule_cannot_reach():
    ok, bad = measured(m0.I15), measured(m0.I15, ours=(190, 200))
    assert m0.skip_reason(m0.B15, {m0.I15: ok}, False) is None  # off: score everything
    assert m0.skip_reason(m0.I3, {m0.I15: ok}, True) is None  # instruct always scored
    assert m0.skip_reason(m0.B05, {}, True) == "a comparison row only"
    assert "already picked Qwen/Qwen2.5-1.5B-Instruct" in m0.skip_reason(m0.B15, {m0.I15: ok}, True)
    both_fail = {m0.I15: bad, m0.I3: measured(m0.I3, bfcl=(90, 100))}
    assert m0.skip_reason(m0.B15, both_fail, True) is None
    assert m0.skip_reason(m0.B3, {**both_fail, m0.B15: measured(m0.B15)}, True) is not None
    assert m0.skip_reason(m0.B3, {**both_fail, m0.B15: measured(m0.B15, mixed=(0, 200))},
                          True) is None
