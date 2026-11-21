"""The batched engine's pure parts: batch plan, stop search, fixed-point sampling, seeds."""

import random
from collections import Counter

import pytest
import torch

from tinytools import engine


def test_plan_sorts_longest_first_and_keeps_ties_in_order():
    assert engine.plan_batches([3, 9, 5, 9, 1], 1, 2) == [[1, 3], [2, 0], [4]]


def test_plan_keeps_an_items_samples_together():
    # batch of 16 rows at 8 samples: two items per batch
    assert engine.plan_batches([1, 2, 3], 8, 16) == [[2, 1], [0]]
    # a batch smaller than one item's samples still takes one item
    assert engine.plan_batches([1, 2], 8, 4) == [[1], [0]]


def test_plan_covers_every_item_once():
    lengths = [random.Random(1).randrange(100) for _ in range(37)]
    plan = engine.plan_batches(lengths, 1, 8)
    assert sorted(i for b in plan for i in b) == list(range(37))
    assert all(len(b) == 8 for b in plan[:-1])


def test_plan_rejects_nonsense():
    with pytest.raises(ValueError):
        engine.plan_batches([1], 1, 0)


def test_first_stop():
    assert engine.first_stop([5, 6, 7, 6], {6}) == 1
    assert engine.first_stop([5, 7], {6}) is None
    assert engine.first_stop([], {6}) is None


def test_fixed_point_totals():
    p = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    run = engine.fixed_point(p)
    assert run.dtype == torch.int64
    one = 2**engine.SCALE_BITS
    assert run.tolist() == [[one // 2, 3 * one // 4, one, one]]


def test_pick_token_boundaries():
    # increments 2, 0, 3: draws 0-1 -> token 0, 2-4 -> token 2; token 1 never
    run = torch.tensor([[2, 2, 5]] * 5, dtype=torch.int64)
    got = engine.pick_token(run, torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64))
    assert got.tolist() == [0, 0, 2, 2, 2]


def test_pick_token_is_exact_over_every_draw():
    p = torch.tensor([[0.1, 0.0, 0.6, 0.3]])
    run = engine.fixed_point(p)
    # scale down so every draw can be enumerated: the same arithmetic at 2^-8
    small = torch.floor(p * 256).to(torch.int64).cumsum(-1)
    total = int(small[0, -1])
    counts = Counter(engine.pick_token(small.expand(total, -1).contiguous(),
                                       torch.arange(total, dtype=torch.int64)).tolist())
    incr = torch.diff(small[0], prepend=torch.tensor([0])).tolist()
    assert dict(counts) == {i: q for i, q in enumerate(incr) if q}
    assert 1 not in counts and run[0, 1] == run[0, 0]  # zero probability: never drawn


def test_row_seed_names_the_reply():
    assert engine.row_seed(0, "test-0001", 3) == "0:test-0001:3"
    a = random.Random(engine.row_seed(0, "x", 0)).randrange(10**9)
    b = random.Random(engine.row_seed(0, "x", 0)).randrange(10**9)
    c = random.Random(engine.row_seed(0, "x", 1)).randrange(10**9)
    assert a == b and a != c


def test_default_batch_size():
    assert engine.default_batch_size("cuda") == 64
    assert engine.default_batch_size("xpu") == engine.DEFAULT_BATCH["cpu"]


def test_scale_draw_covers_the_range_evenly():
    top = 2**engine.DRAW_BITS
    assert engine.scale_draw(0, 7) == 0
    assert engine.scale_draw(top - 1, 7) == 6
    # 2^96 inputs spread over 7 values: each gets floor or ceil of 2^96 / 7
    edges = [-(-k * top // 7) for k in range(8)]  # first input landing on value k
    sizes = [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    assert max(sizes) - min(sizes) <= 1
    assert all(engine.scale_draw(e, 7) == k for k, e in enumerate(edges[:-1]))


def test_a_replys_draws_ignore_the_totals():
    # the generator advances by DRAW_BITS per token whatever the total: the second
    # draw is the same after a first draw against a total just below or above 2^40
    def second(total):
        rng = random.Random("0:x:0")
        engine.scale_draw(rng.getrandbits(engine.DRAW_BITS), total)
        return rng.getrandbits(engine.DRAW_BITS)
    assert second(2**40 - 5) == second(2**40 + 5)
