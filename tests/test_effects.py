"""The comparisons' arithmetic: paired bootstrap, exact shares, the effects, right function
vs chance, groups of 8 and the transition table, and the printed formats."""

from fractions import Fraction

import numpy as np
import pytest

from tinytools import effects as E
from tinytools.classify import (CALLED_ANYWAY, DECLINED, NO_CALL, RIGHT, UNPARSEABLE,
                                WRONG_ARGUMENTS, WRONG_COUNT, WRONG_FUNCTION)


def test_bootstrap_is_seeded_and_centred():
    d = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1] * 30
    a = E.bootstrap_interval(d)
    assert a == E.bootstrap_interval(d)  # a fresh generator per call
    assert a != E.bootstrap_interval(d, seed=1)
    lo, hi = a
    assert lo < np.mean(d) < hi
    assert hi - lo == pytest.approx(2 * 1.96 * np.std(d) / np.sqrt(len(d)), rel=0.15)
    assert E.bootstrap_interval([0.25] * 40) == (0.25, 0.25)
    with pytest.raises(ValueError):
        E.bootstrap_interval([])


def test_bootstrap_resamples_tasks_with_replacement():
    # by hand: numpy's generator, 10,000 x n indices, percentiles 2.5 and 97.5
    d = np.array([0.0, 1.0, -1.0, 0.5, 1 / 3])
    idx = np.random.default_rng(0).integers(0, 5, size=(10_000, 5))
    want = tuple(float(x) for x in np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))
    assert E.bootstrap_interval(d) == want


def test_excludes_zero():
    assert E.excludes_zero(0.01, 0.2) and E.excludes_zero(-0.3, -0.001)
    assert not E.excludes_zero(-0.01, 0.2) and not E.excludes_zero(0.0, 0.1)


def test_seed_means_and_exact_shares():
    assert E.seed_mean([[1, 0, 1], [0, 0, 1], [1, 0, 0]]) == pytest.approx([2 / 3, 0, 2 / 3])
    with pytest.raises(ValueError):
        E.seed_mean([[1, 0], [1]])
    assert E.exact_mean([2 / 3, 0, 2 / 3], 3) == Fraction(4, 9)
    assert E.exact_mean([1.0, 0.0, 0.0, 0.0], 1) == Fraction(1, 4)
    with pytest.raises(ValueError):
        E.exact_mean([0.5], 3)


def test_paired_difference_is_exact():
    a = [2 / 3, 1 / 3, 1.0, 0.0]
    b = [0.0, 0.0, 1.0, 1.0]
    p = E.paired("SFT", a, "untrained", b, 3, 1)
    assert (p.share_a, p.share_b, p.diff, p.n) == (Fraction(1, 2), Fraction(1, 2), 0, 4)
    with pytest.raises(ValueError):
        E.paired("a", [1.0], "b", [1.0, 0.0])


def test_an_effect_gets_three_paired_differences():
    untrained = [NO_CALL, NO_CALL, RIGHT, WRONG_ARGUMENTS]
    sft = [[RIGHT, NO_CALL, RIGHT, RIGHT], [RIGHT, RIGHT, RIGHT, RIGHT]]
    control = [[NO_CALL, NO_CALL, UNPARSEABLE, RIGHT], [NO_CALL, NO_CALL, NO_CALL, RIGHT]]
    r = E.effect_result(E.CALLING, E.OURS_CALL, untrained, sft, control)
    assert r.shares == {"untrained": Fraction(1, 2), "SFT": Fraction(1, 8),
                        "control": Fraction(3, 4)}
    assert [(p.a, p.b) for p in r.diffs] == [("SFT", "untrained"), ("control", "untrained"),
                                            ("SFT", "control")]
    assert [p.diff for p in r.diffs] == [Fraction(-3, 8), Fraction(1, 4), Fraction(-5, 8)]
    assert r.primary == 1
    assert E.effect_result(E.CHOOSING, E.OURS_CALL, untrained, sft, control).primary is None


def test_effects_are_the_plans():
    assert E.CALLING.outcomes == {NO_CALL, UNPARSEABLE}
    assert E.FILLING.outcomes == {WRONG_ARGUMENTS}
    assert E.DECLINING.outcomes == {CALLED_ANYWAY}
    assert E.CHOOSING.outcomes == {WRONG_FUNCTION, WRONG_COUNT} and E.CHOOSING.harm_check
    assert E.RIGHT_SHARE.outcomes == {RIGHT, DECLINED}
    assert sorted(E.PRIMARY.values()) == [1, 2, 3, 4, 5]  # five primary comparisons
    for (eff, key) in E.PRIMARY:
        assert eff is E.RIGHT_SHARE or key in E.EFFECT_SETS[eff]


def test_right_function_against_chance():
    outcomes = [RIGHT, WRONG_FUNCTION, NO_CALL, WRONG_ARGUMENTS, WRONG_COUNT, UNPARSEABLE]
    ks = [3, 4, 5, 2, 3, 4]
    hits, n, chance = E.right_function(outcomes, ks)
    assert (hits, n) == (3, 4)  # no call and unparseable did not decode to calls
    assert chance == pytest.approx(1 / 3 + 1 / 4 + 1 / 2 + 1 / 3)


def test_groups_and_transitions():
    assert [E.group(r, 8) for r in (0, 1, 7, 8)] == ["all wrong", "mixed", "mixed", "all right"]
    with pytest.raises(ValueError):
        E.group(9, 8)
    t = E.transitions([0, 3, 8, 0, 4], [2, 8, 8, 0, 0], 8)
    assert t[("all wrong", "mixed")] == 1 and t[("mixed", "all right")] == 1
    assert t[("all right", "all right")] == 1 and t[("all wrong", "all wrong")] == 1
    assert t[("mixed", "all wrong")] == 1 and sum(t.values()) == 5
    both = E.add_tables([t, t])
    assert both[("mixed", "all right")] == 2 and sum(both.values()) == 10
    v = E.movement_values([0, 3, 8], [[2, 8, 8], [0, 8, 8]], 8, ("all wrong", "mixed"))
    assert v == [0.5, 0.0, 0.0]
    assert E.group_values([0, 3, 8], 8, "mixed") == [0.0, 1.0, 0.0]


def test_printed_formats_round_half_away_from_zero():
    assert E.points(Fraction(1, 8)) == "+12.5"
    assert E.points(Fraction(-1, 800)) == "-0.1"  # -0.125 points
    assert E.points(Fraction(1, 2000)) == "+0.1"  # 0.05 points, half up
    assert E.points(-0.0004) == "0.0" and E.points(0) == "0.0"
    assert E.percent(Fraction(49, 400)) == "12.3%"  # 12.25%, half up
    assert E.percent(Fraction(1, 3)) == "33.3%"
    assert E.interval(-0.0213, 0.0451) == "[-2.1, +4.5]"
