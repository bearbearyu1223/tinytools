"""engine-check's comparison arithmetic (no model)."""

import pytest

from tinytools import engine_check as ec


def test_compare_counts_identical_differing_and_changed():
    c = ec.compare(["a", "b", "c"], ["x", "y", "z"], ["x", "Y", "Z"],
                   ["right", "right", "no call"], ["right", "right", "right"])
    assert (c.n, c.identical, c.differ, c.outcome_changed) == (3, 1, ["b", "c"], ["c"])


def test_compare_needs_the_same_items():
    with pytest.raises(ValueError):
        ec.compare(["a"], ["x"], [], ["right"], [])


def test_divergence():
    assert ec.divergence("abcd", "abXd") == 2
    assert ec.divergence("ab", "abc") == 2


def test_same_count_is_position_by_position():
    assert ec.same_count([["a", "b"], ["c"]], [["a", "c"], ["c"]]) == 2
    with pytest.raises(ValueError):
        ec.same_count([["a"]], [["a"], ["b"]])


def test_speedup():
    assert ec.speedup(30.0, 10.0) == "3.0x"
    assert ec.speedup(1.0, 0.0) == "-"


def test_clip_fits_the_report():
    line = "  " + f"  {'one at a time':<13}: ..." + ec._clip("a\nb" * 50)
    assert len(line) <= 74


def test_sizes_and_capture_names():
    assert ec.sizes_for(False, None, None, None) == ec.FULL
    assert ec.tag_for(False, ec.FULL) == "engine-check"
    assert ec.tag_for(True, ec.sizes_for(True, None, None, None)) == "engine-check-quick"
    half = ec.sizes_for(False, 10, 4, 10)
    assert ec.tag_for(False, half) == "engine-check-10-4-10"
