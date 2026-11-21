"""BFCL's dev slice and splits: seeded, ~10% per category, held out of eval."""

import pytest

from tinytools import bfcl


def test_dev_ids_is_seeded_ten_percent_in_order():
    ids = [f"c_{i}" for i in range(240)]
    dev = bfcl.dev_ids(ids, "c")
    assert len(dev) == 24 and dev == [i for i in ids if i in set(dev)]
    assert dev == bfcl.dev_ids(list(ids), "c")  # same seed, same slice
    assert dev != bfcl.dev_ids(ids, "c", seed=1)
    assert dev != bfcl.dev_ids(ids, "other")  # each category draws its own
    assert bfcl.dev_ids(ids[:4], "c") and len(bfcl.dev_ids(ids[:4], "c")) == 1  # at least one
    assert bfcl.dev_ids(ids[:15], "c") == bfcl.dev_ids(ids[:15], "c") and \
        len(bfcl.dev_ids(ids[:15], "c")) == 2  # 1.5 rounds half up
    assert bfcl.dev_ids([], "c") == []


def test_dev_ids_does_not_depend_on_the_order_given():
    ids = [f"c_{i}" for i in range(100)]
    assert set(bfcl.dev_ids(ids, "c")) == set(bfcl.dev_ids(list(reversed(ids)), "c"))


def test_splits_partition_the_ids():
    ids = [f"c_{i}" for i in range(50)]
    dev = bfcl.dev_ids(ids, "c")
    ev = bfcl.in_split(ids, dev, "eval")
    assert bfcl.in_split(ids, dev, "dev") == dev
    assert not set(ev) & set(dev) and len(ev) + len(dev) == 50
    assert bfcl.in_split(ids, dev, "all") == ids
    with pytest.raises(ValueError):
        bfcl.in_split(ids, dev, "test")


@pytest.mark.parametrize("category, n", [("simple", 400), ("multiple", 200), ("parallel", 200),
                                         ("parallel_multiple", 200), ("irrelevance", 240)])
def test_dev_slice_on_bfcl(category, n):
    dev = bfcl.dev_slice(category)
    assert len(dev) == n // 10
    ev = [it.id for it in bfcl.load_items(category, None, "eval")]
    assert len(ev) == n - n // 10 and not set(ev) & set(dev)
    assert [it.id for it in bfcl.load_items(category, None, "dev")] == dev
    assert [it.id for it in bfcl.load_items(category, 3, "eval")] == ev[:3]
    assert len(bfcl.load_items(category, None, "all")) == n
