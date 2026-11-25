"""Functions offered per BFCL item, from the pinned bfcl-eval data files."""

from collections import Counter

import pytest

from tinytools import bfclfuncs as bf


def test_functions_per_item_is_pure_counting():
    entries = [{"function": [{}]}, {"function": [{}, {}]}, {"function": [{}]}]
    assert bf.functions_per_item(entries) == Counter({1: 2, 2: 1})
    assert bf.functions_per_item([]) == Counter()


def test_count_pct():
    assert bf.count_pct(79, 200) == "79 (39.5%)"
    assert bf.count_pct(240, 240) == "240 (100%)"
    assert bf.count_pct(375, 500) == "375 (75%)"
    assert bf.count_pct(17, 180) == "17 (9.4%)"
    assert bf.count_pct(0, 0) == "0 (-)"


def test_version_prefix_matches_bfcl_eval():
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX

    assert bf.VERSION_PREFIX == VERSION_PREFIX


@pytest.mark.parametrize("category", bf.ORDER)
def test_file_order_and_dev_slice_match_the_loader(category):
    """The raw file's ids are the loader's (bfcl.category_ids), and the dev slice removed
    here is bfcl.dev_slice's, so these counts describe the items bfcl-score scores."""
    from tinytools import bfcl

    entries = bf.read_entries(bf.data_file(category))
    assert [e["id"] for e in entries] == bfcl.category_ids(category)
    c = bf.count(category)
    assert c.dev == len(bfcl.dev_slice(category))
    assert c.n == len(entries) and c.n_after == c.n - c.dev


def test_counts_on_the_pinned_files():
    """The plan's expected counts (bfcl-eval 2025.12.17). If the package changes, this is
    where it shows."""
    got = {c: bf.count(c) for c in bf.ORDER}
    for c in bf.ORDER:
        assert got[c].sha256 == bf.PINNED_SHA256[c] and got[c].pinned, c
    assert got["irrelevance"].every == Counter({1: 240})
    assert got["simple"].every == Counter({1: 400}) and got["parallel"].every == Counter({1: 200})
    assert got["multiple"].every == Counter({2: 79, 3: 85, 4: 36})
    assert got["parallel_multiple"].every == Counter({2: 100, 3: 80, 4: 20})
    # after the dev slice, every category (the items post 10 scores)
    assert got["simple"].after_dev == Counter({1: 360})
    assert got["parallel"].after_dev == Counter({1: 180})
    assert got["multiple"].after_dev == Counter({2: 69, 3: 79, 4: 32})
    assert got["parallel_multiple"].after_dev == Counter({2: 93, 3: 70, 4: 17})
    assert got["irrelevance"].after_dev == Counter({1: 216})
    assert got["simple"].n_after + got["parallel"].n_after == 540


def test_another_file_is_not_pinned(tmp_path, monkeypatch):
    """A different file (another bfcl-eval version) reports pinned NO, not an error."""
    f = tmp_path / "BFCL_v4_irrelevance.json"
    f.write_text('{"id": "irrelevance_0", "function": [{}]}\n')
    monkeypatch.setattr(bf, "data_file", lambda category: f)
    c = bf.count("irrelevance")
    assert not c.pinned and c.every == Counter({1: 1})


def test_cli_report_fits_the_width(tmp_path, monkeypatch, capsys):
    from tinytools import cli

    monkeypatch.setattr(cli, "OUTPUTS", tmp_path)
    cli.main(["bfcl-functions"])
    out = capsys.readouterr().out
    assert (tmp_path / "bfcl-functions.txt").read_text() == out
    assert "  irrelevance             240 (100%)      24         216 (100%)" in out
    assert "every file the pinned one          yes (5 of 5)" in out
    assert out.count(" pinned yes") == 5
    assert all(len(line) <= 74 and line == line.rstrip() for line in out.splitlines())
