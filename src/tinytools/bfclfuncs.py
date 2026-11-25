"""Functions offered per BFCL item, counted from the pinned bfcl-eval package's data files.

Post 10 asks whether a model reads the one tool it is offered, so it needs to know how
many functions each BFCL prompt offers. This reads the five single-turn categories'
JSON-lines files straight from the installed bfcl-eval (no model, no bfcl-eval import),
counts ``len(entry["function"])`` per item, and does it twice: every item in the file,
and after the dev slice (:func:`tinytools.bfcl.dev_ids`, the same slice every published
number leaves out). The report prints each file's sha256 and whether it is the pinned
one (:data:`PINNED_SHA256`, bfcl-eval 2025.12.17), so the counts are tied to the bytes
they came from, and another package version shows as "pinned NO".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# bfcl-eval's file names: f"{VERSION_PREFIX}_{name}.json" (bfcl_eval.constants
# .category_mapping.VERSION_PREFIX in 2025.12.17; a test checks it).
VERSION_PREFIX = "BFCL_v4"
# Our names (tinytools.bfcl.CATEGORIES) in the order the report prints them.
ORDER = ("simple", "parallel", "multiple", "parallel_multiple", "irrelevance")
# The five files' sha256 in bfcl-eval 2025.12.17 (the leaderboard's version).
PINNED_SHA256 = {
    "simple": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",
    "parallel": "19f51a82eff42e5d62541aa500115a056eb78f437c2ba1f10415fd7c8e5dda84",
    "multiple": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
    "parallel_multiple": "8863ea8433239f55c5f016154cf0830853c89f693c6ea270396a2fa121960579",
    "irrelevance": "2b6ed4c2e992cdcf5f1678a701851f944bef7550ee026ed1ddb89efed5be01a6",
}


def data_dir() -> Path:
    """bfcl-eval's data directory, found without importing the package."""
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or spec.origin is None:
        raise RuntimeError("bfcl-eval is not installed (uv sync)")
    return Path(spec.origin).parent / "data"


def data_file(category: str) -> Path:
    from .bfcl import CATEGORIES

    return data_dir() / f"{VERSION_PREFIX}_{CATEGORIES[category]}.json"


def read_entries(path: Path) -> list[dict]:
    """A bfcl-eval data file: one JSON object per line."""
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def functions_per_item(entries: list[dict]) -> Counter:
    """How many items offer 1, 2, ... functions. Pure."""
    return Counter(len(e["function"]) for e in entries)


@dataclass(frozen=True)
class CategoryCount:
    category: str
    path: Path
    sha256: str
    every: Counter  # functions offered -> items, every item in the file
    after_dev: Counter  # the same, the dev slice removed
    dev: int  # items in the dev slice

    @property
    def pinned(self) -> bool:
        return self.sha256 == PINNED_SHA256[self.category]

    @property
    def n(self) -> int:
        return sum(self.every.values())

    @property
    def n_after(self) -> int:
        return sum(self.after_dev.values())


def count(category: str) -> CategoryCount:
    from .bfcl import dev_ids

    path = data_file(category)
    entries = read_entries(path)
    dev = set(dev_ids([e["id"] for e in entries], category))
    return CategoryCount(
        category=category, path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        every=functions_per_item(entries),
        after_dev=functions_per_item([e for e in entries if e["id"] not in dev]),
        dev=len(dev))


def count_pct(n: int, total: int) -> str:
    """``"79 (39.5%)"``; whole percentages without the decimal (``"240 (100%)"``). Pure."""
    if total <= 0:
        return f"{n} (-)"
    p = 100.0 * n / total
    text = f"{p:.1f}"
    return f"{n} ({text[:-2] if text.endswith('.0') else text}%)"
