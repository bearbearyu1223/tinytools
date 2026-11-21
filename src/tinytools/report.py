"""Fixed-width plain-text reports, printed and optionally saved to a file.

Every command prints the same shape::

    rep = Report("tinytools bfcl-score", out=Path("outputs/bfcl-score.txt"))
    rep.header()
    rep.section("Greedy, five-way split")
    rep.kv("items scored", 100)
    rep.table(["category", "right"], rows)

The layout is the one the series' receipts use (copied from tinyvlm): 74 columns, ASCII only, so a
report can be pasted into a Markdown code fence and read on a phone. With
``out``, every line also goes to that file, so a run on a remote GPU leaves
its report next to its checkpoints.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from . import progress
from .device import hardware_info

WIDTH = 74

__all__ = ["Report", "WIDTH"]


class Report:
    """Fixed-width console reporter shared by every command."""

    def __init__(self, title: str, out: Path | None = None) -> None:
        self.title = title
        self._file = None
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            self._file = out.open("w")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def print(self, line: str = "") -> None:
        print(line, flush=True)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    # -- structure ---------------------------------------------------------

    def header(self) -> None:
        """Title block plus the hardware snapshot the numbers depend on."""
        self.print("=" * WIDTH)
        self.print(self.title)
        self.print("=" * WIDTH)
        for line in hardware_info().render().splitlines():
            self.print(line)

    def section(self, title: str) -> None:
        progress.section(title)  # stderr, and only when stdout is a file
        self.print()
        self.print("-" * WIDTH)
        self.print(title)
        self.print("-" * WIDTH)

    def note(self, text: str) -> None:
        self.print(f"  {text}")

    def blank(self) -> None:
        self.print()

    # -- values ------------------------------------------------------------

    def kv(self, label: str, value: object, width: int = 34) -> None:
        """One aligned ``label : value`` line."""
        self.print(f"  {label:<{width}} {self._fmt(value)}")

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
        """Left-aligned first column, right-aligned numeric columns."""
        rendered = [[self._fmt(c) for c in row] for row in rows]
        widths = [len(h) for h in headers]
        for row in rendered:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def line(cells: Sequence[str]) -> str:
            out = [f"  {cells[0]:<{widths[0]}}"]
            out += [f"{c:>{widths[i]}}" for i, c in enumerate(cells[1:], start=1)]
            return "  ".join(out).rstrip()

        self.print(line(list(headers)))
        self.print("  " + "-" * (sum(widths) + 2 * len(widths)))
        for row in rendered:
            self.print(line(row))

    def takeaway(self, text: str) -> None:
        """One concluding sentence, wrapped to the report width."""
        self.print()
        for i, line in enumerate(textwrap.wrap(text, width=WIDTH - 5)):
            self.print(f"  {'>>' if i == 0 else '  '} {line}")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _fmt(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
                return f"{value:.3e}"
            return f"{value:.4f}"
        return str(value)
