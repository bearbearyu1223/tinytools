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

**Nothing wider than WIDTH is printed** (T8 review S1): :meth:`Report.print` raises
``ValueError`` on a wider line, before it reaches stdout or the file, so a value that
only the GPU prints (a CUDA flag, a full-size count) fails where it is made instead of
landing in a capture. Values that can be long are printed with ``kv(..., wrap=True)``
or ``note(..., wrap=True)``. The check applies to what is printed from now on; captures
already in ``outputs/`` (``milestone0.txt``'s 97-column determinism line, from the
Lambda run) are records of what ran and are not rewritten.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

from . import progress
from .device import hardware_info

WIDTH = 74
STRICT_ENV = "TINYTOOLS_STRICT_WIDTH"  # "1": a line over WIDTH raises (the tests); else folded


def fold(line: str) -> str:
    """``line`` cut into pieces of at most WIDTH columns, each continuation indented 4."""
    if len(line) <= WIDTH:
        return line
    out, rest = [line[:WIDTH]], line[WIDTH:]
    while rest:
        out.append("    " + rest[:WIDTH - 4])
        rest = rest[WIDTH - 4:]
    return "\n".join(out)

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
        wide = [x for x in line.split("\n") if len(x) > WIDTH]
        if wide:
            if os.environ.get(STRICT_ENV) == "1":  # the tests set it (tests/conftest.py)
                raise ValueError(f"a report line is {len(wide[0])} columns, over {WIDTH}: "
                                 f"{wide[0]!r}")
            # A real run never dies over a line's width (a Lambda run did, 2026-11-22): the
            # line is folded at WIDTH, continuation lines indented, and stderr says so.
            line = "\n".join(fold(x) for x in line.split("\n"))
            sys.stderr.write(f"report: folded a line over {WIDTH} columns: {wide[0][:60]!r}\n")
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

    def note(self, text: str, wrap: bool = False) -> None:
        """An indented line; ``wrap``: a text of any length, wrapped to the width."""
        if not wrap:
            self.print(f"  {text}")
            return
        lead = len(text) - len(text.lstrip(" "))
        for line in textwrap.wrap(text.strip(), width=WIDTH - 2 - lead) or [""]:
            self.print(f"  {' ' * lead}{line}")

    def blank(self) -> None:
        self.print()

    # -- values ------------------------------------------------------------

    def kv(self, label: str, value: object, width: int = 34, wrap: bool = False) -> None:
        """One aligned ``label : value`` line. ``wrap``: a long value goes on over
        more lines, each starting at the value's column."""
        text = self._fmt(value)
        if not wrap:
            self.print(f"  {label:<{width}} {text}")
            return
        lines = textwrap.wrap(text, width=WIDTH - width - 3) or [""]
        self.print(f"  {label:<{width}} {lines[0]}")
        for line in lines[1:]:
            self.print(f"  {'':<{width}} {line}")

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
        # The rule runs 2 columns past the rows (kept: narrower tables' captures stay byte
        # for byte), capped so its line never passes WIDTH and folds (round-3 review).
        self.print("  " + "-" * min(sum(widths) + 2 * len(widths), WIDTH - 2))
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
