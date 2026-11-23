"""The post's comparisons, model-free: per-task shares, paired bootstrap intervals, the
effects table and the Q6 transition table (plan 3c and 3d, as fixed before the main run).

**Arms.** ``untrained`` is one greedy pass (one value per task); ``SFT`` and
``control`` have one pass per training seed. An arm's per-task value is the mean of
its seeds' 0/1 values (plan 3c), so every comparison is paired by task.

**Paired bootstrap (plan 3c).** For two arms' per-task values ``a`` and ``b`` on the
same tasks: ``d = a - b`` per task; resample the tasks with replacement 10,000 times
(numpy ``default_rng(0)``, created afresh for each comparison, so adding a comparison
changes no other); the 95% interval is the 2.5th and 97.5th percentiles of the
resampled means (numpy's default linear interpolation). An interval "excludes 0" when
both ends are on the same side of it.

**Effects (plan 3d, critique 2 B1).** Each is an unconditional per-task share: on
each task a reply has exactly one outcome, and the effect's variable is 1 when the
outcome is in the effect's set. Each gets three paired differences: SFT - untrained
(the only one a claim of what SFT taught comes from), control - untrained, and
SFT - control (the gap between the arms).

**Groups of 8 (plan 3d, "Reading Q6").** A task's group is all wrong (0 of 8 right),
mixed, or all right (8 of 8). The transition table counts, per seed, each task's
untrained group (rows) against the arm's group (columns).

Tested in ``tests/test_effects.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np

from .classify import (CALLED_ANYWAY, NO_CALL, RIGHT, UNPARSEABLE, WRONG_ARGUMENTS, WRONG_COUNT,
                       WRONG_FUNCTION, DECLINED)

RESAMPLES = 10_000
BOOT_SEED = 0
LEVEL = 95

UNTRAINED, SFT, CONTROL = "untrained", "SFT", "control"
DIFFERENCES = ((SFT, UNTRAINED), (CONTROL, UNTRAINED), (SFT, CONTROL))


# -- bootstrap -----------------------------------------------------------------------------


def bootstrap_interval(diffs, resamples: int = RESAMPLES, seed: int = BOOT_SEED,
                       level: int = LEVEL) -> tuple[float, float]:
    """The percentile interval of the mean of ``diffs`` over task resamples."""
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or len(d) == 0:
        raise ValueError("need one value per task")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    tail = (100 - level) / 2
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)


def excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def exact_mean(values: list[float], seeds: int) -> Fraction:
    """The mean of per-task values that are each k / ``seeds`` (k an integer), exactly:
    the shares print rounded half up from the true ratio, not from a float sum."""
    total = sum(values) * seeds
    k = round(total)
    if abs(total - k) > 1e-6:
        raise ValueError(f"values are not multiples of 1/{seeds}")
    return Fraction(k, seeds * len(values))


@dataclass(frozen=True)
class Paired:
    """Arm ``a`` minus arm ``b`` on the same tasks, as shares (0-1, exact)."""

    a: str
    b: str
    n: int
    share_a: Fraction
    share_b: Fraction
    lo: float
    hi: float

    @property
    def diff(self) -> float:
        return self.share_a - self.share_b

    @property
    def excludes_zero(self) -> bool:
        return excludes_zero(self.lo, self.hi)


def paired(a_name: str, a: list[float], b_name: str, b: list[float], seeds_a: int = 1,
           seeds_b: int = 1) -> Paired:
    """``a`` and ``b``: per-task values (seed means over ``seeds_a`` / ``seeds_b`` seeds)."""
    if len(a) != len(b):
        raise ValueError("paired arms must cover the same tasks")
    lo, hi = bootstrap_interval([x - y for x, y in zip(a, b, strict=True)])
    return Paired(a_name, b_name, len(a), exact_mean(a, seeds_a), exact_mean(b, seeds_b), lo, hi)


def seed_mean(per_seed: list[list[float]]) -> list[float]:
    """Per task, the mean over seeds (``per_seed[s][t]``: seed s, task t)."""
    if not per_seed or any(len(x) != len(per_seed[0]) for x in per_seed):
        raise ValueError("every seed must cover the same tasks")
    return [sum(col) / len(col) for col in zip(*per_seed, strict=True)]


def indicator(outcomes: list[str], wanted) -> list[float]:
    return [1.0 if o in wanted else 0.0 for o in outcomes]


# -- effects ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    name: str
    variable: str  # what a 1 means
    outcomes: frozenset[str]
    harm_check: bool = False


CALLING = Effect("calling when a tool fits", "no call or unparseable",
                 frozenset({NO_CALL, UNPARSEABLE}))
FILLING = Effect("filling", "wrong arguments", frozenset({WRONG_ARGUMENTS}))
DECLINING = Effect("declining when none fits", "called anyway", frozenset({CALLED_ANYWAY}))
CHOOSING = Effect("choosing", "wrong function or wrong number of calls",
                  frozenset({WRONG_FUNCTION, WRONG_COUNT}), harm_check=True)
RIGHT_SHARE = Effect("right", "right", frozenset({RIGHT, DECLINED}))
EFFECTS = (CALLING, FILLING, DECLINING, CHOOSING)

# The task sets the report reads, by key: (label, needs a call?)
OURS_CALL, OURS_NOFIT, BFCL_CALL, BFCL_IRR = "ours-call", "ours-nofit", "bfcl-call", "bfcl-irr"
SETS = {OURS_CALL: "ours, tasks that need a call", OURS_NOFIT: "ours, no tool fits",
        BFCL_CALL: "BFCL call items, pooled", BFCL_IRR: "BFCL irrelevance"}
# Which effect is read on which sets (declining only where no tool fits).
EFFECT_SETS = {CALLING: (OURS_CALL, BFCL_CALL), FILLING: (OURS_CALL, BFCL_CALL),
               DECLINING: (OURS_NOFIT, BFCL_IRR), CHOOSING: (OURS_CALL, BFCL_CALL)}
# The five primary comparisons (critique 2 S1), each SFT - untrained on the test splits.
PRIMARY = {(CALLING, OURS_CALL): 1, (FILLING, OURS_CALL): 2, (DECLINING, OURS_NOFIT): 3,
           (RIGHT_SHARE, BFCL_CALL): 4, (DECLINING, BFCL_IRR): 5}


@dataclass(frozen=True)
class EffectResult:
    effect: Effect
    set_key: str
    n: int
    shares: dict[str, Fraction]  # arm -> share (seed mean)
    diffs: tuple[Paired, ...]  # in DIFFERENCES order
    primary: int | None


def effect_result(effect: Effect, set_key: str, untrained: list[str],
                  sft: list[list[str]], control: list[list[str]]) -> EffectResult:
    """One effect on one task set. ``untrained``: an outcome per task; ``sft`` and
    ``control``: per seed, an outcome per task (same task order)."""
    vals = {UNTRAINED: indicator(untrained, effect.outcomes),
            SFT: seed_mean([indicator(x, effect.outcomes) for x in sft]),
            CONTROL: seed_mean([indicator(x, effect.outcomes) for x in control])}
    seeds = {UNTRAINED: 1, SFT: len(sft), CONTROL: len(control)}
    diffs = tuple(paired(a, vals[a], b, vals[b], seeds[a], seeds[b]) for a, b in DIFFERENCES)
    return EffectResult(effect, set_key, len(untrained),
                        {k: exact_mean(v, seeds[k]) for k, v in vals.items()}, diffs,
                        PRIMARY.get((effect, set_key)))


# -- right-function rate vs chance (r5b, the control's diagnostic) ----------------------------

PARSEABLE = frozenset({WRONG_FUNCTION, WRONG_COUNT, WRONG_ARGUMENTS, RIGHT})
RIGHT_FUNCTION = frozenset({WRONG_COUNT, WRONG_ARGUMENTS, RIGHT})


def right_function(outcomes: list[str], tools_offered: list[int]) -> tuple[int, int, float]:
    """Among replies that decoded to calls: (right function, parseable, expected by
    chance = sum of 1/k over them, k the tools offered on that task)."""
    hits = n = 0
    chance = 0.0
    for o, k in zip(outcomes, tools_offered, strict=True):
        if o in PARSEABLE:
            n += 1
            hits += o in RIGHT_FUNCTION
            chance += 1 / k
    return hits, n, chance


# -- groups of 8 and transitions (r6a) ------------------------------------------------------------

ALL_WRONG, MIXED, ALL_RIGHT = "all wrong", "mixed", "all right"
GROUPS = (ALL_WRONG, MIXED, ALL_RIGHT)


def group(right: int, samples: int) -> str:
    if not 0 <= right <= samples:
        raise ValueError(f"{right} right of {samples}")
    return ALL_WRONG if right == 0 else ALL_RIGHT if right == samples else MIXED


def transitions(untrained: list[int], arm: list[int], samples: int) -> dict[tuple[str, str], int]:
    """Tasks by (untrained group, arm group), one seed's arm."""
    out = {(a, b): 0 for a in GROUPS for b in GROUPS}
    for u, x in zip(untrained, arm, strict=True):
        out[(group(u, samples), group(x, samples))] += 1
    return out


def add_tables(tables: list[dict]) -> dict:
    return {k: sum(t[k] for t in tables) for k in tables[0]}


# Named movements (plan 3d, "Reading Q6"): (from untrained, to arm) -> reading.
# "{arm}" is filled with the arm's name when printed.
MOVEMENTS = (
    ((MIXED, ALL_RIGHT), "{arm} taught it; RL has less left to do there"),
    ((ALL_WRONG, MIXED), "new signal"),
    ((MIXED, ALL_WRONG), "damage"),
    ((ALL_WRONG, ALL_WRONG), "neither helps (post 10's case for RL)"),
)


def movement_values(untrained: list[int], per_seed: list[list[int]], samples: int,
                    move: tuple[str, str]) -> list[float]:
    """Per task, the share of the arm's seeds where the task made this movement."""
    return seed_mean([[1.0 if (group(u, samples), group(x, samples)) == move else 0.0
                       for u, x in zip(untrained, seed, strict=True)] for seed in per_seed])


def group_values(rights: list[int], samples: int, which: str) -> list[float]:
    return [1.0 if group(r, samples) == which else 0.0 for r in rights]


# -- printing helpers ---------------------------------------------------------------------------


def _dec(x) -> Decimal:
    """100 x as a Decimal: exact for a Fraction, from the shortest repr for a float."""
    if isinstance(x, Fraction):
        return Decimal(100 * x.numerator) / Decimal(x.denominator)
    return Decimal(repr(100 * float(x)))


def points(x) -> str:
    """A share difference in percentage points, signed, one decimal, half away from 0."""
    v = _dec(x).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return "0.0" if v == 0 else f"{'+' if v > 0 else ''}{v}"


def interval(lo: float, hi: float) -> str:
    return f"[{points(lo)}, {points(hi)}]"


def share_interval(lo: float, hi: float) -> str:
    """An interval of a share (not a difference), in percent, unsigned."""
    return f"[{percent(lo)[:-1]}, {percent(hi)[:-1]}]"


def percent(x) -> str:
    return f"{_dec(x).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"
