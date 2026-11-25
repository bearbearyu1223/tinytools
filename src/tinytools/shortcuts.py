"""Shortcut controls for the one-tool test: rules that never read the offered tool's
operation.

A "read the tool" result means something only if a rule that ignores the tool can't
reach it. Each rule is scored as if it were a model (T1's review, plan 3e; T1b's
review 1):

    Rule A           decline if the request has a delete, change, share or symbolic-
                     math word (:data:`RULE_A_WORDS`), otherwise call
    job identity     decline if the job the request needs isn't one of the executable
                     tools' jobs (it looks at the task's job, never at the tool offered)
    domain identity  call iff the offered tool is in the request's job domain
                     (calendar, contacts, documents, math): reads the tool's app,
                     never its operation

and, printed beside them as descriptions rather than controls:

    tool-name overlap  call iff a word of the offered tool's name (or its plural) is in
                       the request (it reads the tool's name, part of which is the
                       operation)
    call iff tricky    call iff the request's values are written the tricky way: a check
                       that R's two sides are phrased alike (it should score D_R ~ 0)

Three scores for any set of calls (:func:`scores`):

* the whole set: c+ = call rate on fits items, c- = call rate on doesn't-fit items,
  D = c+ - c- (:func:`discrimination`);
* D_R, on the reading-required subset R (:func:`reading_subset`): doesn't-fit items
  whose job is an executable tool that wasn't offered, and the fits items of the same
  jobs, stratified by job (:func:`stratified`): for each job with items on both sides,
  c+_j and c-_j; c+, c- and D_R are the means over those jobs, so unequal counts per
  job don't weight the answer. On R the job-identity rule always calls (every job in R
  is executable), so its D_R is 0 by construction. R's doesn't-fit side mixes near
  misses (the same domain) with cross-domain items (another domain), so a rule that
  knows only the tool's domain scores well on it (domain identity: D_R about +41);
* D_near (:func:`near_stratified`): per operation family, operation-verb fits (the
  family's own tool offered) against near misses (another tool of the same domain
  offered), the same templates on both sides; c+, c-, D_near are the means over the
  families with both sides present. The domain is held fixed, so only reading the
  operation moves it: domain identity scores 0 here by construction.

Rates are in [0, 1]; the report prints them in points. Pure; tested in
``tests/test_shortcuts.py``. T2's r5c reuses :func:`scores` (any arm's calls) and
:data:`ALL_RULES` (the control rows).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from . import onetool_tasks as ot
from . import tools

FITS = ot.FITS
NOT_FIT = ot.NOT_FIT
VERB = ot.VERB  # operation-verb fits
NEAR = ot.NEAR  # near-miss doesn't-fit

RULE_A_FAMILIES = {
    "delete": ("delete", "remove", "cancel", "wipe", "drop", "clear", "trash", "rid"),
    "change": ("move", "reschedule", "push", "change", "update", "rename"),
    "share": ("share", "give"),
    "symbolic": ("derivative", "differentiate", "integral", "integrate", "antiderivative",
                 "factor", "plot", "graph", "draw", "sketch"),
}
RULE_A_WORDS = frozenset(w for ws in RULE_A_FAMILIES.values() for w in ws)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def rule_a_calls(task: dict) -> bool:
    """Rule A: call unless the request has a delete/change/share/symbolic word."""
    return not (_words(task["request"]) & RULE_A_WORDS)


def job_identity_calls(task: dict) -> bool:
    """The job-identity rule: call if the job needed (``tool``) is an executable tool."""
    return task["tool"] in tools.EXECUTABLE_NAMES


def domain_identity_calls(task: dict) -> bool:
    """The domain-identity rule: call iff the offered tool's domain is the job's."""
    return ot.JOBS[task["tools"][0]].domain == ot.JOBS[task["tool"]].domain


def tool_name_words(name: str) -> frozenset[str]:
    """The words of a tool's name ("get" aside) and their plurals. Pure."""
    ws = set(name.split("_")) - {"get"}
    return frozenset(ws | {w + "s" for w in ws})


def tool_name_overlap_calls(task: dict) -> bool:
    """Call iff a word of the offered tool's name (or its plural) is in the request."""
    return bool(_words(task["request"]) & tool_name_words(task["tools"][0]))


def tricky_calls(task: dict) -> bool:
    """Call iff the request's values are written the tricky way (a phrasing check)."""
    return task["phrasing"] == ot.TRICKY


# The shortcut controls: a "read the tool" verdict must beat each of them where it is
# read (D_R, D_near).
RULES: dict[str, Callable[[dict], bool]] = {"Rule A (verb only)": rule_a_calls,
                                            "job identity": job_identity_calls,
                                            "domain identity": domain_identity_calls}
# Printed beside the controls: what reading the tool's name alone reaches, and a check
# that R's sides are phrased alike.
DESCRIPTIVE: dict[str, Callable[[dict], bool]] = {"tool-name overlap": tool_name_overlap_calls}
CHECKS: dict[str, Callable[[dict], bool]] = {"call iff tricky": tricky_calls}
ALL_RULES = {**RULES, **DESCRIPTIVE, **CHECKS}


def call_expected(task: dict) -> bool:
    if task["kind"] not in (FITS, NOT_FIT):
        raise ValueError(f"not a one-tool task: {task['kind']!r}")
    return task["kind"] == FITS


@dataclass(frozen=True)
class Rates:
    n_plus: int  # items needing a call
    n_minus: int  # items needing none
    c_plus: float
    c_minus: float

    @property
    def d(self) -> float:
        return self.c_plus - self.c_minus


def discrimination(tasks: Iterable[dict], called: Iterable[bool]) -> Rates:
    """c+, c- and D over a set of one-tool items and whether each was called. Pure."""
    plus, minus = [], []
    for t, c in zip(tasks, called, strict=True):
        (plus if call_expected(t) else minus).append(bool(c))
    return Rates(len(plus), len(minus), sum(plus) / len(plus) if plus else float("nan"),
                 sum(minus) / len(minus) if minus else float("nan"))


def reading_jobs(tasks: Iterable[dict]) -> list[str]:
    """Jobs with items on both sides of R, in first-seen order. Pure."""
    tasks = list(tasks)
    minus = {t["tool"] for t in tasks if t.get("reading_required")}
    plus = {t["tool"] for t in tasks if call_expected(t)}
    return [j for j in dict.fromkeys(t["tool"] for t in tasks) if j in minus and j in plus]


def reading_subset(tasks: Iterable[dict]) -> list[dict]:
    """R: reading-required doesn't-fit items and the fits items of the same jobs, for
    jobs with items on both sides. Pure."""
    tasks = list(tasks)
    jobs = set(reading_jobs(tasks))
    return [t for t in tasks if t["tool"] in jobs
            and (t.get("reading_required") or call_expected(t))]


@dataclass(frozen=True)
class Stratified:
    jobs: tuple[str, ...]  # the strata: jobs (D_R) or operation families (D_near)
    per_job: dict  # stratum -> Rates
    c_plus: float  # mean over the strata of c+_j
    c_minus: float

    @property
    def d(self) -> float:
        return self.c_plus - self.c_minus

    @property
    def n_plus(self) -> int:
        return sum(r.n_plus for r in self.per_job.values())

    @property
    def n_minus(self) -> int:
        return sum(r.n_minus for r in self.per_job.values())


def _by_stratum(pairs: list[tuple[dict, bool]], key: Callable[[dict], str]) -> Stratified:
    """c+_s - c-_s per stratum with both sides present; means over those strata."""
    groups: dict[str, list[tuple[dict, bool]]] = {}
    for t, c in pairs:
        groups.setdefault(key(t), []).append((t, c))
    per = {}
    for s, g in groups.items():
        r = discrimination([t for t, _ in g], [c for _, c in g])
        if r.n_plus and r.n_minus:
            per[s] = r
    if not per:
        return Stratified((), {}, float("nan"), float("nan"))
    return Stratified(tuple(per), per, sum(r.c_plus for r in per.values()) / len(per),
                      sum(r.c_minus for r in per.values()) / len(per))


def _pairs(tasks: Iterable[dict], called: Iterable[bool]) -> list[tuple[dict, bool]]:
    tasks, called = list(tasks), list(called)
    if len(tasks) != len(called):
        raise ValueError("one call flag per task")
    return list(zip(tasks, called, strict=True))


def stratified(tasks: Iterable[dict], called: Iterable[bool]) -> Stratified:
    """D_R: per job on R, c+_j - c-_j; c+, c-, D_R are means over the jobs. Pure."""
    pairs = _pairs(tasks, called)
    keep = {id(t) for t in reading_subset(t for t, _ in pairs)}
    return _by_stratum([(t, c) for t, c in pairs if id(t) in keep], lambda t: t["tool"])


def near_stratified(tasks: Iterable[dict], called: Iterable[bool]) -> Stratified:
    """D_near: per operation family, operation-verb fits vs near miss (the same domain,
    the same templates); c+, c-, D_near are means over the families with both. Pure."""
    pairs = _pairs(tasks, called)
    return _by_stratum([(t, c) for t, c in pairs if t.get("subkind") in (VERB, NEAR)],
                       lambda t: t["tool"])


@dataclass(frozen=True)
class Scores:
    whole: Rates
    reading: Stratified  # D_R
    near: Stratified  # D_near


def scores(tasks: Iterable[dict], called: Iterable[bool]) -> Scores:
    """Whole-set D, D_R and D_near for one set of calls (a rule's or a model's). Pure."""
    pairs = _pairs(tasks, called)
    ts, cs = [t for t, _ in pairs], [c for _, c in pairs]
    return Scores(discrimination(ts, cs), stratified(ts, cs), near_stratified(ts, cs))


def rule_scores(tasks: list[dict], rule: Callable[[dict], bool]) -> Scores:
    """A rule's calls scored on the whole set, on R and on the near-miss families."""
    return scores(tasks, [rule(t) for t in tasks])


def score_rule(tasks: list[dict], rule: Callable[[dict], bool]) -> tuple[Rates, Stratified]:
    """A rule's calls scored on the whole set and on R."""
    s = rule_scores(tasks, rule)
    return s.whole, s.reading
