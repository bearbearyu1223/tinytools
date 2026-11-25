"""Shortcut controls: rules that never read the offered tool, scored on the whole
one-tool set and on the reading-required subset R (stratified by job)."""

import math

import pytest

from tinytools import shortcuts as sc

FITS, NOT_FIT = sc.FITS, sc.NOT_FIT


def task(kind, job, request="Look this up.", reading=None):
    return {"kind": kind, "tool": job, "request": request,
            "reading_required": kind == NOT_FIT and job != "get_weather"
            if reading is None else reading}


def test_rule_a_words_and_job_identity():
    assert not sc.rule_a_calls({"request": "Cancel the Haircut event."})
    assert not sc.rule_a_calls({"request": "What's the derivative of x^2?"})
    assert sc.rule_a_calls({"request": "Add Dana Lee to my contacts."})
    assert sc.rule_a_calls({"request": "Cancellation policy?"})  # whole words only
    assert sc.job_identity_calls({"tool": "delete_contact"})
    assert not sc.job_identity_calls({"tool": "get_weather"})
    assert not sc.job_identity_calls({"tool": "delete_calendar_event"})  # a distractor


def test_discrimination():
    tasks = [task(FITS, "a"), task(FITS, "a"), task(NOT_FIT, "a"), task(NOT_FIT, "b")]
    r = sc.discrimination(tasks, [True, False, True, False])
    assert (r.n_plus, r.n_minus, r.c_plus, r.c_minus, r.d) == (2, 2, 0.5, 0.5, 0.0)
    with pytest.raises(ValueError):
        sc.discrimination([{"kind": "choose+fill"}], [True])


def test_stratified_weights_jobs_equally():
    # job a: 1 fits called, 3 doesn't-fit, 1 called; job b: 3 fits, 0 called; 1 doesn't-fit
    tasks = ([task(FITS, "a")] + [task(NOT_FIT, "a")] * 3
             + [task(FITS, "b")] * 3 + [task(NOT_FIT, "b")]
             + [task(NOT_FIT, "get_weather")])  # a distractor job: not in R
    called = [True, True, False, False, False, False, False, True, True]
    s = sc.stratified(tasks, called)
    assert s.jobs == ("a", "b")
    assert math.isclose(s.c_plus, (1.0 + 0.0) / 2)
    assert math.isclose(s.c_minus, (1 / 3 + 1.0) / 2)
    assert math.isclose(s.d, s.c_plus - s.c_minus)
    assert (s.n_plus, s.n_minus) == (4, 4)


def test_reading_subset_needs_both_sides():
    tasks = [task(FITS, "a"), task(NOT_FIT, "a"), task(FITS, "c"), task(NOT_FIT, "d")]
    assert sc.reading_jobs(tasks) == ["a"]
    assert sc.reading_subset(tasks) == tasks[:2]
    assert math.isnan(sc.stratified(tasks[2:], [True, True]).d)


def test_job_identity_is_zero_on_r_by_construction():
    tasks = [task(FITS, "delete_contact", "Delete Jon Smith."),
             task(NOT_FIT, "delete_contact", "Remove Jon Smith."),
             task(FITS, "calculator", "Work out 2 times 3."),
             task(NOT_FIT, "calculator", "Work out 4 times 3."),
             task(NOT_FIT, "get_weather", "Weather in Oslo?")]
    whole, r = sc.score_rule(tasks, sc.job_identity_calls)
    assert r.d == 0.0 and r.c_plus == r.c_minus == 1.0
    assert math.isclose(whole.c_minus, 2 / 3)
    _, ra = sc.score_rule(tasks, sc.rule_a_calls)
    assert ra.d == 0.0  # the same job's words on both sides


def item(kind, subkind, job, offered, request="Do it.", phrasing="explicit values"):
    return {"kind": kind, "subkind": subkind, "tool": job, "tools": [offered],
            "request": request, "phrasing": phrasing,
            "reading_required": kind == NOT_FIT and job != "get_weather"}


def test_domain_identity_and_tool_name_overlap():
    assert sc.domain_identity_calls(item(NOT_FIT, sc.NEAR, "add_contact", "delete_contact"))
    assert not sc.domain_identity_calls(item(NOT_FIT, "cross-domain", "add_contact",
                                             "integrate"))
    assert not sc.domain_identity_calls(item(NOT_FIT, "cross-domain", "get_weather",
                                             "calculator"))
    assert sc.tool_name_words("get_calendar_event") == {"calendar", "event", "calendars",
                                                        "events"}
    assert sc.tool_name_overlap_calls(item(FITS, sc.VERB, "delete_contact", "delete_contact",
                                           "Delete Jon Smith from my contacts."))
    assert not sc.tool_name_overlap_calls(item(FITS, sc.VERB, "delete_contact",
                                               "delete_contact", "Remove Jon Smith."))
    assert sc.tricky_calls({"phrasing": "tricky values"})
    assert not sc.tricky_calls({"phrasing": "explicit values"})


def test_d_near_holds_the_domain_fixed():
    """D_near: per family, operation-verb fits vs near miss; cross-domain and plain items
    don't enter it. Domain identity calls on both sides of every family: 0."""
    tasks = [item(FITS, sc.VERB, "delete_contact", "delete_contact"),
             item(NOT_FIT, sc.NEAR, "delete_contact", "add_contact"),
             item(NOT_FIT, sc.NEAR, "delete_contact", "update_contact"),
             item(FITS, sc.VERB, "integrate", "integrate"),
             item(NOT_FIT, sc.NEAR, "integrate", "differentiate"),
             item(NOT_FIT, "cross-domain", "integrate", "add_contact"),
             item(FITS, "plain", "calculator", "calculator"),
             item(FITS, sc.VERB, "share_document", "share_document")]  # no near miss: out
    called = [True, True, False, True, False, False, True, True]
    n = sc.near_stratified(tasks, called)
    assert n.jobs == ("delete_contact", "integrate")
    assert math.isclose(n.c_plus, 1.0) and math.isclose(n.c_minus, (0.5 + 0.0) / 2)
    assert (n.n_plus, n.n_minus) == (2, 3)
    s = sc.rule_scores(tasks, sc.domain_identity_calls)
    assert s.near.d == 0.0 and s.reading.d > 0  # cross-domain is on R's minus side
    whole, r = sc.score_rule(tasks, sc.domain_identity_calls)
    assert (whole, r) == (s.whole, s.reading)
    with pytest.raises(ValueError):
        sc.scores(tasks, called[:-1])


def test_the_rule_groups():
    assert list(sc.RULES) == ["Rule A (verb only)", "job identity", "domain identity"]
    assert list(sc.ALL_RULES) == [*sc.RULES, "tool-name overlap", "call iff tricky"]
