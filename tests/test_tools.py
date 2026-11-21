"""The five tools over the fixed data: determinism, strict arguments, word matching, the calculator."""

import datetime as dt

import pytest

from tinytools import tools
from tinytools.tools import execute, words


def test_clock_is_frozen_and_consistent():
    c = tools.clock()
    assert (c["date"], c["time"], c["weekday"]) == ("2026-03-11", "09:30", "Wednesday")
    assert dt.date.fromisoformat(c["date"]).strftime("%A") == c["weekday"]
    assert tools.context_line() == "The current time is 2026-03-11 09:30, Wednesday."


def test_every_full_name_finds_exactly_its_contact():
    for c in tools.contacts():
        got = execute("search_contacts", {"query": c["name"]})["contacts"]
        assert [x["id"] for x in got] == [c["id"]], c["name"]


def test_confusable_names_are_told_apart_and_a_surname_alone_is_broad():
    def ids(q):
        return [x["name"] for x in execute("search_contacts", {"query": q})["contacts"]]
    assert ids("Jon Smith") == ["Jon Smith"]
    assert ids("John Smith") == ["John Smith"]
    assert ids("Smith") == ["Jon Smith", "John Smith"]
    assert ids("siobhan o'neill") == ["Siobhan O'Neill"]  # case doesn't matter
    assert ids("Siobhan O'Neill's") == []  # a possessive is not the name
    assert ids("Sean Murphy") == ["Sean Murphy"]


def test_titles_and_documents_never_contain_each_other():
    """No title's words are a subset of another's, so searching a full title finds only it."""
    for records, key in [(tools.documents(), "title"), (tools.calendar(), "title")]:
        sets = {r[key]: words(r[key], drop_stopwords=True) for r in records}
        for a, wa in sets.items():
            for b, wb in sets.items():
                if a != b:
                    assert not wa <= wb, (a, b)


def test_every_document_title_finds_exactly_itself():
    for d in tools.documents():
        got = execute("search_documents", {"query": d["title"]})["documents"]
        assert [x["id"] for x in got] == [d["id"]]
    broad = execute("search_documents", {"query": "sales report"})["documents"]
    assert len(broad) == 4


def test_calendar_lookup_by_date_and_title():
    got = execute("get_calendar_event", {"date": "2026-03-11"})["events"]
    assert got and all(e["date"] == "2026-03-11" for e in got)
    assert "error" in execute("get_calendar_event", {})
    assert "error" in execute("get_calendar_event", {"date": "March 11"})
    assert "error" in execute("get_calendar_event", {"title": "the meeting"})  # only filler words


def test_create_changes_only_the_state_it_is_given():
    state = tools.fresh_state()
    before = len(state.calendar)
    out = execute("create_calendar_event", {"title": "  Budget   sync ", "date": "2026-03-17",
                                            "start_time": "9:00", "duration_minutes": 45.0},
                  state)
    assert out["created"]["title"] == "Budget sync"
    assert out["created"]["start_time"] == "09:00" and out["created"]["duration_minutes"] == 45
    assert len(state.calendar) == before + 1
    assert len(tools.fresh_state().calendar) == before  # the next call starts fresh


@pytest.mark.parametrize("args, reason", [
    ({"title": "x", "date": "2026-03-17"}, "missing"),
    ({"title": "x", "date": "2026-03-17", "start_time": "15:00", "room": "A"}, "unknown"),
    ({"title": "x", "date": "2026-03-17", "start_time": "3pm"}, "HH:MM"),
    ({"title": "x", "date": "2026-02-30", "start_time": "15:00"}, "no such date"),
    ({"title": "x", "date": "2026-03-17", "start_time": "15:00", "duration_minutes": "60"},
     "integer"),
    ({"title": "x", "date": "2026-03-17", "start_time": "15:00", "duration_minutes": 60.5},
     "integer"),
    ({"title": "x", "date": "2026-03-17", "start_time": "15:00", "duration_minutes": True},
     "integer"),
])
def test_bad_arguments_are_errors(args, reason):
    state = tools.fresh_state()
    n = len(state.calendar)
    out = execute("create_calendar_event", args, state)
    assert reason in out["error"]
    assert len(state.calendar) == n  # nothing half-created


@pytest.mark.parametrize("expr, value", [
    ("348 * 27", 9396), ("(3 + 4) * 2", 14), ("2 ** 10", 1024), ("-5 + 2", -3),
    ("7 / 2", 3.5), ("10 % 4", 2),
])
def test_calculator(expr, value):
    assert execute("calculator", {"expression": expr}) == {"result": value}


@pytest.mark.parametrize("expr", ["1,250 + 3", "2 ^ 10", "x + 1", "__import__('os')", "1 / 0",
                                  "9 ** 9 ** 9", "10 ** 99 * 10 ** 99", "(-8) ** 0.5", ""])
def test_calculator_refuses(expr):
    assert "error" in execute("calculator", {"expression": expr})


def test_distractors_cannot_run_and_have_schemas():
    for name in tools.DISTRACTOR_NAMES:
        assert "error" in execute(name, {})
        assert tools.schema(name)["name"] == name
    assert not set(tools.TOOL_NAMES) & set(tools.DISTRACTOR_NAMES)
