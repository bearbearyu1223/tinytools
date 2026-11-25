"""Post 10's twelve operation tools: their own registry (post 9 unchanged), each one's
execution check, and that the check can't be gamed (fresh state, whole-state compare,
exact names and titles, a too-broad call fails)."""

import hashlib
import json

import pytest

from tinytools import checkers, tools
from tinytools import onetool_tasks as ot
from tinytools.checkers import check_reply, run


def reply(name, args):
    return '<tool_call>\n%s\n</tool_call>' % json.dumps({"name": name, "arguments": args})


def passes(name, args, ref_name, ref_args):
    ref = {"name": ref_name, "arguments": ref_args}
    return check_reply(reply(name, args), ref).execution


# -- the registry --------------------------------------------------------------------


def test_post9s_registry_is_untouched():
    assert tools.TOOL_NAMES == ("calculator", "search_contacts", "get_calendar_event",
                                "create_calendar_event", "search_documents")
    assert tools.DISTRACTOR_NAMES == ("send_email", "delete_calendar_event", "get_weather",
                                      "set_reminder", "get_stock_price", "translate_text")
    assert hashlib.sha256(ot.post9_schema_bytes()).hexdigest() == ot.POST9_SCHEMAS_SHA256
    assert not set(tools.OPERATION_NAMES) & (set(tools.SCHEMAS) | set(tools.DISTRACTORS))
    for name in tools.TOOL_NAMES + tools.DISTRACTOR_NAMES:  # schema() returns post 9's
        assert tools.schema(name) == (tools.SCHEMAS.get(name) or tools.DISTRACTORS[name])


def test_twelve_operation_tools_that_all_run():
    assert len(tools.OPERATION_NAMES) == 12 and "plot_graph" not in tools.OPERATION_NAMES
    assert tools.EXECUTABLE_NAMES == tools.TOOL_NAMES + tools.OPERATION_NAMES
    for name in tools.OPERATION_NAMES:
        s = tools.schema(name)
        assert s["name"] == name and set(s["parameters"]["required"]) <= set(
            s["parameters"]["properties"])


def test_delete_calendar_event_stays_a_distractor():
    out = tools.execute("delete_calendar_event", {"event_id": "evt-002"})
    assert "error" in out and "distractor" in out["error"]


# -- each tool's effect, and its execution check ----------------------------------------


def _ids(records):
    return {r["id"] for r in records}


def test_cancel_calendar_event():
    state = tools.fresh_state()
    out = tools.execute("cancel_calendar_event", {"title": "Usability testing"}, state)
    assert out["cancelled"]["id"] == "evt-002"
    assert "evt-002" not in _ids(state.calendar) and len(state.calendar) == 77
    assert "evt-002" in _ids(tools.fresh_state().calendar)  # state reset for the next call
    # a title that occurs 9 times needs its date; with it, exactly one goes
    assert "error" in tools.execute("cancel_calendar_event", {"title": "Weekly planning"})
    s2 = tools.fresh_state()
    tools.execute("cancel_calendar_event", {"title": "Weekly planning", "date": "2026-02-23"}, s2)
    assert len(s2.calendar) == 77
    # a part of a title is not the title (no search: a too-broad call fails)
    assert "error" in tools.execute("cancel_calendar_event", {"title": "Usability"})
    assert passes("cancel_calendar_event", {"title": "the usability testing event"},
                  "cancel_calendar_event", {"title": "Usability testing"})
    assert not passes("cancel_calendar_event", {"title": "Sprint retrospective"},
                      "cancel_calendar_event", {"title": "Usability testing"})


def test_update_calendar_event():
    state = tools.fresh_state()
    out = tools.execute("update_calendar_event",
                        {"title": "Usability testing", "start_time": "15:30"}, state)
    assert out["updated"]["start_time"] == "15:30" and out["updated"]["duration_minutes"] == 60
    assert "error" in tools.execute("update_calendar_event", {"title": "Usability testing"})
    assert "error" in tools.execute("update_calendar_event",
                                    {"title": "Design review", "start_time": "10:00"})
    ref = {"title": "Usability testing", "start_time": "15:30"}
    assert passes("update_calendar_event", {"title": "Usability testing",
                                            "start_time": "15:30:00"},
                  "update_calendar_event", ref)
    assert not passes("update_calendar_event", {"title": "Usability testing",
                                                "start_time": "15:30", "date": "2026-03-20"},
                      "update_calendar_event", ref)  # changed the day too
    # a new event at the new time is not a moved event
    assert not passes("create_calendar_event", {"title": "Usability testing", "date": "2026-02-23",
                                                "start_time": "15:30"},
                      "update_calendar_event", ref)


def test_add_update_delete_contact():
    s = tools.fresh_state()
    out = tools.execute("add_contact", {"name": "Alma Abbott", "phone": "+1-555-111-2222"}, s)
    assert out["added"]["id"] == "c-091" and len(s.contacts) == 91
    assert "error" in tools.execute("add_contact", {"name": "maria  garcia"})  # exists
    ref = {"name": "Alma Abbott", "phone": "+1-555-111-2222"}
    assert passes("add_contact", {"name": "alma abbott", "phone": "+1 555 111 2222"},
                  "add_contact", ref)  # phone digits, name case
    assert not passes("add_contact", {"name": "Alma Abbott", "phone": "+1-555-111-2222",
                                      "team": "Sales"}, "add_contact", ref)  # an extra field

    ref = {"name": "Jon Smith", "team": "Legal"}
    assert passes("update_contact", ref, "update_contact", ref)
    # contact-list spelling: John Smith is another contact
    assert not passes("update_contact", {"name": "John Smith", "team": "Legal"},
                      "update_contact", ref)
    assert "error" in tools.execute("update_contact", {"name": "Jon Smith"})  # nothing new
    assert "error" in tools.execute("update_contact", {"name": "Smith", "team": "Legal"})

    ref = {"name": "Jon Smith"}
    assert passes("delete_contact", {"name": "jon smith"}, "delete_contact", ref)
    assert not passes("delete_contact", {"name": "John Smith"}, "delete_contact", ref)
    assert "error" in tools.execute("delete_contact", {"name": "Smith"})  # too broad: none


def test_document_tools():
    s = tools.fresh_state()
    out = tools.execute("create_document", {"title": "Q1 vendor scorecard"}, s)
    assert out["created"]["updated"] == tools.clock()["date"] and len(s.documents) == 81
    assert "error" in tools.execute("create_document", {"title": "the Q1 sales report"})
    ref = {"title": "Q1 sales report"}
    assert passes("delete_document", {"title": "q1 sales report doc"}, "delete_document", ref)
    assert not passes("delete_document", {"title": "Q2 sales report"}, "delete_document", ref)
    assert "error" in tools.execute("delete_document", {"title": "sales report"})  # 4 match
    ref = {"title": "Q1 sales report", "new_title": "Q1 revenue report"}
    assert passes("rename_document", ref, "rename_document", ref)
    assert "error" in tools.execute("rename_document", {"title": "Q1 sales report",
                                                        "new_title": "Q2 sales report"})
    s = tools.fresh_state()
    tools.execute("share_document", {"title": "Q1 sales report", "contact": "fiona grant"}, s)
    doc = next(d for d in s.documents if d["title"] == "Q1 sales report")
    assert doc["shared_with"] == ["Fiona Grant"]  # stored as the list spells it
    ref = {"title": "Q1 sales report", "contact": "Jon Smith"}
    assert passes("share_document", {"title": "Q1 sales report", "contact": "JON SMITH"},
                  "share_document", ref)
    assert not passes("share_document", {"title": "Q1 sales report", "contact": "John Smith"},
                      "share_document", ref)
    assert "error" in tools.execute("share_document", {"title": "Q1 sales report",
                                                       "contact": "Dana Nobody"})


def test_math_tools():
    assert tools.execute("differentiate", {"polynomial": "x^3 + 2x"}) == {"result": "3x^2 + 2"}
    assert tools.execute("integrate", {"polynomial": "3x^2"}) == {"result": "x^3"}
    assert "error" in tools.execute("factor_polynomial", {"polynomial": "x^2 + 1"})
    assert "error" in tools.execute("integrate", {"polynomial": "sin(x)"})
    ref = {"polynomial": "x^2 + 5x + 6"}
    assert passes("factor_polynomial", {"polynomial": "x**2 + 5*x + 6"}, "factor_polynomial", ref)
    assert passes("differentiate", {"polynomial": "6 + 5x + x^2"}, "differentiate", ref)
    # the answer worked out by hand, sent to the tool, is not the work asked for
    assert not passes("differentiate", {"polynomial": "2x + 5"}, "differentiate", ref)
    assert not passes("integrate", {"polynomial": "x^2 + 5x + 6"}, "differentiate", ref)


def test_factor_result_is_sorted_by_root():
    assert tools.execute("factor_polynomial", {"polynomial": "x^2 + 5x + 6"}) == {
        "result": "(x + 3)(x + 2)"}
    assert tools.execute("factor_polynomial", {"polynomial": "x^2 - x - 6"}) == {
        "result": "(x + 2)(x - 3)"}


@pytest.mark.parametrize("name, args", [
    ("cancel_calendar_event", {"title": "Usability testing"}),
    ("update_contact", {"name": "Jon Smith", "team": "Legal"}),
    ("share_document", {"title": "Q1 sales report", "contact": "Fiona Grant"}),
    ("integrate", {"polynomial": "x^2"}),
])
def test_deterministic_and_reset(name, args):
    a, b = run(name, args), run(name, args)
    assert a == b and a.compared is not None
    assert checkers.whole_state_key(tools.fresh_state()) == checkers.whole_state_key(
        tools.fresh_state())  # nothing leaked into the data


def test_a_state_tool_is_compared_on_the_whole_state():
    """Deleting the right document plus another (two calls) can't pass: one call is
    checked, and the checker compares every record afterwards."""
    one = run("delete_document", {"title": "Q1 sales report"}).compared
    other = run("delete_document", {"title": "Q2 sales report"}).compared
    assert one != other and one[0] == other[0]  # calendar the same, documents differ


# -- T1b review 3 and 5: wrong calls that passed, right calls that failed -------------


@pytest.mark.parametrize("given", ["x^2 + 3x", "x^2 + 3x - 99", "x^2 + 3x + 6"])
def test_differentiate_on_another_constant_term_fails(given):
    """d/dx drops the constant, so the derivatives agree: the input is compared too."""
    ref = {"polynomial": "x^2 + 3x + 5"}
    assert run("differentiate", {"polynomial": given}).result == run("differentiate", ref).result
    assert not passes("differentiate", {"polynomial": given}, "differentiate", ref)


@pytest.mark.parametrize("given", ["x^2 - x - 6 = 0", "(x^2 - x - 6)", "x^2 - x + -6"])
def test_factor_reads_the_same_polynomial_written_otherwise(given):
    assert passes("factor_polynomial", {"polynomial": given}, "factor_polynomial",
                  {"polynomial": "x^2 - x - 6"})


@pytest.mark.parametrize("given", ['"Q1 vendor scorecard"', "Q1 vendor scorecard.",
                                   "The Q1 vendor scorecard", "q1  VENDOR scorecard doc"])
def test_a_written_title_compares_as_the_tool_reads_titles(given):
    assert passes("create_document", {"title": given}, "create_document",
                  {"title": "Q1 vendor scorecard"})
    ref = {"title": "Q3 sales report", "new_title": "Q1 vendor scorecard"}
    assert passes("rename_document", {"title": "Q3 sales report", "new_title": given},
                  "rename_document", ref)


def test_a_written_title_with_other_words_fails():
    ref = {"title": "Q1 vendor scorecard"}
    for given in ("Q2 vendor scorecard", "Q1 vendor scorecards", "vendor scorecard"):
        assert not passes("create_document", {"title": given}, "create_document", ref)
    assert tools.title_key('The "Q1 vendor scorecard".') == ("q1", "scorecard", "vendor")


def test_a_phone_compares_by_its_digits():
    ref = {"name": "Maria Garcia", "phone": "+1-555-123-4567"}
    for given in ("1-555-123-4567", "+1 (555) 123 4567", "15551234567"):
        assert passes("update_contact", {"name": "Maria Garcia", "phone": given},
                      "update_contact", ref)
    assert not passes("update_contact", {"name": "Maria Garcia", "phone": "555-123-4567"},
                      "update_contact", ref)  # the country code dropped: other digits


def test_post9s_checks_are_unchanged():
    """The title, phone and math changes touch the operation tools only: post 9's
    create_calendar_event still compares titles as casefolded text, and the searches
    compare record ids."""
    ref = {"title": "Team sync", "date": "2026-03-02", "start_time": "10:00",
           "duration_minutes": 30}
    assert passes("create_calendar_event", {**ref, "title": "team  SYNC"},
                  "create_calendar_event", ref)
    assert not passes("create_calendar_event", {**ref, "title": '"Team sync"'},
                      "create_calendar_event", ref)
    assert not passes("create_calendar_event", {**ref, "title": "The Team sync"},
                      "create_calendar_event", ref)
    assert run("search_documents", {"query": "Q1 sales report"}).compared == frozenset(
        d["id"] for d in tools.documents() if d["title"] == "Q1 sales report")
