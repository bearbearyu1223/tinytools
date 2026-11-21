"""The task generator: sizes, mix, disjointness, references that run, relative dates, hashes."""

import datetime as dt
import json
from collections import Counter

import pytest

from tinytools import tasks, tools
from tinytools.checkers import check_reply, run


@pytest.fixture(scope="module")
def splits():
    return tasks.generate(tasks.DEFAULT_SEED)


# Pinned: regenerating the splits with seed 0 must give these bytes. A change to a
# template, the data or the generator shows up here first; update deliberately.
# (train-sample was train-r6a until review round 1; the file name is not part of the
# hash input, so its hash did not change with the rename.)
PINNED_SHA256 = {
    "test": "ef5f1ae004e76b72d10ff6f87053f93360d92c6d22c06fa98f8723b629a45cd4",
    "dev": "fabf92cbf4ea84fecef3fcffd9051a67520bcdff539f13d4744d1d5796bcb83c",
    "train": "d44888a0368a1b0a718c53b40c83c248441fd265ce3be66dee0f856db71c61ea",
    "train-sample": "fd6c706bd0e1bfe7c35c831cce11581f4f087187b44fbc6073bf05c4307c5651",
}


def test_regenerating_gives_the_same_bytes(splits):
    again = tasks.generate(tasks.DEFAULT_SEED)
    for name in splits:
        assert tasks.sha256(splits[name]) == tasks.sha256(again[name])
        assert tasks.sha256(splits[name]) == PINNED_SHA256[name], name


def test_sizes_and_mix(splits):
    for name, sizes in [*tasks.SPLITS.items(), (tasks.SAMPLE_SPLIT, tasks.SAMPLE_SIZES)]:
        got = Counter(t["kind"] for t in splits[name])
        assert [got[k] for k in tasks.KINDS] == list(sizes), name


def test_splits_disjoint_by_text_and_by_template_slots(splits):
    keys = {}
    for name in tasks.SPLITS:
        keys[name] = ({t["request"] for t in splits[name]},
                      {(t["template"], repr(sorted(t["slots"].items()))) for t in splits[name]})
        assert len(keys[name][0]) == len(splits[name])  # unique within the split too
    names = list(tasks.SPLITS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not keys[a][0] & keys[b][0]
            assert not keys[a][1] & keys[b][1]
    train_ids = {t["id"] for t in splits["train"]}
    assert {t["id"] for t in splits[tasks.SAMPLE_SPLIT]} <= train_ids


def test_offered_tools(splits):
    for t in (t for name in tasks.SPLITS for t in splits[name]):
        assert 3 <= len(t["tools"]) <= 5 and len(set(t["tools"])) == len(t["tools"])
        if t["reference"] is None:
            assert t["tool"] not in t["tools"]
            assert not set(tasks.SUBSTITUTES.get(t["tool"], ())) & set(t["tools"])
        else:
            assert t["reference"]["name"] == t["tool"] and t["tool"] in t["tools"]


def test_every_reference_runs_finds_something_and_checks_right(splits):
    for name in tasks.SPLITS:
        for t in splits[name]:
            ref = t["reference"]
            if ref is None:
                continue
            r = run(ref["name"], ref["arguments"])
            assert "error" not in r.result, t["id"]
            if ref["name"] != "calculator" and ref["name"] != "create_calendar_event":
                assert r.compared, t["id"]  # never an empty result set
            raw = '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (
                ref["name"], json.dumps(ref["arguments"]))
            c = check_reply(raw, ref)
            assert c.exact and c.execution, t["id"]


def test_relative_dates_against_the_frozen_clock():
    today = dt.date(2026, 3, 11)  # a Wednesday
    got = dict(tasks.relative_dates(today))
    expect = {"tomorrow": "2026-03-12", "next Tuesday": "2026-03-17",
              "next Wednesday": "2026-03-18", "this Friday": "2026-03-13",
              "last Friday": "2026-03-06", "on the Monday after next": "2026-03-23",
              "a week from Friday": "2026-03-20", "on the 20th": "2026-03-20",
              "on the 2nd of next month": "2026-04-02", "in 5 days": "2026-03-16",
              "yesterday": "2026-03-10", "five days from now": "2026-03-16"}
    for phrase, iso in expect.items():
        assert got[phrase].isoformat() == iso, phrase
    assert "next Friday" not in got and "last Monday" not in got  # the ambiguous ones
    assert tools.now().date() == today


def test_spoken_times():
    got = dict(tasks.spoken_times())
    assert got["3pm"] == "15:00" and got["half past two in the afternoon"] == "14:30"
    assert got["noon"] == "12:00" and got["9:30am"] == "09:30"
    assert got["eight in the morning"] == "08:00" and got["5:15 pm"] == "17:15"


def test_per_tool_and_ordinal():
    assert tasks.per_tool(200, 6) == [34, 34, 33, 33, 33, 33]
    assert sum(tasks.per_tool(1000, 6)) == 1000
    assert [tasks.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 31)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "31st"]


def test_prompt_messages_carry_the_clock(splits):
    m = tasks.messages(splits["test"][0])
    assert m[0]["role"] == "system" and "2026-03-11 09:30, Wednesday" in m[0]["content"]
    assert m[1] == {"role": "user", "content": splits["test"][0]["request"]}
