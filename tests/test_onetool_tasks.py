"""Post 10's one-tool kinds (plan 3b v3.2): sizes, disjointness (within, and from post
9), pools big enough, the near-miss rule, per-family balance, dev coverage, the entity
holdout, references that run, the shortcut controls, the step mix."""

import datetime as dt
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from tinytools import onetool_tasks as ot
from tinytools import shortcuts as sc
from tinytools import tasks, tools
from tinytools.checkers import check_reply, run

TESTS = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def post9():
    return tasks.generate(tasks.DEFAULT_SEED)


@pytest.fixture(scope="module")
def splits(post9):
    return ot.generate(tasks.DEFAULT_SEED, post9=post9)


def every(splits):
    return [t for name in ot.SPLITS for t in splits[name]]


# Pinned: regenerating with seed 0 must give these bytes. Update deliberately.
# T1c (T1b review 2 and 4) moved them from T1b's d3595879c78e93c7 / 21b4981371d855bd /
# 052681ec0dc30636 / 6fbc67470c088526.
PINNED_SHA256 = {
    "test": "7b0ae3f5230de8ac",
    "dev": "b80cdf8aecf2f7b8",
    "train": "bc4289c6b3b26635",
    "train-sample": "9ac4fcba61d7adb1",
}


def test_regenerating_gives_the_same_bytes(splits):
    again = ot.generate(tasks.DEFAULT_SEED)
    for name in splits:
        assert ot.sha256(splits[name]) == ot.sha256(again[name])
        assert ot.sha256(splits[name])[:16] == PINNED_SHA256[name], name


def test_post9_splits_unchanged(post9, splits):
    for name, pinned in ot.POST9_SHA256.items():
        assert tasks.sha256(post9[name]) == pinned, name
    assert tasks.sha256(tasks.generate(tasks.DEFAULT_SEED)["test"]) == ot.POST9_SHA256["test"]


def test_post9_pins_agree_with_post9s_own_test():
    spec = importlib.util.spec_from_file_location("post9_task_tests", TESTS / "test_tasks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.PINNED_SHA256 == ot.POST9_SHA256


def test_post9_never_offers_a_new_tool(post9):
    for name in tasks.SPLITS:
        for t in post9[name]:
            assert not set(t["tools"]) & set(tools.OPERATION_NAMES)
            assert all(f["function"] == tools.schema(n)
                       for f, n in zip(tasks.tool_list(t), t["tools"], strict=True))


def test_sizes(splits):
    for name, sizes in [*ot.SPLITS.items(), (ot.SAMPLE_SPLIT, ot.SAMPLE_SIZES)]:
        assert dict(ot.counts(splits[name])) == sizes, name
    # plan 3b: test 250 / 250 / 250 / 250 (plain 188 / 62; cross-domain 125 + 125)
    assert [len(splits[s]) for s in splits] == [1000, 80, 2000, 200]
    for name in splits:
        n = len(splits[name])
        sub = ot.subkind_counts(splits[name])
        assert all(sub[(k, s)] == n // 4 for k in ot.KINDS for s in ot.SUBKINDS[k]), name


def test_deal():
    rng = random.Random(0)
    got = ot.deal(20, 14, rng)
    assert sum(got) == 20 and set(got) == {1, 2} and got.count(2) == 6
    assert ot.deal(6, 3, rng) == [2, 2, 2]
    with pytest.raises(ValueError):
        ot.deal(3, 0, rng)


def test_cell_sizes_add_up():
    for split, sizes in ot.SPLITS.items():
        got = Counter()
        for c, n in ot.cell_sizes(sizes, 0, split).items():
            got[c.stratum] += n
        assert dict(got) == sizes


def _keys(records):
    return ({t["request"] for t in records},
            {(t["template"], json.dumps(t["slots"], sort_keys=True)) for t in records})


def test_splits_disjoint_by_text_and_template_slots(splits):
    keys = {}
    for name in ot.SPLITS:
        keys[name] = _keys(splits[name])
        assert len(keys[name][0]) == len(splits[name])
        assert len(keys[name][1]) == len(splits[name])
    names = list(ot.SPLITS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not keys[a][0] & keys[b][0], (a, b)
            assert not keys[a][1] & keys[b][1], (a, b)
    assert {t["id"] for t in splits[ot.SAMPLE_SPLIT]} <= {t["id"] for t in splits["train"]}


def test_disjoint_from_every_post9_split(post9, splits):
    p9_text = set().union(*(_keys(post9[n])[0] for n in tasks.SPLITS))
    for name in ot.SPLITS:
        assert not _keys(splits[name])[0] & p9_text, name
        assert all(t["template"].startswith("1t-") for t in splits[name])
        assert all(t["id"].startswith(ot.FILE_PREFIX) for t in splits[name])


def test_distractor_entities_held_out_from_post9(post9, splits):
    held = ot.post9_entities(post9)
    assert held  # post 9's no-tool-fits items are in it
    for t in every(splits):
        if t["job_class"] == ot.DISTRACTOR:
            assert ot.entity(t["tool"], t["slots"]) not in held, t["id"]


def test_pools_cover_test_dev_and_train(post9):
    rows = ot.pool_capacity(tasks.DEFAULT_SEED, post9=post9)
    assert {(r.job, r.phrasing) for r in rows} == {(c.job, c.phrasing) for c in ot.cells()}
    for r in rows:
        assert r.candidates >= r.drawn, r
    assert min(r.headroom for r in rows) >= 2.0  # the smallest margin, seen here first


def test_a_pool_too_small_raises(monkeypatch, post9):
    big = {**ot.SPLITS["train"], "plain, tricky": 20000}
    monkeypatch.setitem(ot.SPLITS, "train", big)
    with pytest.raises(RuntimeError, match="pool too small"):
        ot.generate(tasks.DEFAULT_SEED, post9=post9)


def test_test_split_does_not_depend_on_train_size(monkeypatch, post9, splits):
    monkeypatch.setitem(ot.SPLITS, "train", {k: v // 2 for k, v in ot.SPLITS["train"].items()})
    monkeypatch.setattr(ot, "SAMPLE_SIZES", {k: v // 4 for k, v in ot.SAMPLE_SIZES.items()})
    other = ot.generate(tasks.DEFAULT_SEED, post9=post9)
    assert ot.sha256(other["test"]) == ot.sha256(splits["test"])
    assert ot.sha256(other["dev"]) == ot.sha256(splits["dev"])


def test_records_carry_the_fields(splits):
    for t in every(splits) + splits[ot.SAMPLE_SPLIT]:
        job = ot.JOBS[t["tool"]]
        assert (t["job_domain"], t["job_operation"]) == (job.domain, job.operation)
        assert t["verb_family"] == (t["tool"] if t["tool"] in ot.FAMILIES else None)
        assert t["job_class"] == (ot.EXECUTABLE if t["tool"] in tools.EXECUTABLE_NAMES
                                  else ot.DISTRACTOR)
        assert t["reading_required"] == (t["kind"] == ot.NOT_FIT
                                         and t["job_class"] == ot.EXECUTABLE)
        assert t["phrasing"] in (ot.EXPLICIT, ot.TRICKY)
        assert len(t["tools"]) == 1 and t["tools"][0] in tools.EXECUTABLE_NAMES


def test_names_never_collide_with_post9s_kinds():
    ours = {*ot.KINDS, *(s for k in ot.KINDS for s in ot.SUBKINDS[k]), ot.EXPLICIT, ot.TRICKY}
    assert not ours & {*tasks.KINDS, *tasks.SHORT_KIND.values()}


def test_fits_reference_is_the_offered_tool_and_runs(splits):
    for t in every(splits):
        if t["kind"] != ot.FITS:
            continue
        ref = t["reference"]
        assert ref["name"] == t["tool"] == t["tools"][0], t["id"]
        r = run(ref["name"], ref["arguments"])
        assert "error" not in r.result, t["id"]
        if ref["name"] in ("search_contacts", "get_calendar_event", "search_documents"):
            assert r.compared, t["id"]  # never an empty result set
        raw = '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (
            ref["name"], json.dumps(ref["arguments"]))
        c = check_reply(raw, ref)
        assert c.exact and c.execution, t["id"]


def test_fits_events_name_one_event(splits):
    unique = set(tasks._unique_titles())
    for t in every(splits):
        if t["kind"] == ot.FITS and t["tool"] in ("cancel_calendar_event",
                                                   "update_calendar_event"):
            args = t["reference"]["arguments"]
            assert args["title"] in unique or (t["tool"] == "cancel_calendar_event"
                                               and "date" in args), t["id"]
            assert "push" not in t["request"].lower()  # dropped: needs the current start


def test_doesnt_fit_pairs_really_dont_fit(splits):
    seen = Counter()
    for t in every(splits):
        if t["kind"] != ot.NOT_FIT:
            continue
        offered, job = t["tools"][0], t["tool"]
        me, need = ot.JOBS[offered], ot.JOBS[job]
        assert t["reference"] is None and job != offered, t["id"]
        assert ot.excluded_reason(offered, job) is None, t["id"]
        if t["subkind"] == ot.CROSS:
            assert need.domain != me.domain, t["id"]
            assert not ot.mentions_domain(offered, t["request"]), t["request"]
        else:
            assert t["subkind"] == ot.NEAR and job in ot.FAMILIES
            assert need.domain == me.domain and need.operation != me.operation, t["id"]
            assert ot.names_operation(job, t["request"]), t["request"]
        seen[t["subkind"]] += 1
    assert seen[ot.CROSS] and seen[ot.NEAR]


# The pairs plan 3b v3.2 names (Han's narrow reading, 2026-11-24): the offered tool
# does a direct step of the job.
NAMED_FIRST_STEP = {
    ("get_calendar_event", "update_calendar_event"), ("get_calendar_event", "cancel_calendar_event"),
    ("search_contacts", "update_contact"), ("search_contacts", "delete_contact"),
    ("search_documents", "delete_document"), ("search_documents", "rename_document"),
    ("search_documents", "share_document"), ("create_calendar_event", "update_calendar_event"),
    ("create_document", "rename_document"), ("add_contact", "update_contact"),
}
KEPT = {("cancel_calendar_event", "update_calendar_event"), ("delete_contact", "update_contact"),
        ("delete_document", "rename_document")}


def test_the_step_rule_drops_exactly_the_named_pairs():
    assert set(ot.dropped_near_pairs()) == NAMED_FIRST_STEP
    for o, j in KEPT:  # deleting is not a step of changing (one call can't edit)
        assert o in ot.near_offers(j)
    assert not ot.near_shortfall()  # every family has a near-miss pair


def test_no_near_miss_item_comes_from_an_excluded_pair(splits):
    bad = [t["id"] for t in every(splits) + splits[ot.SAMPLE_SPLIT]
           if t["subkind"] == ot.NEAR and ((t["tools"][0], t["tool"]) in NAMED_FIRST_STEP
                                           or ot.excluded_reason(t["tools"][0], t["tool"]))]
    assert not bad


def test_operation_verb_and_near_miss_balanced_per_family(splits):
    for name in splits:
        for fam in ot.FAMILIES:
            f = sum(t["tool"] == fam and t["subkind"] == ot.VERB for t in splits[name])
            n = sum(t["tool"] == fam and t["subkind"] == ot.NEAR for t in splits[name])
            assert f >= 1 and n >= 1 and abs(f - n) <= 1, (name, fam, f, n)


def test_every_family_template_names_its_operation():
    for fam in ot.FAMILIES:
        pool = ot.job_pool(fam, ot.EXPLICIT, random.Random(0))
        assert pool, fam
        assert all(ot.names_operation(fam, c.request) for c in pool), fam


def test_operation_verb_items_use_the_near_miss_templates(splits):
    for fam in ot.FAMILIES:
        verb = {t["template"] for t in splits["train"]
                if t["tool"] == fam and t["subkind"] == ot.VERB}
        near = {t["template"] for t in splits["train"]
                if t["tool"] == fam and t["subkind"] == ot.NEAR}
        assert verb & near, fam


def test_dev_covers_jobs_domains_and_families(splits):
    dev = splits["dev"]
    cross = [t for t in dev if t["subkind"] == ot.CROSS]
    assert {t["tool"] for t in cross} >= set(ot.DISTRACTOR_JOBS)
    assert {t["job_domain"] for t in cross if t["job_class"] == ot.EXECUTABLE} == {
        "math", "contacts", "calendar", "documents"}
    assert {t["tool"] for t in dev} >= set(ot.FAMILIES)


def test_cross_domain_half_executable(splits):
    for name in splits:
        c = Counter(t["job_class"] for t in splits[name] if t["subkind"] == ot.CROSS)
        assert c[ot.EXECUTABLE] == c[ot.DISTRACTOR], name


def test_shortcut_rules_on_the_reading_required_subset(splits):
    """Plan 3e: on R the job-identity rule scores 0 by construction, and Rule A should be
    near 0 (verbs balanced within each job)."""
    test = splits["test"]
    _, r = sc.score_rule(test, sc.job_identity_calls)
    assert r.d == 0.0 and len(r.jobs) == 17
    _, r = sc.score_rule(test, sc.rule_a_calls)
    assert abs(r.d) < 0.05
    assert sum(t["reading_required"] for t in test) == 375


def test_domain_identity_reaches_D_R_but_not_D_near(splits):
    """T1b review 1: R's doesn't-fit side mixes near misses (same domain) with cross-
    domain items, so a rule that reads only the tool's domain scores D_R ~ +41.5; on
    D_near (fits vs near miss per family, the domain held fixed) it scores 0. A verb
    rule and the job rule score ~0 on D_near too: only the operation moves it."""
    test = splits["test"]
    x = sc.rule_scores(test, sc.domain_identity_calls)
    assert x.reading.d > 0.35
    assert x.near.d == 0.0 and x.near.c_plus == x.near.c_minus == 1.0
    assert len(x.near.jobs) == len(ot.FAMILIES) == 14
    for rule in (sc.rule_a_calls, sc.job_identity_calls):
        assert abs(sc.rule_scores(test, rule).near.d) < 0.02


def test_call_iff_tricky_scores_little_D_R(splits):
    """T1b review 4: cross-domain items of the four tricky jobs are 3 : 1 explicit :
    tricky, as plain fits are; 'call iff tricky' (never reads the tool) scored D_R +6.0
    when cross-domain was explicit only. The two calendar families keep a gap (their
    near misses and operation-verb fits are explicit), hence < 3 points, not 0."""
    test = splits["test"]
    x = sc.rule_scores(test, sc.tricky_calls)
    assert abs(x.reading.d) < 0.03 and x.near.d == 0.0
    for j in ("calculator", "search_contacts"):
        r = x.reading.per_job[j]
        assert abs(r.c_plus - r.c_minus) < 0.05, j


def test_cross_domain_tricky_jobs_are_dealt_three_to_one(splits):
    assert [ot.tricky_part(m) for m in (0, 1, 2, 7, 8, 14, 15)] == [0, 0, 1, 2, 2, 4, 4]
    st = ot.STRATUM["cross-domain, executable"]
    assert ot.phrasings_of(st, "calculator") == [ot.EXPLICIT, ot.TRICKY]
    assert ot.phrasings_of(st, "delete_contact") == [ot.EXPLICIT]
    assert ot.phrasings_of(ot.STRATUM["near miss"], "get_calendar_event") == [ot.EXPLICIT]
    for name in ot.SPLITS:
        for job in ot.TRICKY_TOOLS:
            mine = [t for t in splits[name] if t["subkind"] == ot.CROSS and t["tool"] == job]
            tricky = sum(t["phrasing"] == ot.TRICKY for t in mine)
            assert tricky == ot.tricky_part(len(mine)), (name, job)
    # a tricky cross-domain item comes from its job's tricky pool, as plain tricky ones do
    for job in ot.TRICKY_TOOLS:
        pool = {c.request for c in ot.job_pool(job, ot.TRICKY,
                                               ot._pool_rng(tasks.DEFAULT_SEED, job, ot.TRICKY))}
        mine = [t for name in ot.SPLITS for t in splits[name] if t["subkind"] == ot.CROSS
                and t["tool"] == job and t["phrasing"] == ot.TRICKY]
        assert mine and all(t["request"] in pool for t in mine), job


def test_no_later_split_repeats_an_earlier_splits_task(splits):
    """T1b review 2: 108 test items had a one-tool train item with the same job, slot
    values and tool offered (the same task, reworded). Dev and train now skip any task
    an earlier split has; a date's written format is wording, not the task."""
    keys = {name: {ot.record_key(t) for t in splits[name]} for name in ot.SPLITS}
    assert not keys["test"] & keys["dev"]
    assert not keys["test"] & keys["train"]
    assert not keys["dev"] & keys["train"]
    assert not {ot.record_key(t) for t in splits[ot.SAMPLE_SPLIT]} & keys["test"]
    a = ot.task_key("get_calendar_event", {"when": "on 3 March", "format": 0}, "calculator")
    b = ot.task_key("get_calendar_event", {"when": "on 3 March", "format": 2}, "calculator")
    assert a == b != ot.task_key("get_calendar_event", {"when": "on 3 March"}, "add_contact")


def test_jobs_and_offers():
    assert len(ot.EXECUTABLE_JOBS) == 17 and set(ot.EXECUTABLE_JOBS) == set(tools.EXECUTABLE_NAMES)
    assert ot.DISTRACTOR_JOBS == ("send_email", "get_weather", "set_reminder", "get_stock_price",
                                  "translate_text")
    assert len(ot.FAMILIES) == 14 and "plot_graph" not in ot.JOBS
    assert "create_calendar_event" not in ot.cross_offers("set_reminder")  # stands in
    assert "search_contacts" not in ot.cross_offers("send_email")
    for job in ot.EXECUTABLE_JOBS + ot.DISTRACTOR_JOBS:
        assert ot.cross_offers(job) and job not in ot.cross_offers(job)


def test_names_operation_is_whole_words():
    assert ot.names_operation("delete_contact", "Delete Ann Lee from my contacts.")
    assert not ot.names_operation("add_contact", "Look up Maddie's address.")


def test_polynomials():
    assert ot._quadratic(2, 3) == "x^2 + 5x + 6"
    assert ot._quadratic(-4, 4) == "x^2 - 16"
    assert ot._quadratic(-1, 2) == "x^2 + x - 2"
    rng = random.Random(3)
    for _ in range(200):
        p = ot._poly(rng)
        assert "x" in p and "+ -" not in p and "  " not in p and not p.startswith("+")
        assert "error" not in tools.execute("differentiate", {"polynomial": p})


def test_step_mix():
    assert sum(ot.STEP_MIX.values()) == 100
    assert ot.step_mix() == ot.STEP_MIX
    assert ot.step_mix(25, 15) == ot.FALLBACK_MIX
    assert list(ot.STEP_MIX) == [*tasks.KINDS, *ot.KINDS]
    shares = ot.mix_shares(ot.FALLBACK_MIX)
    assert abs(sum(shares.values()) - 100) < 1e-9
    assert abs(shares[ot.FITS] - 2500 / 110) < 1e-9
    with pytest.raises(ValueError):
        ot.mix_shares({"a": 0})
    with pytest.raises(ValueError):
        ot.mix_shares({"a": 5, "b": -1})


def test_train_sample_is_stratified_and_seeded(splits):
    again = ot.train_sample(splits["train"], tasks.DEFAULT_SEED)
    assert again == splits[ot.SAMPLE_SPLIT]
    assert ot.train_sample(splits["train"], 1) != again
    ids = [t["id"] for t in again]
    assert ids == sorted(ids)


def test_write_names_files_apart_from_post9(tmp_path, splits):
    paths = ot.write(splits, tmp_path)
    assert sorted(p.name for p in paths.values()) == [
        "onetool-dev.jsonl", "onetool-test.jsonl", "onetool-train-sample.jsonl",
        "onetool-train.jsonl"]
    assert paths["onetool-test"].read_bytes() == ot.jsonl_bytes(splits["test"])


def test_cli_report_fits_the_width(tmp_path, monkeypatch, capsys):
    from tinytools import cli

    monkeypatch.setattr(cli, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(cli, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    cli.main(["tasks", "--onetool"])
    out = capsys.readouterr().out
    assert "post 9's split hashes unchanged    yes (4 of 4 match)" in out
    assert "post 9's schemas byte-identical    yes" in out
    assert "every pool covers its draw         yes" in out
    assert "largest fits - near-miss gap       1 (any family, any split)" in out
    assert "dev: every distractor job          yes" in out
    # T1c review N1: counted with a date counted once however written (pinned; the
    # generator's key keeps the written date, so these are not 0)
    assert "test items whose task is in        dev 6, train 73" in out
    assert "dev items whose task is in         train 7" in out
    assert "right answer in one-tool train     same 92, both 57, opposite 83" in out
    assert "  all                        219 (21.9%)     232 (23.2%)" in out
    assert "however worded" not in out.replace("counted once however written", "")
    for label in [*sc.RULES, *sc.DESCRIPTIVE, *sc.CHECKS]:
        assert f"  {label} " in out, label
    assert (tmp_path / "outputs" / "tasks-onetool.txt").read_text() == out
    assert not (tmp_path / "tasks" / "test.jsonl").exists()  # post 9's files untouched
    assert all(len(line) <= 74 and line == line.rstrip() for line in out.splitlines())


# -- the date-normalized key (T1c review N1) ----------------------------------------------

def test_two_writings_of_a_date_are_one_task():
    today = dt.date(2026, 3, 4)
    phrases = ot.date_phrases(today)
    keys = {ot.dated_task_key("get_calendar_event", {"when": w, "format": f},
                              "get_calendar_event", phrases)
            for f, w in enumerate(["on March 19", "on Mar 19", "on 19 March", "on March 19th",
                                   "on 2026-03-19", "on Thursday, March 19"])}
    assert keys == {("get_calendar_event|" + '{"when": "2026-03-19"}', "get_calendar_event")}
    # update's day slot has no "on"; a relative phrase maps to its date too
    assert (ot.dated_entity("update_calendar_event", {"day": "19 March", "title": "X"}, phrases)
            == ot.dated_entity("update_calendar_event", {"day": "2026-03-19", "title": "X"},
                               phrases))
    assert phrases["tomorrow"] == "2026-03-05"
    # another date, another tool offered, or a non-calendar job: not merged / not mapped
    assert (ot.dated_task_key("get_calendar_event", {"when": "on March 20"}, "calculator",
                              phrases)
            != ot.dated_task_key("get_calendar_event", {"when": "on March 19"}, "calculator",
                                 phrases))
    assert (ot.dated_task_key("get_calendar_event", {"when": "on March 19"}, "calculator",
                              phrases)
            != ot.dated_task_key("get_calendar_event", {"when": "on March 19"}, "add_contact",
                                 phrases))
    assert ot.dated_entity("get_weather", {"city": "Riga", "when": "tomorrow"}, phrases) == \
        ot.entity("get_weather", {"city": "Riga", "when": "tomorrow"})
    # the generator's key still keeps the written date (changing it changes the data)
    assert (ot.task_key("get_calendar_event", {"when": "on March 19", "format": 0}, "x")
            != ot.task_key("get_calendar_event", {"when": "on 19 March", "format": 2}, "x"))


def test_date_phrases_mean_one_day_each():
    phrases = ot.date_phrases(dt.date(2026, 3, 4))
    assert phrases["March 19"] == phrases["on March 19"] == "2026-03-19"
    assert phrases["September 5"] == "2025-09-05"  # 180 days back; 185 ahead is out
    assert "September 4" not in phrases  # 181 back and 184 ahead: both out
    with pytest.raises(ValueError):
        ot.date_phrases(dt.date(2026, 3, 4), span=200)


def test_every_date_slot_in_the_splits_maps(splits):
    for t in every(splits):
        if t["tool"] in ot.DATED_JOBS:
            ot.dated_record_key(t)  # raises KeyError on a phrase it can't read


def _item(kind, tool="search_contacts", name="Dana Lee", offered="search_contacts"):
    return {"kind": kind, "tool": tool, "slots": {"name": name}, "tools": [offered]}


def test_answer_agreement():
    fits, no = ot.FITS, ot.NOT_FIT
    train = [_item(fits, name="A"), _item(no, name="A"), _item(no, name="B"),
             _item(fits, name="C")]
    test = [_item(fits, name="A"), _item(fits, name="B"), _item(no, name="B"),
            _item(fits, name="C"), _item(fits, name="Z")]
    got = ot.answer_agreement(test, train, key=lambda t: ot.entity(t["tool"], t["slots"]))
    assert got == {ot.SAME_ANSWER: 2, ot.BOTH_ANSWERS: 1, ot.OPPOSITE_ANSWER: 1}
