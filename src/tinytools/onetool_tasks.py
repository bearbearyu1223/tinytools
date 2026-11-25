"""Post 10's one-tool tasks: one tool offered, and the request either needs it or doesn't.

Two kinds, each with two sub-kinds:

    one tool, fits          the request needs the one tool offered; right = execution
                            match (as post 9's call kinds)
        plain               post 9's five tools, with explicit values (3 parts) or
                            tricky ones (1 part: relative dates, spoken times, unit
                            conversions, contact-list spellings; four tools)
        operation verb      one of the 14 operation-family tools (:data:`FAMILIES`)
                            offered and its own operation asked for, in the words the
                            near-miss requests use ("Cancel ...", "Rename ...")
    one tool, doesn't fit   the request needs something the tool can't do; right =
                            declined (no call)
        cross-domain        another domain's job: half an executable tool's (any of the
                            17; the four with tricky values 3 : 1 explicit : tricky),
                            half a distractor's (weather, email, ...)
        near miss           the same domain, another operation family's job:
                            delete_contact offered for "add Dana Lee"

**Why it is built this way (T1's review, plan 3b v3.2).** In T1's set an operation
word (delete, change, share, symbolic math) always meant "decline", and so did a
request that none of the five tools could do: two rules that never read the offered
tool scored D = +44.6 and +72.0. So the operation jobs are real tools here
(:data:`tools.OPERATIONS`, their own registry: post 9's prompts don't change), each
operation family is asked for as often with its own tool offered (fits, operation
verb) as with another tool of its domain (near miss), +/- 1 per family in every split,
and a doesn't-fit item whose job is an executable tool carries ``reading_required``:
only reading the offered tool tells it from a fits item of the same job
(:mod:`tinytools.shortcuts` scores its rules on that subset, and on the near-miss
families, where the domain is held fixed).

**Near-miss pairs that are dropped** (:func:`excluded_reason`, printed in r2a): any
pair where the offered tool does a direct step of the job, by one rule rather than a
list: same domain, and the offered tool's operation is one of :data:`STEP_OF`'s for
the job's operation. A look-up finds the record a change, delete or share acts on
(get_calendar_event -> cancel); creating a record is a step of changing one (create
the event at the new time, then remove the old: create_calendar_event -> update).
Deleting is *not* counted as a step of changing (Han, 2026-11-24: one call can't do
the edit, so declining is right): cancel_calendar_event -> update stays. Cross-domain
pairs drop the two stand-ins post 9 knew (:data:`CROSS_EXCLUDED`).

**Every record** carries the job needed (``tool``, as post 9's no-tool-fits records
carry the tool needed), its domain and operation, its verb family (the job, if it is
one of the 14), whether it is executable or a distractor, and ``reading_required``.

**Own pools.** Templates are their own (ids start ``1t-``), and any request text post
9's splits use is skipped. For the distractor jobs, any (job, slot values) post 9's
splits use is skipped too (the entity holdout; their slot grids get new values here).
Each cell (stratum, offered tool, job, phrasing) shuffles its pool with its own seed;
cells draw split by split, **test first, then dev, then train**, skipping any request
text or (template, slots) already taken, and any task an earlier split already has
(:func:`task_key`: the same job, slot values and tool offered; T1b review 2). So the
splits are disjoint, test never depends on train's size, and a pool too small for its
demand raises. That key keeps the written date, so one date written two ways is two
tasks, and some test calendar tasks recur in train with the date written another way
(:func:`dated_task_key` counts them for r2a; with only 45 event days the skip can't
be tightened; T1c review N1).

**Phrasing** (T1b review 4): cross-domain items for the four jobs with tricky values
are dealt 3 : 1 explicit : tricky (:func:`phrasings_of`), as plain fits items are, so
"call iff the values are tricky" can't separate R's two sides.

**Sizes** (:data:`SPLITS`, by :data:`STRATA`): a seeded deal (:func:`deal`) over the
stratum's jobs, then over the tools that can be offered for each job, so every job
gets ``n // k`` or one more, and which ones get the remainder is drawn per split.
``train-sample`` is a sample of train dealt the same way.

The step mix (:data:`STEP_MIX`) is the RL sampler's fixed weight per kind; the gate's
fallback (:data:`FALLBACK_MIX`) raises one tool, fits to 25.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import random
import re
from collections import Counter
from dataclasses import dataclass

from . import tasks, tools
from .tasks import Candidate

FITS = "one tool, fits"
NOT_FIT = "one tool, doesn't fit"
KINDS = (FITS, NOT_FIT)
SHORT_KIND = {FITS: "one tool, fits", NOT_FIT: "one tool, no fit"}

PLAIN = "plain"
VERB = "operation verb"
CROSS = "cross-domain"
NEAR = "near miss"
SUBKINDS = {FITS: (PLAIN, VERB), NOT_FIT: (CROSS, NEAR)}

# How a request's values are written (post 9's two call kinds). Not "choose+fill" /
# "tricky arguments": those are post 9's kind names, and a grouping by sub-kind or
# phrasing must never merge the two posts' items.
EXPLICIT = "explicit values"
TRICKY = "tricky values"
EXECUTABLE = "executable"
DISTRACTOR = "distractor"


@dataclass(frozen=True)
class Stratum:
    name: str
    kind: str
    subkind: str
    phrasing: str = EXPLICIT
    job_class: str | None = None  # cross-domain only: the job's class


STRATA = (
    Stratum("plain, explicit", FITS, PLAIN, EXPLICIT),
    Stratum("plain, tricky", FITS, PLAIN, TRICKY),
    Stratum("operation verb", FITS, VERB),
    Stratum("cross-domain, executable", NOT_FIT, CROSS, job_class=EXECUTABLE),
    Stratum("cross-domain, distractor", NOT_FIT, CROSS, job_class=DISTRACTOR),
    Stratum("near miss", NOT_FIT, NEAR),
)
STRATUM = {s.name: s for s in STRATA}

# Sizes per split, by stratum (plan 3b v3.1/v3.2): plain 3 : 1 explicit : tricky;
# cross-domain half executable jobs, half distractors.
_SIZES = {
    "test": (188, 62, 250, 125, 125, 250),
    "dev": (15, 5, 20, 10, 10, 20),
    "train": (375, 125, 500, 250, 250, 500),
}
SPLITS = {split: dict(zip((s.name for s in STRATA), n, strict=True))
          for split, n in _SIZES.items()}
SAMPLE_SPLIT = "train-sample"
SAMPLE_SIZES = dict(zip((s.name for s in STRATA), (38, 12, 50, 25, 25, 50), strict=True))
FILE_PREFIX = "onetool-"  # data/tasks/onetool-<split>.jsonl; ids onetool-<split>-NNNN

# Post 9's split hashes (tests/test_tasks.py's PINNED_SHA256, which a test keeps equal to
# this): the one-tool kinds must leave them unchanged, and r2a prints the check.
POST9_SHA256 = {
    "test": "ef5f1ae004e76b72d10ff6f87053f93360d92c6d22c06fa98f8723b629a45cd4",
    "dev": "fabf92cbf4ea84fecef3fcffd9051a67520bcdff539f13d4744d1d5796bcb83c",
    "train": "d44888a0368a1b0a718c53b40c83c248441fd265ce3be66dee0f856db71c61ea",
    "train-sample": "fd6c706bd0e1bfe7c35c831cce11581f4f087187b44fbc6073bf05c4307c5651",
}
# Post 9's tool schemas (real tools and distractors, with their order), pinned the same
# way: sha256 of :func:`post9_schema_bytes`. The operation tools must not change them.
POST9_SCHEMAS_SHA256 = "4c14a216716dc804b633901e735660e5479def14cd2ce533b1bc050504feaff6"


def post9_schema_bytes() -> bytes:
    return json.dumps({"SCHEMAS": tools.SCHEMAS, "DISTRACTORS": tools.DISTRACTORS,
                       "order": [list(tools.SCHEMAS), list(tools.DISTRACTORS)]},
                      sort_keys=True).encode()


def subkind_counts(records: list[dict]) -> Counter:
    """(kind, sub-kind) counts of a split. Pure."""
    return Counter((t["kind"], t["subkind"]) for t in records)


def stratum_of(t: dict) -> str:
    """The stratum a record was drawn for. Pure."""
    if t["subkind"] == PLAIN:
        return "plain, tricky" if t["phrasing"] == TRICKY else "plain, explicit"
    if t["subkind"] == CROSS:
        return f"cross-domain, {t['job_class']}"
    return t["subkind"]


def counts(records: list[dict]) -> Counter:
    """Stratum counts of a split. Pure."""
    return Counter(stratum_of(t) for t in records)


# -- the step mix --------------------------------------------------------------------

STEP_MIX = {tasks.CHOOSE: 40, tasks.TRICKY: 15, tasks.NO_FIT: 15, FITS: 15, NOT_FIT: 15}
FALLBACK_MIX = {**STEP_MIX, FITS: 25}  # the gate's fallback (plan 3a): fits raised to 25


def step_mix(fits: int = 15, not_fit: int = 15) -> dict[str, int]:
    """The RL sampler's weight per kind, post 9's three kinds fixed (40 / 15 / 15)."""
    return {**STEP_MIX, FITS: fits, NOT_FIT: not_fit}


def mix_shares(weights: dict[str, int]) -> dict[str, float]:
    """Each kind's share of a step, in percent (weights need not sum to 100). Pure."""
    total = sum(weights.values())
    if total <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative with a positive sum")
    return {k: 100.0 * w / total for k, w in weights.items()}


# -- jobs: what a request needs ------------------------------------------------------


@dataclass(frozen=True)
class Job:
    name: str
    domain: str
    operation: str


_JOBS = [
    Job("calculator", "math", "arithmetic"),
    Job("differentiate", "math", "differentiate"),
    Job("integrate", "math", "integrate"),
    Job("factor_polynomial", "math", "factor"),
    Job("search_contacts", "contacts", "look up"),
    Job("add_contact", "contacts", "add"),
    Job("update_contact", "contacts", "change"),
    Job("delete_contact", "contacts", "delete"),
    Job("get_calendar_event", "calendar", "look up"),
    Job("create_calendar_event", "calendar", "add"),
    Job("update_calendar_event", "calendar", "change"),
    Job("cancel_calendar_event", "calendar", "delete"),
    Job("search_documents", "documents", "look up"),
    Job("create_document", "documents", "add"),
    Job("rename_document", "documents", "change"),
    Job("delete_document", "documents", "delete"),
    Job("share_document", "documents", "share"),
    # distractor jobs (schemas only; cross-domain, always "decline")
    Job("send_email", "email", "send"),
    Job("get_weather", "weather", "look up"),
    Job("set_reminder", "reminders", "add"),
    Job("get_stock_price", "stocks", "look up"),
    Job("translate_text", "language", "translate"),
]
JOBS = {j.name: j for j in _JOBS}
EXECUTABLE_JOBS = tuple(j.name for j in _JOBS if j.name in tools.EXECUTABLE_NAMES)
DISTRACTOR_JOBS = tuple(j.name for j in _JOBS if j.name not in tools.EXECUTABLE_NAMES)
OFFERED = EXECUTABLE_JOBS  # any tool that runs can be the one offered
PLAIN_TOOLS = tools.TOOL_NAMES
TRICKY_TOOLS = tasks.KIND_TOOLS[tasks.TRICKY]
# The operation families (D_near: operation-verb fits vs near miss): the twelve
# operation tools plus post 9's two calendar tools whose requests name an operation
# (create, look up an event).
FAMILIES = tuple(j for j in EXECUTABLE_JOBS
                 if j in tools.OPERATION_NAMES
                 or j in ("get_calendar_event", "create_calendar_event"))

# The direct steps of an operation: a job whose operation is the key can be done in
# steps by a tool of the same domain whose operation is in the value.
STEP_OF = {"change": ("look up", "add"), "delete": ("look up",), "share": ("look up",)}
STEP_RULE = ("same domain, and the tool offered does a direct step of the job: a "
             "look-up finds the record a change, delete or share acts on; creating "
             "a record is a step of changing one (create anew, drop the old)")
# Cross-domain stand-ins (post 9's SUBSTITUTES and T1's): a calendar event is a fair
# reminder; a contact search finds the address an email needs.
CROSS_EXCLUDED = {("create_calendar_event", "set_reminder"), ("search_contacts", "send_email")}


def excluded_reason(offered: str, job: str) -> str | None:
    """Why (offered, job) never makes a doesn't-fit item, or ``None``. Pure."""
    me, need = JOBS[offered], JOBS[job]
    if offered == job or (me.domain, me.operation) == (need.domain, need.operation):
        return "the same job"
    if (offered, job) in CROSS_EXCLUDED or offered in tasks.SUBSTITUTES.get(job, ()):
        return "stands in for the job"
    if me.domain == need.domain and me.operation in STEP_OF.get(need.operation, ()):
        return "does a step of the job"
    return None


def near_offers(job: str) -> list[str]:
    """The tools that can be offered for a near miss needing ``job``. Pure."""
    me = JOBS[job]
    return [o for o in OFFERED if JOBS[o].domain == me.domain and excluded_reason(o, job) is None]


def cross_offers(job: str) -> list[str]:
    """The tools that can be offered for a cross-domain item needing ``job``. Pure."""
    me = JOBS[job]
    return [o for o in OFFERED if JOBS[o].domain != me.domain and excluded_reason(o, job) is None]


def dropped_near_pairs() -> list[tuple[str, str]]:
    """(offered, job) pairs of the same domain dropped by the step rule. Pure."""
    return [(o, j) for j in FAMILIES for o in OFFERED
            if JOBS[o].domain == JOBS[j].domain and o != j
            and excluded_reason(o, j) == "does a step of the job"]


def near_shortfall() -> list[str]:
    """Families with no valid near-miss pair (their share would go to the others)."""
    return [j for j in FAMILIES if not near_offers(j)]


def jobs_of(stratum: Stratum) -> list[str]:
    """The jobs a stratum deals over, in a fixed order. Pure."""
    if stratum.subkind == PLAIN:
        return list(TRICKY_TOOLS if stratum.phrasing == TRICKY else PLAIN_TOOLS)
    if stratum.subkind == VERB:
        return list(FAMILIES)
    if stratum.subkind == NEAR:
        return [j for j in FAMILIES if near_offers(j)]
    return list(EXECUTABLE_JOBS if stratum.job_class == EXECUTABLE else DISTRACTOR_JOBS)


def offers_of(stratum: Stratum, job: str) -> list[str]:
    if stratum.kind == FITS:
        return [job]
    return near_offers(job) if stratum.subkind == NEAR else cross_offers(job)


# The words a near-miss (and operation-verb) request uses for its operation; every
# template of a family contains one (a test checks).
OPERATION_WORDS = {
    "differentiate": ("derivative", "differentiate"),
    "integrate": ("integral", "integrate", "antiderivative"),
    "factor_polynomial": ("factor", "factorise", "factorize"),
    "add_contact": ("add", "save", "create"),
    "update_contact": ("change", "update", "move"),
    "delete_contact": ("delete", "remove", "take"),
    "cancel_calendar_event": ("cancel", "delete", "remove", "wipe", "drop", "clear"),
    "update_calendar_event": ("move", "reschedule", "change"),
    "create_calendar_event": ("add", "schedule", "book", "make", "need"),
    "get_calendar_event": ("what", "which", "check", "is", "tell", "look", "when", "find"),
    "create_document": ("create", "start", "make"),
    "delete_document": ("delete", "remove", "trash", "rid"),
    "rename_document": ("rename", "change"),
    "share_document": ("share", "give"),
}

# A cross-domain request never mentions the offered tool's own domain by name.
DOMAIN_WORDS = {"calendar": ("calendar", "schedule"), "contacts": ("contact", "address book"),
                "documents": ("document", "doc ", "docs"), "math": ("calculat",)}


def names_operation(job: str, request: str) -> bool:
    """Whether ``request`` uses one of ``job``'s operation words, as a whole word. Pure."""
    words = set(re.findall(r"[a-z]+", request.lower()))
    return bool(words & set(OPERATION_WORDS[job]))


def mentions_domain(offered: str, request: str) -> bool:
    words = DOMAIN_WORDS.get(JOBS[offered].domain, ())
    text = request.lower() + " "
    return any(w in text for w in words)


# -- templates -------------------------------------------------------------------------

CALC_PLAIN = [
    ("Work out {a} times {b}.", "{a} * {b}"),
    ("Multiply {a} by {b} for me.", "{a} * {b}"),
    ("How much is {a} plus {b}?", "{a} + {b}"),
    ("Take {b} away from {a}.", "{a} - {b}"),
    ("What do I get if I divide {a} by {b}?", "{a} / {b}"),
    ("Add up {a}, {b} and {c}.", "{a} + {b} + {c}"),
    ("What's {s} cubed?", "{s} ** 3"),
    ("Calculate {a} times the sum of {b} and {c}.", "{a} * ({b} + {c})"),
    ("What is {a} minus {b} minus {c}?", "{a} - {b} - {c}"),
    ("Raise {s} to the power {n}.", "{s} ** {n}"),
]
CALC_UNITS = [
    ("How many minutes are in {d} days?", "{d} * 1440"),
    ("How many hours is {m} minutes?", "{m} / 60"),
    ("How many weeks is {d} days?", "{d} / 7"),
    ("Convert {kg} kilograms into grams.", "{kg} * 1000"),
    ("How many millimeters are in {cm} centimeters?", "{cm} * 10"),
    ("What's {p} percent of {x}?", "{x} * {p} / 100"),
    ("A {x} dollar jacket is {p}% off. How many dollars do I save?", "{x} * {p} / 100"),
    ("How many meters is {km} km?", "{km} * 1000"),
    ("How many minutes is {h2} hours and {m2} minutes?", "{h2} * 60 + {m2}"),
    ("At {rate} dollars to the euro, what are {eur} euros in dollars?", "{eur} * {rate}"),
    ("How many seconds are there in {h} hours?", "{h} * 3600"),
    ("How many days do {w} weeks make?", "{w} * 7"),
]
CONTACT_TEMPLATES = [
    "Find {name}'s phone number in my contacts.",
    "What email do I have for {name}?",
    "Check my address book for {name}.",
    "Look {name} up in my contacts.",
    "What team does {name} work on?",
    "Bring up {name}'s contact info.",
    "Do I have an email address for {name}?",
    "Search the contacts for {name}, please.",
    "What's the best number to reach {name}?",
    "Get {name}'s details from my contacts.",
]
DAY_TEMPLATES = [
    "Which events are on my calendar {when}?",
    "Check my calendar {when}.",
    "What's happening {when}? Check my calendar.",
    "Is anything scheduled {when}?",
    "Tell me what meetings I have {when}.",
    "Look at my calendar {when} for me.",
]
TITLE_TEMPLATES = [
    'What day is "{title}"?',
    "When does the {title} start?",
    'Check when "{title}" is on my calendar.',
    "Find the date of the {title}.",
    'Look up the "{title}" event.',
]
CREATE_PLAIN = [
    'Add an event "{title}" {when} at {time} for {dur} minutes{with_}.',
    'Please schedule "{title}" {when} at {time}, {dur} minutes{with_}.',
    'Can you book "{title}" {when} at {time} for {dur} minutes{with_}?',
    'Make a calendar event "{title}" {when} at {time} lasting {dur} minutes{with_}.',
    'I need "{title}" on my calendar {when} at {time} for {dur} minutes{with_}.',
]
CREATE_SPOKEN = [
    'Add an event "{title}" {when} at {time} for {dur}{with_}.',
    'Please schedule "{title}" {when} at {time}; it runs {dur}{with_}.',
    'Can you book "{title}" {when} at {time} for {dur}{with_}?',
    'Make a calendar event "{title}" {when} at {time} lasting {dur}{with_}.',
    'I need "{title}" on my calendar {when} at {time} for {dur}{with_}.',
]
DOC_TEMPLATES = [
    'Find me the "{title}" document.',
    "Search the shared docs for the {title}.",
    "Where's the {title}?",
    "Can you locate the {title}?",
    'Look for a document titled "{title}".',
    "Bring up the {title}.",
    "I'm looking for the {title}.",
    'Pull up "{title}" from the documents.',
    "Do we have a {title} document?",
    "Find the doc called {title}.",
]
# Distractor jobs. Their slot grids get new values beside post 9's, because (job, slot
# values) post 9 used are held out (stock prices: post 9 used every company it had).
WEATHER_TEMPLATES = ["Is it going to be sunny in {city} {when}?",
                     "Tell me the temperature in {city} {when}.",
                     "Do I need a jacket in {city} {when}?",
                     "Check the weather in {city} {when}."]
EMAIL_TEMPLATES = ["Send an email to {name} letting them know {msg}.",
                   "Drop {name} an email: {msg}.",
                   "Can you email {name} that {msg}?"]
REMIND_TEMPLATES = ["Ping me {when} to {task}.", "I need a reminder {when} to {task}.",
                    "Don't let me forget to {task} {when}."]
STOCK_TEMPLATES = ["What's the share price of {company} right now?",
                   "How much is one share of {company} worth today?",
                   "Look up {company}'s latest stock price.",
                   "Where is {company} stock trading now?",
                   "Give me a quote for {company} shares.",
                   "What's {company} trading at this morning?"]
TRANSLATE_TEMPLATES = ["Translate '{phrase}' to {lang}, please.",
                       "Can you put '{phrase}' into {lang}?",
                       "What is the {lang} for '{phrase}'?"]
NEW_CITIES = ["Helsinki", "Accra", "Bogota", "Kyoto", "Perth", "Quito", "Riga", "Tunis",
              "Hobart", "Porto", "Bergen", "Daegu", "Cusco", "Malmo", "Graz"]
NEW_COMPANIES = ["Atlassian", "Dropbox", "Etsy", "Zoom", "Snap", "Pinterest", "Roku", "Lyft",
                 "DoorDash", "Unilever", "Nestle", "Heineken", "Ferrari", "Volvo", "Nintendo",
                 "Honda", "Panasonic", "Philips", "Ericsson", "Nokia", "Adidas", "Puma",
                 "Danone", "Carrefour", "Rakuten", "Infosys", "Baidu", "Tencent", "Alibaba",
                 "Xiaomi"]
NEW_PHRASES = ["the train is delayed", "where is the museum", "I lost my keys",
               "the room is too cold", "two tickets please", "is breakfast included",
               "my flight was cancelled", "where can I park", "the food was excellent",
               "could you speak more slowly"]
NEW_REMIND_TASKS = ["return the library books", "book a haircut", "call my sister",
                    "order more printer ink", "pay the electricity bill", "pack my gym bag",
                    "move the car", "update my passwords", "reply to the landlord",
                    "pick up the prescription"]

# Operation jobs: the near-miss templates (also the operation-verb fits').
CANCEL_TEMPLATES = ['Cancel the "{title}" event, please.',
                    'Please delete the "{title}" event.',
                    'Remove "{title}" from my calendar for good.',
                    'Wipe "{title}" off the calendar.',
                    'Drop "{title}" from my calendar entirely.',
                    'Clear "{title}" off my schedule.']
CANCEL_DATED_TEMPLATES = ['Cancel the "{title}" {when}.',
                          'Delete the "{title}" event {when}.',
                          'Remove "{title}" {when} from my calendar.']
UPDATE_TIME_TEMPLATES = ['Move "{title}" to {time}.',
                         'Change the start time of "{title}" to {time}.',
                         'Reschedule "{title}" to start at {time}.']
UPDATE_DATE_TEMPLATES = ['Move "{title}" to {day}.',
                         'Reschedule "{title}" for {day}.']
# Not used: "Push {title} back by {dur}". Its reference needs the event's current start,
# which a request offering one tool (update) gives the model no way to read: dropped.
DIFF_TEMPLATES = ["Find the derivative of {poly}.", "Differentiate {poly} with respect to x.",
                  "What's the derivative of f(x) = {poly}?"]
INTEGRAL_TEMPLATES = ["Find the indefinite integral of {poly}.",
                      "Integrate {poly} with respect to x.",
                      "What's the antiderivative of {poly}?"]
FACTOR_TEMPLATES = ["Factor {quad}.", "Factor the polynomial {quad} into two brackets.",
                    "Can you factor {quad}?"]
ADD_CONTACT_TEMPLATES = ["Add {new} to my contacts with the number {phone}.",
                         "Save {new} as a new contact, email {email}.",
                         "Create a new contact: {new}, phone {phone}."]
UPDATE_CONTACT_TEMPLATES = ["Change {name}'s phone number to {phone}.",
                            "Update {name}'s email address to {email}.",
                            "Move {name} to the {team} team in my contacts.",
                            "Update the team for {name} to {team}."]
DELETE_CONTACT_TEMPLATES = ["Delete {name} from my contacts.",
                            "Remove {name} from my address book.",
                            "Take {name} out of my contacts list.",
                            "Please delete the contact {name}.",
                            "Remove {name} from my contacts, please."]
CREATE_DOC_TEMPLATES = ['Create a new document called "{new}".',
                        'Start a blank doc titled "{new}".',
                        'Make a new document named "{new}".',
                        'Please create a document titled "{new}".']
DELETE_DOC_TEMPLATES = ['Delete the "{title}" document.',
                        'Remove "{title}" from the shared documents.',
                        'Trash the "{title}" doc.',
                        'Please delete the document called "{title}".',
                        'Get rid of the "{title}" document.']
RENAME_DOC_TEMPLATES = ['Rename "{title}" to "{new}".',
                        'Change the title of "{title}" to "{new}".',
                        'Rename the document "{title}" to "{new}".']
SHARE_DOC_TEMPLATES = ['Share the "{title}" document with {name}.',
                       'Give {name} edit access to "{title}".',
                       'Share "{title}" with {name}, please.']

NEW_FIRST = ["Alma", "Bruno", "Celia", "Dario", "Elif", "Farid", "Greta", "Hamid", "Ines",
             "Jonas", "Kira", "Luca", "Maren", "Nils", "Odile", "Pavel", "Rania", "Soren",
             "Talia", "Viktor"]
NEW_LAST = ["Abbott", "Barros", "Castillo", "Dunn", "Eriksen", "Ferreira", "Gill", "Horvat",
            "Ibsen", "Jaramillo", "Kowalski", "Lindgren", "Mahler", "Nakamura", "Okafor",
            "Pereira", "Quinn", "Rahman", "Sato", "Tanaka"]
NEW_DOC_TOPICS = ["vendor scorecard", "hiring retrospective", "launch checklist",
                  "data retention plan", "support playbook", "pricing FAQ", "office survey",
                  "accessibility plan", "team charter", "release notes", "travel budget",
                  "security roadmap", "mentoring guide", "sales playbook", "brand audit",
                  "customer journey map", "on-call handbook", "event plan", "training plan",
                  "partner FAQ"]
TEAMS = ["Design", "Marketing", "Sales", "Finance", "Engineering", "Support", "Legal",
         "Operations"]


# -- pools -----------------------------------------------------------------------------


def _grid(prefix: str, templates: list[str], reference=None, **axes) -> list[Candidate]:
    """Every (template, slot values); ``reference(slots)`` gives the call, if any."""
    keys = list(axes)
    out = []
    for i, t in enumerate(templates):
        used = [k for k in keys if "{" + k + "}" in t]
        for combo in itertools.product(*(axes[k] for k in used)):
            slots = dict(zip(used, combo, strict=True))
            out.append(Candidate(f"{prefix}-{i}", slots, t.format(**slots),
                                 reference(slots) if reference else None))
    return out


def _contacts(tricky: bool) -> list[Candidate]:
    return [Candidate(f"1t-contact-{i}", {"name": n}, t.format(name=n),
                      tasks._call("search_contacts", query=n))
            for i, t in enumerate(CONTACT_TEMPLATES) for n in tasks._names(tricky)]


def _get_events(tricky: bool) -> list[Candidate]:
    days = tasks._event_days()
    out = []
    if tricky:
        for phrase, day in tasks.relative_dates(tools.now().date()):
            if day.isoformat() not in days:
                continue  # an empty result would match any query that finds nothing
            for i, t in enumerate(DAY_TEMPLATES):
                out.append(Candidate(f"1t-day-{i}", {"when": phrase}, t.format(when=phrase),
                                     tasks._call("get_calendar_event", date=day.isoformat())))
        return out
    for iso in sorted(days):
        for j, phrase in enumerate(tasks.explicit_dates(dt.date.fromisoformat(iso))):
            for i, t in enumerate(DAY_TEMPLATES):
                out.append(Candidate(f"1t-day-{i}", {"when": phrase, "format": j},
                                     t.format(when=phrase),
                                     tasks._call("get_calendar_event", date=iso)))
    for title in tasks._unique_titles():
        for i, t in enumerate(TITLE_TEMPLATES):
            out.append(Candidate(f"1t-title-{i}", {"title": title}, t.format(title=title),
                                 tasks._call("get_calendar_event", title=title)))
    return out


def _documents() -> list[Candidate]:
    return [Candidate(f"1t-doc-{i}", {"title": d["title"]}, t.format(title=d["title"]),
                      tasks._call("search_documents", query=d["title"]))
            for i, t in enumerate(DOC_TEMPLATES) for d in tools.documents()]


def _poly(rng: random.Random) -> str:
    """A polynomial in x, degree 2 or 3, written the way a person types it."""
    deg = rng.choice([2, 2, 3])
    coeffs = [rng.randint(1, 9)] + [rng.choice([0, *range(-20, 21)]) for _ in range(deg)]
    terms = []
    for power, c in zip(range(deg, -1, -1), coeffs, strict=True):
        if c == 0:
            continue
        mag = abs(c)
        body = ("" if mag == 1 and power else str(mag)) + ("x" if power else "") + \
            (f"^{power}" if power > 1 else "")
        terms.append((c < 0, body))
    text = ("-" if terms[0][0] else "") + terms[0][1]
    for neg, body in terms[1:]:
        text += (" - " if neg else " + ") + body
    return text


def _quadratic(r1: int, r2: int) -> str:
    """(x + r1)(x + r2), multiplied out. Pure."""
    b, c = r1 + r2, r1 * r2
    text = "x^2"
    if b:
        text += (" - " if b < 0 else " + ") + ("" if abs(b) == 1 else str(abs(b))) + "x"
    if c:
        text += (" - " if c < 0 else " + ") + str(abs(c))
    return text


def _math_pool(job: str, rng: random.Random, size: int = 3000) -> list[Candidate]:
    if job == "factor_polynomial":
        roots = [r for r in range(-12, 13) if r]
        quads = sorted({_quadratic(a, b) for a in roots for b in roots if a <= b})
        return _grid("1t-factor", FACTOR_TEMPLATES,
                     lambda s: tasks._call("factor_polynomial", polynomial=s["quad"]), quad=quads)
    templates = {"differentiate": DIFF_TEMPLATES, "integrate": INTEGRAL_TEMPLATES}[job]
    out = []
    for _ in range(size):
        i = rng.randrange(len(templates))
        poly = _poly(rng)
        out.append(Candidate(f"1t-{job}-{i}", {"poly": poly}, templates[i].format(poly=poly),
                             tasks._call(job, polynomial=poly)))
    return out


def _phone(rng: random.Random) -> str:
    return f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def _contact_op_pool(job: str, rng: random.Random, size: int = 3000) -> list[Candidate]:
    contacts = {c["name"]: c for c in tools.contacts()}
    if job == "delete_contact":
        return _grid("1t-delete-contact", DELETE_CONTACT_TEMPLATES,
                     lambda s: {"name": job, "arguments": {"name": s["name"]}},
                     name=sorted(contacts))
    new = [f"{f} {last}" for f in NEW_FIRST for last in NEW_LAST if f"{f} {last}" not in contacts]
    templates = ADD_CONTACT_TEMPLATES if job == "add_contact" else UPDATE_CONTACT_TEMPLATES
    names = new if job == "add_contact" else tasks._names(False)
    who = "new" if job == "add_contact" else "name"
    out = []
    for _ in range(size):
        i = rng.randrange(len(templates))
        name = rng.choice(names)
        first, last = name.lower().split(" ", 1)
        values = {"phone": _phone(rng),
                  "email": f"{first}.{last.replace(' ', '')}@"
                           f"{rng.choice(['mail.com', 'proton.me', 'work.org'])}",
                  "team": rng.choice(TEAMS)}
        used = {k: v for k, v in values.items() if "{" + k + "}" in templates[i]}
        if job == "update_contact" and used.get("team") == contacts[name]["team"]:
            continue  # no change: not a request anyone makes
        slots = {who: name, **used}
        ref = {"name": job, "arguments": {"name": name, **used}}  # _call's own 1st arg is name
        out.append(Candidate(f"1t-{job}-{i}", slots, templates[i].format(**slots), ref))
    return out


def _new_doc_titles() -> list[str]:
    existing = [d["title"] for d in tools.documents()]
    out = [f"Q{q} {t}" for q in range(1, 5) for t in NEW_DOC_TOPICS]
    return [t for t in out if not any(tools._same_title(t, e) for e in existing)]


def _doc_titles() -> list[str]:
    return [d["title"] for d in tools.documents()]


def _event_pool(job: str) -> list[Candidate]:
    """cancel / update: fits must name one event, so titles that occur once
    (``tasks._unique_titles``), plus, for cancel, a repeated title with its date."""
    events = tools.calendar()
    unique = set(tasks._unique_titles())
    out = []
    if job == "cancel_calendar_event":
        out += _grid("1t-cancel", CANCEL_TEMPLATES,
                     lambda s: tasks._call(job, title=s["title"]), title=sorted(unique))
        for e in sorted(events, key=lambda e: (e["title"], e["date"])):
            if e["title"] in unique:
                continue
            day = dt.date.fromisoformat(e["date"])
            for j, phrase in enumerate(tasks.explicit_dates(day)):
                for i, t in enumerate(CANCEL_DATED_TEMPLATES):
                    slots = {"title": e["title"], "when": phrase, "format": j}
                    out.append(Candidate(f"1t-cancel-dated-{i}", slots,
                                         t.format(title=e["title"], when=phrase),
                                         tasks._call(job, title=e["title"], date=e["date"])))
        return out
    today = tools.now().date()
    days = [today + dt.timedelta(k) for k in range(1, 43)]
    for e in sorted((e for e in events if e["title"] in unique), key=lambda e: e["title"]):
        for i, t in enumerate(UPDATE_TIME_TEMPLATES):
            for time in tasks.TIMES_24:
                if time == e["start_time"]:
                    continue
                out.append(Candidate(f"1t-update-time-{i}", {"title": e["title"], "time": time},
                                     t.format(title=e["title"], time=time),
                                     tasks._call(job, title=e["title"], start_time=time)))
        for i, t in enumerate(UPDATE_DATE_TEMPLATES):
            for d in days:
                if d.isoformat() == e["date"]:
                    continue
                for j, phrase in enumerate(tasks.explicit_dates(d)):
                    phrase = phrase.removeprefix("on ")
                    out.append(Candidate(f"1t-update-date-{i}",
                                         {"title": e["title"], "day": phrase, "format": j},
                                         t.format(title=e["title"], day=phrase),
                                         tasks._call(job, title=e["title"], date=d.isoformat())))
    return out


def job_pool(job: str, phrasing: str, rng: random.Random) -> list[Candidate]:
    """Every candidate request for ``job`` (``phrasing``: explicit values or tricky args).

    Executable jobs' candidates carry their reference call (used when the tool is
    offered); distractors' have none. ``rng`` drives the pools drawn at random
    (calculator, create, differentiate, integrate, add and update contact); the
    others are fixed grids.
    """
    tricky = phrasing == TRICKY
    if tricky and job not in TRICKY_TOOLS:
        raise ValueError(f"no tricky phrasing for {job}")
    if job == "calculator":
        return tasks.calculator_pool(tricky, rng, templates=CALC_UNITS if tricky else CALC_PLAIN,
                                     prefix="1t-calc")
    if job == "search_contacts":
        return _contacts(tricky)
    if job == "get_calendar_event":
        return _get_events(tricky)
    if job == "create_calendar_event":
        return tasks.create_pool(tricky, rng, templates=CREATE_SPOKEN if tricky else CREATE_PLAIN,
                                 prefix="1t-create")
    if job == "search_documents":
        return _documents()
    if job == "get_weather":
        return _grid("1t-weather", WEATHER_TEMPLATES, city=tasks.CITIES + NEW_CITIES,
                     when=tasks.WEATHER_WHEN)
    if job == "send_email":
        return _grid("1t-email", EMAIL_TEMPLATES, name=tasks._names(False), msg=tasks.EMAIL_MSGS)
    if job == "set_reminder":
        return _grid("1t-remind", REMIND_TEMPLATES, task=tasks.REMIND_TASKS + NEW_REMIND_TASKS,
                     when=tasks.REMIND_WHEN)
    if job == "get_stock_price":
        return _grid("1t-stock", STOCK_TEMPLATES, company=tasks.COMPANIES + NEW_COMPANIES)
    if job == "translate_text":
        return _grid("1t-translate", TRANSLATE_TEMPLATES, phrase=tasks.PHRASES + NEW_PHRASES,
                     lang=tasks.LANGUAGES)
    if job in ("cancel_calendar_event", "update_calendar_event"):
        return _event_pool(job)
    if job in ("differentiate", "integrate", "factor_polynomial"):
        return _math_pool(job, rng)
    if job in ("add_contact", "update_contact", "delete_contact"):
        return _contact_op_pool(job, rng)
    if job == "create_document":
        return _grid("1t-create-doc", CREATE_DOC_TEMPLATES,
                     lambda s: tasks._call(job, title=s["new"]), new=_new_doc_titles())
    if job == "delete_document":
        return _grid("1t-delete-doc", DELETE_DOC_TEMPLATES,
                     lambda s: tasks._call(job, title=s["title"]), title=_doc_titles())
    if job == "rename_document":
        return _grid("1t-rename-doc", RENAME_DOC_TEMPLATES,
                     lambda s: tasks._call(job, title=s["title"], new_title=s["new"]),
                     title=_doc_titles(), new=_new_doc_titles())
    if job == "share_document":
        return _grid("1t-share-doc", SHARE_DOC_TEMPLATES,
                     lambda s: tasks._call(job, title=s["title"], contact=s["name"]),
                     title=_doc_titles(), name=tasks._names(False))
    raise ValueError(job)


# -- entities: (job, slot values), for the holdout and the overlap count ---------------

# Post 9's delete-event requests name events by title, as cancel's do: the same entity.
ENTITY_ALIAS = {"cancel_calendar_event": "delete_calendar_event"}


def entity(job: str, slots: dict) -> str:
    return ENTITY_ALIAS.get(job, job) + "|" + json.dumps(slots, sort_keys=True)


def post9_entities(post9: dict[str, list[dict]], splits=tuple(tasks.SPLITS)) -> set[str]:
    """(job, slot values) of post 9's records (a record's ``tool`` is the job it needs)."""
    return {entity(t["tool"], t["slots"]) for s in splits for t in post9[s]}


def task_key(job: str, slots: dict, offered: str) -> tuple[str, str]:
    """(job, slot values, tool offered): the key the generator skips on. It drops a
    date's format id (slot ``format``) but keeps the written date in ``when`` / ``day``,
    so "on March 19" and "on 19 March" key apart (T1c review N1); the r2a report counts
    overlaps with :func:`dated_task_key` instead. Kept as is: changing it changes the
    data. Pure."""
    return entity(job, {k: v for k, v in slots.items() if k != "format"}), offered


def record_key(t: dict) -> tuple[str, str]:
    return task_key(t["tool"], t["slots"], t["tools"][0])


# -- the same task, a date counted once however written (T1c review N1) ------------------

DATED_JOBS = ("get_calendar_event", "create_calendar_event", "cancel_calendar_event",
              "update_calendar_event")
DATE_SLOTS = ("when", "day")


def date_phrases(today: dt.date, span: int = 180) -> dict[str, str]:
    """Every date phrase the calendar pools write, to the ISO date it means: the
    explicit forms (with and without "on") of each day within ``span`` days of
    ``today``, and the relative ones (:func:`tasks.relative_dates`). The window is under
    a year, so "March 19" means one day. Raises if a phrase would mean two. Pure."""
    out: dict[str, str] = {}

    def put(phrase: str, iso: str) -> None:
        if out.setdefault(phrase, iso) != iso:
            raise ValueError(f"{phrase!r} means {out[phrase]} and {iso}")
    for k in range(-span, span + 1):
        d = today + dt.timedelta(k)
        for p in tasks.explicit_dates(d):
            put(p, d.isoformat())
            put(p.removeprefix("on "), d.isoformat())
    for p, d in tasks.relative_dates(today):
        put(p, d.isoformat())
    return out


_DATE_PHRASES: dict[str, str] = {}


def _phrases() -> dict[str, str]:
    if not _DATE_PHRASES:
        _DATE_PHRASES.update(date_phrases(tools.now().date()))
    return _DATE_PHRASES


def dated_entity(job: str, slots: dict, phrases: dict[str, str] | None = None) -> str:
    """:func:`entity` with a date counted once however written: ``format`` dropped and,
    for a calendar job, the date phrase in ``when`` / ``day`` replaced by the ISO date
    the reference call uses. Pure given ``phrases`` (default: the clock's)."""
    phrases = _phrases() if phrases is None else phrases
    s = {k: v for k, v in slots.items() if k != "format"}
    if job in DATED_JOBS:
        for k in DATE_SLOTS:
            if k in s:
                s[k] = phrases[s[k]]
    return entity(job, s)


def dated_task_key(job: str, slots: dict, offered: str,
                   phrases: dict[str, str] | None = None) -> tuple[str, str]:
    """(job, slot values with the date as ISO, tool offered). Pure given ``phrases``."""
    return dated_entity(job, slots, phrases), offered


def dated_record_key(t: dict) -> tuple[str, str]:
    return dated_task_key(t["tool"], t["slots"], t["tools"][0])


def dated_record_entity(t: dict) -> str:
    return dated_entity(t["tool"], t["slots"])


SAME_ANSWER, BOTH_ANSWERS, OPPOSITE_ANSWER = "same", "both", "opposite"


def answer_agreement(test: list[dict], train: list[dict], key=dated_record_entity) -> Counter:
    """For each test item whose ``key`` (default: job and slot values, a date counted
    once) occurs in train: do those train items have the same right answer (call or
    decline, i.e. the same ``kind``) as the test item, some each way, or all the
    opposite? Counts of :data:`SAME_ANSWER`, :data:`BOTH_ANSWERS`,
    :data:`OPPOSITE_ANSWER`. Pure."""
    kinds: dict[str, set[str]] = {}
    for t in train:
        kinds.setdefault(key(t), set()).add(t["kind"])
    out = Counter({SAME_ANSWER: 0, BOTH_ANSWERS: 0, OPPOSITE_ANSWER: 0})
    for t in test:
        seen = kinds.get(key(t))
        if seen is None:
            continue
        out[BOTH_ANSWERS if len(seen) > 1 else
            SAME_ANSWER if t["kind"] in seen else OPPOSITE_ANSWER] += 1
    return out


def post9_texts(post9: dict[str, list[dict]]) -> set[str]:
    return {t["request"] for s in tasks.SPLITS for t in post9[s]}


# -- cells and sizes ---------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    stratum: str
    offered: str
    job: str
    phrasing: str = ""  # the stratum's, unless given (cross-domain's tricky cells)

    def __post_init__(self):
        if not self.phrasing:
            object.__setattr__(self, "phrasing", STRATUM[self.stratum].phrasing)


def phrasings_of(stratum: Stratum, job: str) -> list[str]:
    """The phrasings a stratum's items for ``job`` are drawn in. Pure.

    Cross-domain items for the four jobs with tricky values (calculator, search_contacts,
    get and create_calendar_event) come in both, 3 : 1 explicit : tricky as the fits
    side's plain items do (:func:`tricky_part`), so R's two sides of those jobs differ
    in the tool offered and not in how the request is written (T1b review 4)."""
    if stratum.subkind == CROSS and stratum.job_class == EXECUTABLE and job in TRICKY_TOOLS:
        return [EXPLICIT, TRICKY]
    return [stratum.phrasing]


def tricky_part(m: int) -> int:
    """The tricky share of ``m`` cross-domain items, 3 : 1 explicit : tricky: m / 4
    rounded half up (7 -> 2, 8 -> 2, 15 -> 4, 2 -> 1, 1 -> 0). Pure."""
    return (m + 2) // 4


def deal(total: int, n: int, rng: random.Random) -> list[int]:
    """``total`` over ``n`` slots: ``total // n`` each, and one more for ``total % n`` slots
    drawn by ``rng``. Pure given the rng."""
    if n <= 0:
        raise ValueError("nothing to deal over")
    extra = set(rng.sample(range(n), total % n))
    return [total // n + (1 if i in extra else 0) for i in range(n)]


def cells() -> list[Cell]:
    """Every (stratum, offered tool, job, phrasing), in the order the generator draws."""
    return [Cell(st.name, o, job, ph) for st in STRATA for job in jobs_of(st)
            for ph in phrasings_of(st, job) for o in offers_of(st, job)]


def cell_sizes(sizes: dict[str, int], seed: int, split: str) -> dict[Cell, int]:
    """How many items each cell gives one split: a seeded :func:`deal` over the
    stratum's jobs, then over the tools that can be offered for each job."""
    out: dict[Cell, int] = {}
    for st in STRATA:
        jobs = jobs_of(st)
        rng = random.Random(f"{seed}:onetool:deal:{split}:{st.name}")
        for job, m in zip(jobs, deal(sizes[st.name], len(jobs), rng), strict=True):
            offers = offers_of(st, job)
            phrasings = phrasings_of(st, job)
            parts = [m] if len(phrasings) == 1 else [m - tricky_part(m), tricky_part(m)]
            for ph, part in zip(phrasings, parts, strict=True):
                tag = "" if ph == st.phrasing else f":{ph}"
                sub = random.Random(f"{seed}:onetool:deal:{split}:{st.name}:{job}{tag}")
                for o, k in zip(offers, deal(part, len(offers), sub), strict=True):
                    if k:
                        out[Cell(st.name, o, job, ph)] = k
    return out


def _usable(c: Cell, cand: Candidate, held: set[str]) -> bool:
    """Cross-domain: the request never names the offered tool's domain. Distractor jobs:
    (job, slot values) post 9 used anywhere are held out."""
    st = STRATUM[c.stratum]
    if st.subkind == CROSS and mentions_domain(c.offered, cand.request):
        return False
    return not (c.job in DISTRACTOR_JOBS and entity(c.job, cand.slots) in held)


def _pool_rng(seed: int, job: str, phrasing: str) -> random.Random:
    return random.Random(f"{seed}:onetool:pool:{job}:{phrasing}")


class _Pools:
    """Each (job, phrasing) pool built once, in a fixed order; each cell walks its own
    seeded shuffle of it."""

    def __init__(self, seed: int):
        self.seed = seed
        self._pools: dict[tuple[str, str], list[Candidate]] = {}
        self._iters: dict[Cell, object] = {}

    def pool(self, job: str, phrasing: str) -> list[Candidate]:
        key = (job, phrasing)
        if key not in self._pools:
            cands = job_pool(job, phrasing, _pool_rng(self.seed, job, phrasing))
            self._pools[key] = sorted(cands, key=lambda x: (x.key(), x.request))
        return self._pools[key]

    def next(self, c: Cell) -> Candidate:
        if c not in self._iters:
            cands = list(self.pool(c.job, c.phrasing))
            tag = "" if c.phrasing == STRATUM[c.stratum].phrasing else f":{c.phrasing}"
            random.Random(f"{self.seed}:onetool:cell:{c.stratum}:{c.offered}:{c.job}{tag}"
                          ).shuffle(cands)
            self._iters[c] = iter(cands)
        return next(self._iters[c])


def record(split: str, i: int, c: Cell, cand: Candidate) -> dict:
    st, job = STRATUM[c.stratum], JOBS[c.job]
    executable = c.job in EXECUTABLE_JOBS
    return {
        "id": f"{FILE_PREFIX}{split}-{i:04d}", "split": split, "kind": st.kind,
        "subkind": st.subkind, "phrasing": c.phrasing, "tool": c.job,
        "job_domain": job.domain, "job_operation": job.operation,
        "verb_family": c.job if c.job in FAMILIES else None,
        "job_class": EXECUTABLE if executable else DISTRACTOR,
        "reading_required": st.kind == NOT_FIT and executable,
        "template": cand.template, "slots": cand.slots, "request": cand.request,
        "tools": [c.offered], "reference": cand.reference if st.kind == FITS else None,
    }


def generate(seed: int = tasks.DEFAULT_SEED,
             post9: dict[str, list[dict]] | None = None) -> dict[str, list[dict]]:
    """The one-tool splits: ``{"test", "dev", "train", "train-sample"}``."""
    if post9 is None:
        post9 = tasks.generate(seed)
    seen_text = post9_texts(post9)
    held = post9_entities(post9)
    seen_key: set[str] = set()
    earlier: set[tuple[str, str]] = set()  # task keys of the splits already drawn
    pools = _Pools(seed)
    chosen: dict[str, list[tuple[Cell, Candidate]]] = {s: [] for s in SPLITS}
    for split, sizes in SPLITS.items():  # test first: test never depends on train's size
        mine: set[tuple[str, str]] = set()
        for c, need in cell_sizes(sizes, seed, split).items():
            got = 0
            while got < need:
                try:
                    cand = pools.next(c)
                except StopIteration:
                    raise RuntimeError(f"pool too small: {c} / {split}") from None
                key = task_key(c.job, cand.slots, c.offered)
                if cand.request in seen_text or cand.key() in seen_key or key in earlier \
                        or not _usable(c, cand, held):
                    continue
                seen_text.add(cand.request)
                seen_key.add(cand.key())
                mine.add(key)
                chosen[split].append((c, cand))
                got += 1
        earlier |= mine
    out = {}
    for split, rows in chosen.items():
        random.Random(f"{seed}:onetool:split:{split}").shuffle(rows)
        out[split] = [record(split, i, c, cand) for i, (c, cand) in enumerate(rows)]
    out[SAMPLE_SPLIT] = train_sample(out["train"], seed)
    return out


def train_sample(train: list[dict], seed: int) -> list[dict]:
    """A seeded sample of train, :data:`SAMPLE_SIZES` by stratum, dealt over each
    stratum's jobs as the splits are (:func:`deal`), in id order."""
    picked = []
    for st in STRATA:
        jobs = jobs_of(st)
        rng = random.Random(f"{seed}:onetool:sample:{st.name}")
        for job, n in zip(jobs, deal(SAMPLE_SIZES[st.name], len(jobs), rng), strict=True):
            mine = [t for t in train if stratum_of(t) == st.name and t["tool"] == job]
            picked += rng.sample(mine, n)
    return sorted(picked, key=lambda t: t["id"])


# -- pool headroom ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolRow:
    job: str
    phrasing: str
    candidates: int  # distinct usable requests, post 9's texts (and entities) removed
    drawn: int  # test + dev + train, over every cell that draws from it

    @property
    def headroom(self) -> float:
        return self.candidates / self.drawn if self.drawn else float("inf")


def pool_capacity(seed: int = tasks.DEFAULT_SEED,
                  post9: dict[str, list[dict]] | None = None) -> list[PoolRow]:
    """Per (job, phrasing): candidates left after post 9's texts (and, for distractor
    jobs, post 9's entities), against what all three splits draw from it. Where a
    cross-domain cell skips requests naming its tool's domain, the smallest count
    over the cells is the one given."""
    if post9 is None:
        post9 = tasks.generate(seed)
    used, held = post9_texts(post9), post9_entities(post9)
    demand: Counter = Counter()
    for split, sizes in SPLITS.items():
        for c, n in cell_sizes(sizes, seed, split).items():
            demand[(c.job, c.phrasing)] += n
    pools = _Pools(seed)
    by_source: dict[tuple[str, str], set[str]] = {}
    for c in cells():
        st = STRATUM[c.stratum]
        domain = JOBS[c.offered].domain if st.subkind == CROSS else None
        by_source.setdefault((c.job, c.phrasing), set()).add(domain)
    rows = []
    for (job, phrasing), domains in by_source.items():
        counts_ = []
        for domain in domains:
            probe = Cell(STRATA[3].name if domain else STRATA[0].name,
                         next(o for o in OFFERED if JOBS[o].domain == domain) if domain
                         else job, job, phrasing)
            counts_.append(len({x.request for x in pools.pool(job, phrasing)
                                if x.request not in used and _usable(probe, x, held)}))
        rows.append(PoolRow(job, phrasing, min(counts_), demand[(job, phrasing)]))
    return rows


def jsonl_bytes(records: list[dict]) -> bytes:
    return tasks.jsonl_bytes(records)


def sha256(records: list[dict]) -> str:
    return tasks.sha256(records)


def write(splits: dict[str, list[dict]], directory) -> dict:
    return tasks.write({FILE_PREFIX + k: v for k, v in splits.items()}, directory)
