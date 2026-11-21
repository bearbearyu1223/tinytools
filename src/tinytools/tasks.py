"""A seeded generator of tool-calling tasks over the five tools (:mod:`tinytools.tools`).

A task is a user request, the tools offered (3 to 5 schemas), a reference call
(or ``None``) and a kind:

    choose+fill       pick the right tool and fill its arguments from the request
                      (explicit dates and 24-hour times, plain numbers, names with
                      ordinary spellings, document titles)
    tricky arguments  the arguments need work: dates relative to the frozen clock
                      ("next Tuesday"), 12-hour times and worded durations ("at
                      3pm for an hour and a half"), unit conversions and
                      percentages, and names spelled the way the contact list
                      spells them (a confusable neighbour such as Jon / John Smith,
                      or an unusual spelling such as Siobhan O'Neill)
    no tool fits      the request needs a tool that isn't offered (weather, email,
                      deleting an event, reminders, stock prices, translation);
                      the right answer is not to call

Each (kind, tool) has a pool of candidates, one per (template, slot values).
The pool is shuffled with a seed derived from ``seed``, kind and tool, and the
splits take candidates from it in order **test, dev, train**, skipping any
whose request text or (template, slots) was already used anywhere. So the
splits are disjoint by both, and test never depends on how big train is.

Sizes (choose+fill / tricky / no tool fits): test 600/200/200, dev 120/40/40,
train 3000/1000/1000 (the 60/20/20 mix, frozen for post 10). Within a kind,
tasks are dealt round-robin over its tools. ``train-r6a`` is a seeded sample
of train with the same mix (300/100/100).

Offered tools: k from {3, 4, 5} uniformly; the right tool plus k - 1 others
drawn from the other real tools and the distractors (for no tool fits: k drawn
from everything except the tool the request needs and its stand-ins,
:data:`SUBSTITUTES`); shuffled.

Everything is deterministic: ``random.Random`` with string seeds (hashed with
SHA-512, independent of PYTHONHASHSEED). ``tests/test_tasks.py`` pins the
split hashes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import tools

CHOOSE = "choose+fill"
TRICKY = "tricky arguments"
NO_FIT = "no tool fits"
KINDS = (CHOOSE, TRICKY, NO_FIT)
SHORT_KIND = {CHOOSE: "choose+fill", TRICKY: "tricky args", NO_FIT: "no tool fits"}

SPLITS = {"test": (600, 200, 200), "dev": (120, 40, 40), "train": (3000, 1000, 1000)}
R6A = (300, 100, 100)
DEFAULT_SEED = 0

KIND_TOOLS = {
    CHOOSE: ("calculator", "search_contacts", "get_calendar_event", "create_calendar_event",
             "search_documents"),
    TRICKY: ("calculator", "search_contacts", "get_calendar_event", "create_calendar_event"),
    NO_FIT: tools.DISTRACTOR_NAMES,
}


@dataclass(frozen=True)
class Candidate:
    template: str
    slots: dict
    request: str
    reference: dict | None

    def key(self) -> str:
        return self.template + "|" + json.dumps(self.slots, sort_keys=True)


# -- dates and times -----------------------------------------------------------------

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve"]


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def explicit_dates(day: dt.date) -> list[str]:
    """Ways to write a date that need no clock (choose+fill)."""
    m = MONTHS[day.month - 1]
    return [f"on {m} {day.day}", f"on {m[:3]} {day.day}", f"on {day.day} {m}",
            f"on {m} {ordinal(day.day)}", f"on {day.isoformat()}",
            f"on {WEEKDAYS[day.weekday()]}, {m} {day.day}"]


def relative_dates(today: dt.date) -> list[tuple[str, dt.date]]:
    """Phrases that need the clock, each with the date it means.

    Weekday phrases avoid the ambiguous ones: "next <weekday>" only for days
    already past in the current week (said on a Wednesday, "next Tuesday" can
    only be the coming one), "this" / "last <weekday>" only for days still
    ahead in it. Pure; tested.
    """
    out = [("today", today), ("tomorrow", today + dt.timedelta(1)),
           ("the day after tomorrow", today + dt.timedelta(2)),
           ("yesterday", today - dt.timedelta(1)),
           ("a week from today", today + dt.timedelta(7)),
           ("a week from tomorrow", today + dt.timedelta(8)),
           ("two weeks from today", today + dt.timedelta(14))]
    for i, name in enumerate(WEEKDAYS):
        if i <= today.weekday():
            # Already past (or today) this week: "next" can only mean the coming one.
            ahead = (i - today.weekday()) % 7 or 7
            out.append((f"next {name}", today + dt.timedelta(ahead)))
            out.append((f"on the {name} after next", today + dt.timedelta(ahead + 7)))
        else:
            # Still ahead this week: "next Friday" is ambiguous, so say "this Friday";
            # "last Friday" can only mean last week's.
            ahead = i - today.weekday()
            out.append((f"this {name}", today + dt.timedelta(ahead)))
            out.append((f"a week from {name}", today + dt.timedelta(ahead + 7)))
            out.append((f"last {name}", today + dt.timedelta(ahead - 7)))
    for n in range(2, 22):
        out.append((f"in {n} days", today + dt.timedelta(n)))
        if n <= 12:
            out.append((f"{NUMBER_WORDS[n]} days from now", today + dt.timedelta(n)))
    last = (today.replace(day=28) + dt.timedelta(4)).replace(day=1) - dt.timedelta(1)
    for d in range(today.day + 1, last.day + 1):
        out.append((f"on the {ordinal(d)}", today.replace(day=d)))
    first_next = last + dt.timedelta(1)
    for d in range(1, 25):
        out.append((f"on the {ordinal(d)} of next month", first_next.replace(day=d)))
    return out


TIMES_24 = [f"{h:02d}:{m:02d}" for h in range(8, 18) for m in (0, 15, 30, 45)]


def spoken_times() -> list[tuple[str, str]]:
    """12-hour and spoken times, with their 24-hour HH:MM."""
    out = []
    for h in range(8, 18):
        h12 = h - 12 if h > 12 else h
        ampm = "am" if h < 12 else "pm"
        out.append((f"{h12}{ampm}", f"{h:02d}:00"))
        out.append((f"{h12}:30{ampm}", f"{h:02d}:30"))
        out.append((f"{h12}:15 {ampm}", f"{h:02d}:15"))
        if h >= 13:
            out.append((f"half past {NUMBER_WORDS[h12]} in the afternoon", f"{h:02d}:30"))
            out.append((f"{NUMBER_WORDS[h12]} in the afternoon", f"{h:02d}:00"))
        elif h < 12:
            out.append((f"{NUMBER_WORDS[h12]} in the morning", f"{h:02d}:00"))
    out.append(("noon", "12:00"))
    return out


SPOKEN_DURATIONS = [("an hour", 60), ("half an hour", 30), ("an hour and a half", 90),
                    ("two hours", 120), ("a quarter of an hour", 15), ("three hours", 180),
                    ("ninety minutes", 90), ("forty-five minutes", 45), ("an hour and 15 minutes", 75),
                    ("two and a half hours", 150)]


# -- pools ---------------------------------------------------------------------------


def _call(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


def _names(tricky: bool) -> list[str]:
    return [c["name"] for c in tools.contacts() if c["tricky_spelling"] == tricky]


CONTACT_TEMPLATES = [
    "What's {name}'s phone number?",
    "Look up {name}'s email address.",
    "Find the contact details for {name}.",
    "Can you pull up {name} in my contacts?",
    "I need to reach {name}. What's their number?",
    "Which team is {name} on?",
    "Get me the email for {name}, please.",
    "Search my contacts for {name}.",
    "Who is {name}? Check my contacts.",
    "Do I have a phone number for {name}?",
    "Find {name} in the address book.",
    "What is the email address of {name}?",
    "Look up {name} for me.",
    "I want to call {name}. Can you find them?",
    "Pull up the contact card for {name}.",
    "Is {name} in my contacts?",
]


def contact_pool(tricky: bool) -> list[Candidate]:
    return [Candidate(f"contact-{i}", {"name": n}, t.format(name=n),
                      _call("search_contacts", query=n))
            for i, t in enumerate(CONTACT_TEMPLATES) for n in _names(tricky)]


DAY_TEMPLATES = [
    "What's on my calendar {when}?",
    "Do I have any meetings {when}?",
    "What does my schedule look like {when}?",
    "Show me everything I have {when}.",
    "Anything on the calendar {when}?",
    "List my events {when}, please.",
    "Am I busy {when}?",
    "What have I got going on {when}?",
]

TITLE_TEMPLATES = [
    'When is "{title}"?',
    "When is my {title}?",
    'Find the "{title}" event on my calendar.',
    "What time is the {title}?",
    'Look up "{title}" in my calendar.',
    "Which day is the {title} scheduled for?",
]


def _event_days() -> set[str]:
    return {e["date"] for e in tools.calendar()}


def _unique_titles() -> list[str]:
    from collections import Counter

    counts = Counter(e["title"] for e in tools.calendar())
    return sorted(t for t, n in counts.items() if n == 1)


def get_event_pool(tricky: bool) -> list[Candidate]:
    days = _event_days()
    out = []
    if tricky:
        for phrase, day in relative_dates(tools.now().date()):
            if day.isoformat() not in days:
                continue  # an empty result would match any query that finds nothing
            for i, t in enumerate(DAY_TEMPLATES):
                out.append(Candidate(f"day-{i}", {"when": phrase}, t.format(when=phrase),
                                     _call("get_calendar_event", date=day.isoformat())))
        return out
    for iso in sorted(days):
        day = dt.date.fromisoformat(iso)
        for j, phrase in enumerate(explicit_dates(day)):
            for i, t in enumerate(DAY_TEMPLATES):
                out.append(Candidate(f"day-{i}", {"when": phrase, "format": j},
                                     t.format(when=phrase),
                                     _call("get_calendar_event", date=iso)))
    for title in _unique_titles():
        for i, t in enumerate(TITLE_TEMPLATES):
            out.append(Candidate(f"title-{i}", {"title": title}, t.format(title=title),
                                 _call("get_calendar_event", title=title)))
    return out


NEW_TITLES = ["Budget sync", "Dentist follow-up", "Team offsite", "Client call", "Lunch with Omar",
              "Project kickoff", "Quarterly planning", "Coffee chat", "Design critique",
              "Hiring sync", "Sales pipeline review", "Board call", "Tech talk", "Demo prep",
              "Vendor review", "Retro", "Onboarding session", "Strategy session", "Yoga",
              "Car pickup", "Vet appointment", "Study group", "Podcast recording",
              "Release review", "Budget workshop", "Interview debrief", "Code review",
              "Customer onboarding", "Town hall prep", "Mentor check-in"]

CREATE_PLAIN = [
    'Schedule "{title}" {when} at {time} for {dur} minutes{with_}.',
    'Add "{title}" to my calendar {when} at {time}, {dur} minutes long{with_}.',
    'Put "{title}" on my calendar {when} from {time} for {dur} minutes{with_}.',
    'Create an event called "{title}" {when} at {time} lasting {dur} minutes{with_}.',
    'Book "{title}" {when} at {time} for {dur} minutes{with_}.',
    'Set up "{title}" {when}, starting at {time}, for {dur} minutes{with_}.',
]
CREATE_SPOKEN = [
    'Schedule "{title}" {when} at {time} for {dur}{with_}.',
    'Add "{title}" to my calendar {when} at {time} for {dur}{with_}.',
    'Put "{title}" on my calendar {when} at {time}; it takes {dur}{with_}.',
    'Create an event called "{title}" {when} at {time} lasting {dur}{with_}.',
    'Book "{title}" {when} at {time} for {dur}{with_}.',
    'Set up "{title}" {when}, starting at {time}, for {dur}{with_}.',
]


def create_pool(tricky: bool, rng: random.Random, size: int = 6000) -> list[Candidate]:
    today = tools.now().date()
    people = _names(False)
    if tricky:
        whens = [(p, d) for p, d in relative_dates(today) if d > today]  # future only
        times, durs, templates = spoken_times(), SPOKEN_DURATIONS, CREATE_SPOKEN
    else:
        days = [today + dt.timedelta(k) for k in range(1, 43)]
        whens = [(p, d) for d in days for p in explicit_dates(d)]
        times = [(t, t) for t in TIMES_24]
        durs = [(str(m), m) for m in (15, 20, 30, 45, 60, 75, 90, 120)]
        templates = CREATE_PLAIN
    out = []
    for _ in range(size):
        i = rng.randrange(len(templates))
        title = rng.choice(NEW_TITLES)
        when, day = rng.choice(whens)
        time_text, hhmm = rng.choice(times)
        dur_text, minutes = rng.choice(durs)
        who = rng.sample(people, rng.choice([0, 0, 1, 2]))
        with_ = (" with " + " and ".join(who)) if who else ""
        args = {"title": title, "date": day.isoformat(), "start_time": hhmm,
                "duration_minutes": minutes}
        if who:
            args["attendees"] = who
        slots = {"title": title, "when": when, "time": time_text, "dur": dur_text, "who": who}
        out.append(Candidate(f"create-{'spoken' if tricky else 'plain'}-{i}", slots,
                             templates[i].format(title=title, when=when, time=time_text,
                                                 dur=dur_text, with_=with_),
                             _call("create_calendar_event", **args)))
    return out


DOC_TEMPLATES = [
    'Find the "{title}" document.',
    "Can you pull up the {title}?",
    "Search our docs for the {title}.",
    "Where can I find the {title}?",
    "I need the {title}. Can you find it?",
    'Look up the document called "{title}".',
    "Open the {title} for me.",
    'Is there a doc named "{title}"?',
    "Get me the latest {title}.",
    "Find our {title}.",
    'Search documents for "{title}".',
    "Which file is the {title}?",
]


def document_pool() -> list[Candidate]:
    return [Candidate(f"doc-{i}", {"title": d["title"]}, t.format(title=d["title"]),
                      _call("search_documents", query=d["title"]))
            for i, t in enumerate(DOC_TEMPLATES) for d in tools.documents()]


def _num(rng: random.Random, lo: int, hi: int, decimals: bool = False) -> int | float:
    if decimals and rng.random() < 0.3:
        return round(rng.uniform(lo, hi), rng.choice([1, 2]))
    return rng.randint(lo, hi)


CALC_PLAIN = [
    ("What is {a} times {b}?", "{a} * {b}"),
    ("Calculate {a} multiplied by {b}.", "{a} * {b}"),
    ("What's {a} plus {b}?", "{a} + {b}"),
    ("Add {a} and {b}.", "{a} + {b}"),
    ("What is {a} minus {b}?", "{a} - {b}"),
    ("Subtract {b} from {a}.", "{a} - {b}"),
    ("Divide {a} by {b}.", "{a} / {b}"),
    ("What's {a} divided by {b}?", "{a} / {b}"),
    ("What is {s} to the power of {n}?", "{s} ** {n}"),
    ("Compute ({a} + {b}) * {c}.", "({a} + {b}) * {c}"),
    ("What's the sum of {a}, {b} and {c}?", "{a} + {b} + {c}"),
    ("What is {s} squared?", "{s} ** 2"),
]

CALC_UNITS = [
    ("How many minutes are there in {h} hours?", "{h} * 60"),
    ("How many seconds are in {m} minutes?", "{m} * 60"),
    ("How many hours are in {d} days?", "{d} * 24"),
    ("How many days are in {w} weeks?", "{w} * 7"),
    ("What is {p}% of {x}?", "{x} * {p} / 100"),
    ("Convert {km} kilometers to meters.", "{km} * 1000"),
    ("How many grams is {kg} kg?", "{kg} * 1000"),
    ("Convert {cm} centimeters to meters.", "{cm} / 100"),
    ("If a mile is 1.609344 km, how many km is {mi} miles?", "{mi} * 1.609344"),
    ("At {rate} dollars per euro, how many dollars is {eur} euros?", "{eur} * {rate}"),
    ("What is {h2} hours and {m2} minutes in minutes?", "{h2} * 60 + {m2}"),
    ("How many eggs are in {n} dozen?", "{n} * 12"),
    ("What's a {p}% tip on a {x} dollar bill?", "{x} * {p} / 100"),
    ("How many seconds are in {d} days?", "{d} * 86400"),
]


def calculator_pool(tricky: bool, rng: random.Random, size: int = 6000) -> list[Candidate]:
    templates = CALC_UNITS if tricky else CALC_PLAIN
    out = []
    for _ in range(size):
        i = rng.randrange(len(templates))
        text, expr = templates[i]
        slots = {
            "a": _num(rng, 12, 9999, decimals=True), "b": _num(rng, 2, 999, decimals=True),
            "c": _num(rng, 2, 99), "s": _num(rng, 2, 30), "n": _num(rng, 2, 6),
            "h": _num(rng, 1, 12, decimals=True), "m": _num(rng, 2, 600),
            "d": _num(rng, 2, 30), "w": _num(rng, 2, 52), "p": _num(rng, 5, 95),
            "x": _num(rng, 20, 5000, decimals=True), "km": _num(rng, 1, 500, decimals=True),
            "kg": _num(rng, 1, 200, decimals=True), "cm": _num(rng, 5, 5000),
            "mi": _num(rng, 1, 300, decimals=True), "rate": round(rng.uniform(1.02, 1.25), 2),
            "eur": _num(rng, 5, 2000), "h2": _num(rng, 1, 9), "m2": _num(rng, 1, 59),
        }
        used = {k: v for k, v in slots.items() if "{" + k + "}" in text}
        out.append(Candidate(f"calc-{'units' if tricky else 'plain'}-{i}", used,
                             text.format(**used),
                             _call("calculator", expression=expr.format(**used))))
    return out


CITIES = ["Paris", "Tokyo", "Chicago", "Lagos", "Lima", "Oslo", "Seoul", "Denver", "Madrid",
          "Toronto", "Sydney", "Cairo", "Dublin", "Mumbai", "Nairobi", "Boston", "Berlin",
          "Lisbon", "Austin", "Vienna", "Hanoi", "Seattle", "Prague", "Manila", "Zurich"]
WEATHER_WHEN = ["today", "tomorrow", "this weekend", "on Friday", "next week"]
WEATHER_TEMPLATES = ["What's the weather going to be in {city} {when}?",
                     "Will it rain in {city} {when}?",
                     "Should I bring an umbrella in {city} {when}?",
                     "How warm will it be in {city} {when}?",
                     "What's the forecast for {city} {when}?"]
EMAIL_MSGS = ["the meeting moved to Thursday", "I'm running ten minutes late",
              "the report is ready for review", "thanks for the help yesterday",
              "lunch is on me on Friday", "the contract was signed",
              "please send the slides", "the launch is postponed",
              "I'll be out of office next week", "the budget was approved"]
EMAIL_TEMPLATES = ["Email {name} and tell them {msg}.", "Send {name} an email saying {msg}.",
                   "Shoot {name} a quick email: {msg}.", "Write to {name} by email that {msg}."]
DELETE_TEMPLATES = ['Cancel my "{title}" meeting.', 'Delete "{title}" from my calendar.',
                    'Remove the "{title}" event.', 'Please drop "{title}" from my schedule.',
                    'Get rid of "{title}" on my calendar.']
REMIND_TASKS = ["call the bank", "pick up the dry cleaning", "water the plants",
                "submit my timesheet", "book flights", "renew my passport", "pay the rent",
                "buy a birthday card", "back up my laptop", "take out the recycling",
                "send the invoice", "stretch", "check the oven", "feed the cat", "charge my phone"]
REMIND_WHEN = ["tomorrow at 9am", "in an hour", "tonight at 8", "on Friday morning",
               "next Monday at noon", "in 20 minutes", "this afternoon at 4", "at 6pm today",
               "on the 20th at 10am", "next week"]
REMIND_TEMPLATES = ["Remind me to {task} {when}.", "Set a reminder {when} to {task}.",
                    "Can you remind me {when} to {task}?"]
COMPANIES = ["Apple", "Microsoft", "Nvidia", "Amazon", "Alphabet", "Meta", "Tesla", "Netflix",
             "Intel", "AMD", "Oracle", "Salesforce", "Adobe", "IBM", "Cisco", "Qualcomm",
             "Shopify", "Spotify", "Uber", "Airbnb", "PayPal", "Visa", "Mastercard", "Nike",
             "Disney", "Starbucks", "Coca-Cola", "PepsiCo", "Walmart", "Costco", "Boeing",
             "Ford", "Toyota", "Sony", "Samsung", "Siemens", "SAP", "ASML", "Pfizer", "Moderna"]
STOCK_TEMPLATES = ["What's {company}'s stock trading at?", "How is {company} stock doing today?",
                   "Get me the current share price of {company}.",
                   "What did {company} stock close at yesterday?",
                   "Check the stock price for {company}.", "Is {company} stock up or down today?"]
PHRASES = ["good morning", "where is the train station", "thank you very much",
           "the meeting is cancelled", "how much does this cost", "I would like a coffee",
           "see you tomorrow", "happy birthday", "the invoice is attached", "nice to meet you",
           "where is the nearest pharmacy", "can I pay by card", "the office is closed today",
           "please call me back", "I am allergic to peanuts"]
LANGUAGES = ["French", "Spanish", "German", "Japanese", "Italian", "Portuguese", "Korean",
             "Dutch", "Swedish", "Polish"]
TRANSLATE_TEMPLATES = ["Translate '{phrase}' into {lang}.", "How do you say '{phrase}' in {lang}?",
                       "What's '{phrase}' in {lang}?"]


def no_fit_pool(needs: str) -> list[Candidate]:
    def grid(prefix, templates, **axes) -> list[Candidate]:
        import itertools

        keys = list(axes)
        out = []
        for i, t in enumerate(templates):
            for combo in itertools.product(*axes.values()):
                slots = dict(zip(keys, combo, strict=True))
                out.append(Candidate(f"{prefix}-{i}", slots, t.format(**slots), None))
        return out

    if needs == "get_weather":
        return grid("weather", WEATHER_TEMPLATES, city=CITIES, when=WEATHER_WHEN)
    if needs == "send_email":
        return grid("email", EMAIL_TEMPLATES, name=_names(False), msg=EMAIL_MSGS)
    if needs == "delete_calendar_event":
        return grid("delete", DELETE_TEMPLATES, title=sorted({e["title"] for e in tools.calendar()}))
    if needs == "set_reminder":
        return grid("remind", REMIND_TEMPLATES, task=REMIND_TASKS, when=REMIND_WHEN)
    if needs == "get_stock_price":
        return grid("stock", STOCK_TEMPLATES, company=COMPANIES)
    if needs == "translate_text":
        return grid("translate", TRANSLATE_TEMPLATES, phrase=PHRASES, lang=LANGUAGES)
    raise ValueError(needs)


def pool(kind: str, tool: str, rng: random.Random) -> list[Candidate]:
    tricky = kind == TRICKY
    builders: dict[str, Callable[[], list[Candidate]]] = {
        "calculator": lambda: calculator_pool(tricky, rng),
        "search_contacts": lambda: contact_pool(tricky),
        "get_calendar_event": lambda: get_event_pool(tricky),
        "create_calendar_event": lambda: create_pool(tricky, rng),
        "search_documents": document_pool,
    }
    if kind == NO_FIT:
        return no_fit_pool(tool)
    return builders[tool]()


# -- splits --------------------------------------------------------------------------


def per_tool(total: int, n_tools: int) -> list[int]:
    """Deal ``total`` tasks round-robin over ``n_tools``: the first ones get the remainder."""
    return [total // n_tools + (1 if i < total % n_tools else 0) for i in range(n_tools)]


# On a no-tool-fits task, tools that could stand in for the one needed are not
# offered either: a calendar event is a fair way to be reminded of something.
SUBSTITUTES = {"set_reminder": ("create_calendar_event",)}


def _offer(rng: random.Random, right: str | None, needs: str | None) -> list[str]:
    everything = list(tools.TOOL_NAMES) + list(tools.DISTRACTOR_NAMES)
    k = rng.choice([3, 4, 5])
    if right is not None:
        offered = [right] + rng.sample([t for t in everything if t != right], k - 1)
    else:
        banned = {needs, *SUBSTITUTES.get(needs, ())}
        offered = rng.sample([t for t in everything if t not in banned], k)
    rng.shuffle(offered)
    return offered


def generate(seed: int = DEFAULT_SEED) -> dict[str, list[dict]]:
    """All splits: ``{"test": [...], "dev": [...], "train": [...], "train-r6a": [...]}``."""
    chosen: dict[str, list[tuple[str, str, Candidate]]] = {s: [] for s in SPLITS}
    seen_text: set[str] = set()
    seen_key: set[str] = set()
    for k, kind in enumerate(KINDS):
        names = KIND_TOOLS[kind]
        for t, tool in enumerate(names):
            rng = random.Random(f"{seed}:pool:{kind}:{tool}")
            cands = pool(kind, tool, rng)
            cands.sort(key=lambda c: (c.key(), c.request))  # a fixed order before the shuffle
            rng.shuffle(cands)
            it = iter(cands)
            for split, sizes in SPLITS.items():
                need = per_tool(sizes[k], len(names))[t]
                got = 0
                while got < need:
                    try:
                        c = next(it)
                    except StopIteration:
                        raise RuntimeError(f"pool too small: {kind} / {tool} / {split}") from None
                    if c.request in seen_text or (kind + c.key()) in seen_key:
                        continue
                    seen_text.add(c.request)
                    seen_key.add(kind + c.key())
                    chosen[split].append((kind, tool, c))
                    got += 1
    out = {}
    for split, rows in chosen.items():
        rng = random.Random(f"{seed}:split:{split}")
        rng.shuffle(rows)
        records = []
        for i, (kind, tool, c) in enumerate(rows):
            right = None if kind == NO_FIT else tool
            records.append({
                "id": f"{split}-{i:04d}", "split": split, "kind": kind,
                "tool": tool, "template": c.template, "slots": c.slots, "request": c.request,
                "tools": _offer(rng, right, None if right else tool), "reference": c.reference,
            })
        out[split] = records
    out["train-r6a"] = r6a_sample(out["train"], seed)
    return out


def r6a_sample(train: list[dict], seed: int) -> list[dict]:
    """500 train tasks for the ready-for-RL measurement, 300/100/100 by kind, in id order."""
    rng = random.Random(f"{seed}:r6a")
    picked = []
    for kind, n in zip(KINDS, R6A, strict=True):
        picked += rng.sample([t for t in train if t["kind"] == kind], n)
    return sorted(picked, key=lambda t: t["id"])


def jsonl_bytes(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode()


def sha256(records: list[dict]) -> str:
    return hashlib.sha256(jsonl_bytes(records)).hexdigest()


def write(splits: dict[str, list[dict]], directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, records in splits.items():
        p = directory / f"{name}.jsonl"
        p.write_bytes(jsonl_bytes(records))
        paths[name] = p
    return paths


def load(directory: Path, split: str) -> list[dict]:
    with open(directory / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


# -- the prompt ------------------------------------------------------------------------


def messages(task: dict) -> list[dict]:
    """System (Qwen's default sentence + the frozen clock, ToolACE's shape) and the request."""
    from .toolace import QWEN_DEFAULT_SYSTEM

    return [{"role": "system", "content": QWEN_DEFAULT_SYSTEM + "\n" + tools.context_line()},
            {"role": "user", "content": task["request"]}]


def tool_list(task: dict) -> list[dict]:
    return [{"type": "function", "function": tools.schema(n)} for n in task["tools"]]
