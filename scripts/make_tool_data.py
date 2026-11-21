"""Write the five tools' fixed fake data to src/tinytools/tools/data/ (run once; output is committed).

The JSON files are the source of truth: the tools and the task generator read
them, never this script. It is kept only to show how the data was made. Re-running
it rewrites the same bytes (seeded), which ``tests/test_tools.py`` does not rely on.

    uv run python scripts/make_tool_data.py
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "tinytools" / "tools" / "data"

CLOCK = {"date": "2026-03-11", "time": "09:30", "weekday": "Wednesday",
         "timezone": "none (local wall-clock time)"}

# Ordinary names: easy spellings, no confusable neighbour in the list.
ORDINARY = """Maria Garcia, David Chen, Priya Raman, Lucas Silva, Emma Johansson, Omar Farouk,
Hannah Weber, Kenji Watanabe, Fatima Zahra, Ben Carter, Olivia Brown, Liam Walker, Sofia Rossi,
Noah Kim, Ava Thompson, Ethan Wright, Mia Lopez, Lena Fischer, Raj Patel, Chloe Martin,
Daniel Cohen, Grace Ito, Isaac Mensah, Julia Novak, Kofi Asante, Laura Bianchi, Marco Conti,
Nina Petrova, Oscar Lindqvist, Paula Santos, Quentin Dubois, Rosa Jimenez, Samuel Osei,
Tara Singh, Umar Sheikh, Will Turner, Yara Haddad, Zoe Adams, Adam Fletcher, Bianca Moreno,
Carlos Vega, Diana Prieto, Erik Nilsson, Fiona Grant, Felix Baumann, Gina Russo, Hugo Lambert,
Irene Costa, Jack Harper, Karen Doyle, Leo Marsh, Megan Price, Nathan Ford, Olga Ivanova,
Peter Holm, Ruth Abrams, Simon Clarke, Tessa Vos, Victor Hale, Wendy Ortiz"""

# Names that are easy to misspell: a confusable neighbour in the list, or an
# unusual spelling a model may "correct".
TRICKY = """Jon Smith, John Smith, Sean Murphy, Shaun Murphy, Steven Park, Stephen Park,
Geoffrey Lin, Jeffrey Lin, Caitlin Brooks, Kaitlyn Brooks, Dmitri Volkov, Dmitry Volkov,
Katherine Nguyen, Catherine Nguyen, Mohammed Iqbal, Muhammad Iqbal, Ann Lee, Anne Leigh,
Siobhan O'Neill, Aoife Byrne, Niamh Kavanagh, Saoirse Doyle, Eoghan Gallagher, Padraig Walsh,
Mairead McGrath, Krzysztof Nowak, Wojciech Szczepanski, Ifeoma Nwachukwu, Leigh-Anne Porter,
Rhys Llewellyn"""

TEAMS = ["Engineering", "Sales", "Marketing", "Finance", "Legal", "Design", "Support",
         "Operations", "People", "Research"]

RECURRING = {0: ("Weekly planning", "10:00", 60), 3: ("Design review", "14:00", 60)}

ONE_OFF = """Budget review; Dentist appointment; Quarterly business review; Vendor call with Acme;
Offsite planning; Product demo; Hiring committee; Lunch with Priya Raman; Security training;
Customer interview; Sprint retrospective; Board prep; Marketing sync; Doctor's checkup;
Car service; Parent-teacher conference; Investor update; Roadmap workshop;
Interview: backend engineer; Interview: data scientist; All-hands; Coffee with Ben Carter;
Legal review; Pricing discussion; Launch rehearsal; Partner onboarding; Architecture deep dive;
Team lunch; Performance review; Yoga class; Flight to Berlin; Conference keynote;
Expense report deadline; Code freeze; Release party; Accessibility audit;
Data migration kickoff; Support escalation review; Compliance check-in; Office move planning;
Book club; Mentoring session; Tax advisor; Gym session; Haircut; Hackathon kickoff;
Webinar rehearsal; Recruiting fair; Brand refresh workshop; Analytics office hours;
Contract renewal call; Intern demo day; Wellness talk; Usability testing;
Supplier negotiation; Press briefing; Charity run; Fire drill; Piano lesson; Farewell drinks"""

DOC_SUBJECTS = ["sales report", "budget plan", "hiring plan", "product roadmap",
                "marketing review", "security audit", "customer survey results",
                "board meeting minutes", "support ticket summary", "infrastructure cost report",
                "partner newsletter", "OKR scorecard", "risk register", "churn analysis",
                "pricing proposal", "engineering offsite agenda"]
DOC_STANDALONE = ["Onboarding checklist", "Expense policy", "Brand guidelines", "Code of conduct",
                  "Remote work policy", "Travel policy", "Parental leave policy",
                  "Incident response runbook", "API style guide", "Writing style guide",
                  "Office floor plan", "Holiday calendar", "Laptop setup guide",
                  "Benefits overview", "Org chart", "Interview rubric"]


def _split(text: str, sep: str) -> list[str]:
    return [x.strip() for x in text.replace("\n", " ").split(sep) if x.strip()]


def _email(name: str) -> str:
    return re.sub(r"[^a-z.]", "", name.lower().replace(" ", ".")) + "@example.com"


def contacts(rng: random.Random) -> list[dict]:
    names = _split(ORDINARY, ",") + _split(TRICKY, ",")
    tricky = set(_split(TRICKY, ","))
    out = []
    for i, name in enumerate(names, start=1):
        out.append({"id": f"c-{i:03d}", "name": name, "email": _email(name),
                    "phone": f"+1-555-{rng.randrange(100, 1000):03d}-{rng.randrange(10000):04d}",
                    "team": rng.choice(TEAMS), "tricky_spelling": name in tricky})
    return out


def calendar(rng: random.Random, people: list[str]) -> list[dict]:
    start, end = dt.date(2026, 2, 23), dt.date(2026, 4, 24)
    days = [start + dt.timedelta(d) for d in range((end - start).days + 1)]
    weekdays = [d for d in days if d.weekday() < 5]
    titles = _split(ONE_OFF, ";")
    rng.shuffle(titles)
    # Every Tuesday, Wednesday and Friday gets one one-off event; the rest go to random weekdays.
    plain = [d for d in weekdays if d.weekday() in (1, 2, 4)]
    placed = list(zip(titles, plain, strict=False))  # titles outnumber days
    placed += [(t, rng.choice(weekdays)) for t in titles[len(plain):]]
    times = [f"{h:02d}:{m:02d}" for h in range(8, 17) for m in (0, 30)]
    events = []
    for day in weekdays:
        if day.weekday() in RECURRING:
            title, time, dur = RECURRING[day.weekday()]
            events.append({"date": day.isoformat(), "start_time": time, "duration_minutes": dur,
                           "title": title, "attendees": []})
    for title, day in placed:
        events.append({"date": day.isoformat(), "start_time": rng.choice(times),
                       "duration_minutes": rng.choice([30, 45, 60, 90]), "title": title,
                       "attendees": sorted(rng.sample(people, rng.choice([0, 1, 1, 2, 3])))})
    events.sort(key=lambda e: (e["date"], e["start_time"], e["title"]))
    return [{"id": f"evt-{i:03d}", **e} for i, e in enumerate(events, start=1)]


def documents(rng: random.Random) -> list[dict]:
    titles = [f"Q{q} {s}" for s in DOC_SUBJECTS for q in (1, 2, 3, 4)] + DOC_STANDALONE
    base = dt.date(2025, 6, 1)
    out = []
    for i, title in enumerate(titles, start=1):
        updated = base + dt.timedelta(rng.randrange((dt.date(2026, 3, 10) - base).days))
        out.append({"id": f"doc-{i:03d}", "title": title, "updated": updated.isoformat()})
    return out


def main() -> None:
    rng = random.Random("tinytools fake data v1")
    OUT.mkdir(parents=True, exist_ok=True)
    people = contacts(rng)
    ordinary = [c["name"] for c in people if not c["tricky_spelling"]]
    files = {"clock.json": CLOCK, "contacts.json": people,
             "calendar.json": calendar(rng, ordinary), "documents.json": documents(rng)}
    for name, obj in files.items():
        (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=True) + "\n")
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
