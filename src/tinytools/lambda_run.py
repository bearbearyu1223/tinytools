"""Run an experiment on Lambda Cloud from this laptop, as ``docs/lambda-playbook.md`` describes.

    scripts/lambda.sh doctor               settings, SSH key, agent, API; launches nothing
    scripts/lambda.sh types                instance types, prices, regions with capacity
    scripts/lambda.sh estimate EXP         price x hours, the experiment's and post's caps left
    scripts/lambda.sh run EXP [--approve USD]
                                           up -> setup -> run -> fetch -> down, terminated always
    scripts/lambda.sh list                 every instance; flags this repo's
    scripts/lambda.sh down ID [--yes]      terminate one instance and wait until Lambda confirms

Exit codes: 0 done and terminated; 1 the run failed but its instance is confirmed
terminated; 2 refused before launching (nothing started); 3 TERMINATION NOT CONFIRMED, or
another tinytools instance still up (an instance may be billing: ``list`` and ``down`` now).

Billing stops only when Lambda terminates the instance, so ``run`` terminates it
through the API from this laptop whatever happens: success, a failing command,
an exception, Ctrl-C, SIGTERM/SIGHUP, or the time limit. ``max_hours`` bounds the
instance's whole life (launch to termination), not just the commands: the work
must stop ``RESERVE_S`` before it so fetching and terminating fit inside it.

Python, not bash (tinyvlm's ``lambda.sh`` was bash): the API key travels in a
request header instead of on a ``curl`` command line where ``ps`` shows it; the
cleanup is a ``finally`` plus signal handlers rather than a ``trap``, which is
easier to make ignore a second Ctrl-C while it terminates; and the whole flow is
tested against a local fake of the API (``tests/test_lambda_run.py``). Standard
library only, so it never imports torch.

Secrets: ``lambda.env`` (gitignored) or the environment. Nothing here prints a
value from it; :class:`Redactor` scrubs the API key, any ``Bearer`` token and the
SSH key's path (as written and expanded) from every line written anyway.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import http.client
import io
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
# Lambda's API reference (docs.lambda.com) uses this base; LAMBDA_API_URL overrides it.
API_URL = "https://cloud.lambda.com/api/v1"
NAME_PREFIX = "tinytools"  # every instance this repo launches is named tinytools-<exp>-<stamp>
NAME_MAX = 64  # Lambda's limit on an instance name
RUN_TAG = "tinytools-run"  # the tag key whose value is unique to one run
SSH_USER = "ubuntu"
REMOTE_DIR = "tinytools"
ENV_KEYS = ("LAMBDA_API_KEY", "LAMBDA_SSH_KEY_NAME", "LAMBDA_SSH_KEY")
OPTIONAL_ENV = ("LAMBDA_API_URL", "LAMBDA_REGION")

ENDED = ("terminated", "preempted")  # the instance is gone; billing has stopped
BROKEN = ("unhealthy",)  # up (and billing) but not usable: terminate

RESERVE_S = 600  # kept back from max_hours for stop + fetch + terminate
TERMINATE_ALLOW_S = 300  # of the reserve, kept for terminate + confirm (fetch gets the rest)
REMOTE_MARGIN_S = 30  # the remote `timeout` fires this long before the laptop's deadline
BOOT_TIMEOUT_S = 20 * 60
SSH_WAIT_S = 10 * 60  # SSH must answer within this of the instance going active, and
# the job's polls may fail for at most this long in a row
SETUP_TIMEOUT_S = 25 * 60  # copying the repo and the setup script, together
LOST_LAUNCH_S = 120  # after a launch whose answer was lost: how long to look for it
STATUS_CHECK_S = 300  # while the commands run: how often to ask Lambda for the instance
POLL_S = 15
BOOT_POLL_S = 10
TERMINATE_TRIES = 20  # failed terminate calls before the runner says so loudly (it keeps going)
RESEND_S = 60  # re-send terminate this often until Lambda confirms
CONFIRM_S = 2 * 3600  # keep terminating and checking this long before giving up (exit 3)
FETCH_TIMEOUT_S = 420
STOP_WAIT_S = 20  # after TERM to the job's session: wait this long, then KILL
RATE_RETRIES = 6  # 429s retried per call, waiting 1, 2, 4, 8, 15, 15 s
RATE_WAIT_S = 1.0
RATE_WAIT_MAX_S = 15.0
API_MIN_INTERVAL_S = 1.0  # Lambda allows 1 request per second
POST_CAP_USD = 50.0  # the plan's cap per post, summed over the post's experiments
# GNU timeout exits 124 when it sent TERM, 137 when it sent KILL after --kill-after; 137 is
# also bash's code for a command SIGKILLed by anyone (the OOM killer): see exit_status
KILLED = 137
DEADLINE_ENV = "TINYTOOLS_DEADLINE_EPOCH"  # job.sh: when the remote `timeout` fires (epoch s)
# A line of the job's output exactly this asks for a checkpoint fetch (an experiment's
# commands can `echo` it, e.g. after each seed pair): what was written so far is fetched
# while the job goes on, so a run cut off later still leaves those files on the laptop.
CHECKPOINT_MARK = "TINYTOOLS_FETCH_NOW"
CHECKPOINT_MIN_S = 60  # a checkpoint fetch is skipped with less than this before the deadline
DIED = "died"  # poll(): the job is gone and left no exit code
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_UNCONFIRMED = 0, 1, 2, 3
LEDGER_FIELDS = ("run", "experiment", "post", "commit", "instance_type", "price_usd_per_hour",
                 "launched_at", "ended_at", "hours", "cost_usd", "status")


class Refused(Exception):
    """A check before launch failed; nothing was started."""


class Interrupted(Exception):
    """SIGTERM or SIGHUP arrived."""


class TimeLimit(Exception):
    """The deadline (launch + max_hours - reserve) passed."""


# -- experiments ---------------------------------------------------------------------

FETCH_PATTERN = re.compile(r"[A-Za-z0-9_.*?/-]+")


@dataclass(frozen=True)
class Experiment:
    name: str
    instance_type: str
    max_hours: float
    expected_hours: float
    cap_usd: float
    commands: tuple[str, ...]
    fetch: tuple[str, ...]
    post: int
    setup_record: str | None = None


_TYPES = {"name": str, "post": int, "instance_type": str, "max_hours": (int, float),
          "expected_hours": (int, float), "cap_usd": (int, float), "commands": list,
          "fetch": list, "setup_record": str}
_REQUIRED = ("name", "post", "instance_type", "max_hours", "expected_hours", "cap_usd",
             "commands", "fetch")
_STAMP = "%Y%m%d-%H%M%S"


def instance_name(experiment: str, t: float) -> str:
    return f"{NAME_PREFIX}-{experiment}-{time.strftime(_STAMP, time.localtime(t))}"


def parse_experiment(text: str, stem: str) -> Experiment:
    """An ``experiments/<stem>.toml``, checked; raises ``ValueError`` saying what is wrong."""
    try:
        d = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"not valid TOML: {e}") from None
    unknown = sorted(set(d) - set(_TYPES))
    if unknown:
        raise ValueError(f"unknown keys {unknown}; allowed: {sorted(_TYPES)}")
    missing = [k for k in _REQUIRED if k not in d]
    if missing:
        raise ValueError(f"missing keys {missing}")
    for k, t in _TYPES.items():
        if k in d and (not isinstance(d[k], t) or isinstance(d[k], bool)):
            raise ValueError(f"{k} has the wrong type ({type(d[k]).__name__})")
    if d["name"] != stem:
        raise ValueError(f"name {d['name']!r} differs from the file name {stem!r}")
    if len(instance_name(d["name"], 0)) > NAME_MAX:
        raise ValueError(f"name {d['name']!r} is too long: instance names are at most "
                         f"{NAME_MAX} characters")
    if not 0 < d["max_hours"] <= 24:
        raise ValueError("max_hours must be in (0, 24]")
    if not 0 < d["expected_hours"] <= d["max_hours"]:
        raise ValueError("expected_hours must be in (0, max_hours]")
    if not d["cap_usd"] > 0:
        raise ValueError("cap_usd must be positive")
    cmds = d["commands"]
    if not cmds or not all(isinstance(c, str) and c.strip() and "\n" not in c for c in cmds):
        raise ValueError("commands must be a non-empty list of one-line strings")
    fetch = d["fetch"]
    if not fetch or not all(isinstance(p, str) for p in fetch):
        raise ValueError("fetch must be a non-empty list of strings")
    for p in fetch:
        if not FETCH_PATTERN.fullmatch(p) or p.startswith(("/", "-")) or ".." in p.split("/"):
            raise ValueError(f"fetch pattern {p!r}: relative paths of [A-Za-z0-9_.*?/-] only, "
                             "no '..'")
    rec = d.get("setup_record")
    if rec is not None and not any(Path(rec).match(p) for p in fetch):
        raise ValueError(f"setup_record {rec!r} is not matched by any fetch pattern")
    return Experiment(d["name"], d["instance_type"], float(d["max_hours"]),
                      float(d["expected_hours"]), float(d["cap_usd"]), tuple(cmds), tuple(fetch),
                      d["post"], rec)


def load_experiment(root: Path, name: str) -> tuple[Experiment, Path]:
    path = root / "experiments" / f"{name}.toml"
    if not path.exists():
        raise Refused(f"no experiment file {path.relative_to(root)}")
    try:
        return parse_experiment(path.read_text(encoding="utf-8"), name), path
    except ValueError as e:
        raise Refused(f"{path.relative_to(root)}: {e}") from None


# -- settings and secrets ----------------------------------------------------------------


def parse_env(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines (``export`` and matching quotes allowed; ``#`` comments)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def load_env(root: Path, environ: dict | None = None) -> dict[str, str]:
    """``lambda.env`` next to the README, overridden by the environment."""
    environ = os.environ if environ is None else environ
    path = root / "lambda.env"
    env = parse_env(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k in ENV_KEYS + OPTIONAL_ENV:
        if environ.get(k):
            env[k] = environ[k]
    return env


def describe_env(env: dict[str, str]) -> list[str]:
    """Names and lengths only, never values."""
    return [f"{k}: set ({len(env[k])} characters)" if env.get(k) else f"{k}: MISSING"
            for k in ENV_KEYS]


class Redactor:
    """Replaces every secret value, and any ``Bearer ...`` token, in a string. A value that
    is a path (has a ``/`` or starts with ``~``) is also replaced in its expanded and
    absolute forms, since that is how ssh and its error messages carry it. Only what is
    passed in is replaced: the SSH key's *name* at Lambda is not a secret, and redacting
    a short common word would garble every line that contains it."""

    def __init__(self, secrets: list[str]) -> None:
        values = set()
        for s in secrets:
            if not s:
                continue
            values.add(s)
            if "/" in s or s.startswith("~"):
                e = os.path.expanduser(s)
                values.update({e, os.path.abspath(e), os.path.realpath(e)})
        self.secrets = sorted({v for v in values if len(v) >= 4}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "[redacted]")
        return re.sub(r"(?i)bearer\s+(?!\[redacted\])\S+", "Bearer [redacted]", text)


# -- the API ---------------------------------------------------------------------------


class ApiError(RuntimeError):
    """``status`` is the HTTP status, or ``None`` when no answer came back (a transport
    error, a cut-off body, an unreadable reply): then the request may or may not have
    taken effect."""

    def __init__(self, status: int | None, message: str, code: str = "",
                 api_message: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code  # the API's error.code, e.g. "global/invalid-parameters"
        self.api_message = api_message  # the API's error.message


def _error_fields(body: bytes) -> tuple[str, str]:
    """The API's ``error.code`` and ``error.message`` ("" when the body has none)."""
    try:
        err = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return "", ""
    if not isinstance(err, dict):
        return "", ""
    return str(err.get("code") or ""), str(err.get("message") or "")


TAG_FIELD = re.compile(r"\btags\b")


def refuses_tags(e: ApiError) -> bool:
    """Lambda refused the launch for its ``tags`` field: a 4xx whose error code is
    ``global/invalid-parameters`` and whose message names the field (not any message
    that happens to contain "tag")."""
    return (definite_refusal(e) and e.code == "global/invalid-parameters"
            and TAG_FIELD.search(e.api_message) is not None)


def definite_refusal(e: ApiError) -> bool:
    """A 4xx: Lambda read the request and refused it, so it did nothing."""
    return e.status is not None and 400 <= e.status < 500


def _request_id(payload, headers) -> str | None:
    if isinstance(payload, dict):
        for d in (payload, payload.get("error") if isinstance(payload.get("error"), dict) else {}):
            if d.get("request_id"):
                return str(d["request_id"])
    for k, v in (headers.items() if headers is not None else []):
        if k.lower().replace("_", "-").endswith("request-id") and v:
            return v
    return None


def _error_text(body: bytes, headers=None) -> str:
    try:
        payload = json.loads(body)
        err = payload.get("error", {})
        text = f"{err.get('code', '')}: {err.get('message', '')}".strip(": ")
    except (ValueError, AttributeError):
        payload, text = None, body.decode("utf-8", "replace")[:200]
    rid = _request_id(payload, headers)
    return text + (f" (request_id {rid})" if rid else "")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: urllib would send the Authorization header to the new
    host, and turn a POST into a body-less GET. The 3xx becomes an error instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _is_lambda_host(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host.endswith((".lambda.com", ".lambda.ai", ".lambdalabs.com"))


class LambdaAPI:
    """The few Lambda Cloud API calls the runner needs. Errors are redacted.

    Every failure is an :class:`ApiError`; nothing else escapes ``_call`` (an
    ``http.client`` protocol error or a cut-off body included). A 429 is retried with
    backoff; requests are spaced ``min_interval`` apart."""

    def __init__(self, base: str, key: str, redact: Redactor, timeout: float = 30,
                 min_interval: float = 0.0, rate_wait: float = RATE_WAIT_S) -> None:
        self.base = base.rstrip("/")
        self._key = key
        self.redact = redact
        self.timeout = timeout
        self.min_interval = min_interval
        self.rate_wait = rate_wait
        self._last = 0.0
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _pace(self) -> None:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _once(self, method: str, path: str, body: dict | None):
        req = urllib.request.Request(
            self.base + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                     "Content-Type": "application/json", "User-Agent": "tinytools-runner"})
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            try:
                payload = e.read()
            except (http.client.HTTPException, OSError, ValueError):
                payload = b""
            return e.code, payload, e.headers

    def _call(self, method: str, path: str, body: dict | None = None):
        if not self._key:
            raise ApiError(None, "LAMBDA_API_KEY is not set (lambda.env)")
        for attempt in range(RATE_RETRIES + 1):
            self._pace()
            try:
                status, payload, headers = self._once(method, path, body)
            except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
                raise ApiError(None, self.redact(
                    f"{method} {path}: no answer: {type(e).__name__}: {e}")) from None
            if 200 <= status < 300:
                try:
                    return json.loads(payload)["data"]
                except (ValueError, KeyError, TypeError):
                    raise ApiError(None, f"{method} {path}: unexpected response") from None
            if 300 <= status < 400:
                where = (headers.get("Location") if headers is not None else None) or "?"
                raise ApiError(status, self.redact(
                    f"{method} {path}: HTTP {status}: {self.base} redirects to {where}; not "
                    "followed (the key would go with it). If that is Lambda's new address, "
                    "set LAMBDA_API_URL"))
            if status == 429 and attempt < RATE_RETRIES:
                after = headers.get("Retry-After") if headers is not None else None
                try:
                    wait = float(after)
                except (TypeError, ValueError):
                    wait = self.rate_wait * 2 ** attempt
                time.sleep(min(RATE_WAIT_MAX_S * self.rate_wait / RATE_WAIT_S, wait))
                continue
            code, msg = _error_fields(payload)
            raise ApiError(status, self.redact(
                f"{method} {path}: HTTP {status}: {_error_text(payload, headers)}"),
                code=code, api_message=self.redact(msg))
        raise AssertionError("unreachable")

    def instance_types(self) -> dict:
        return self._call("GET", "/instance-types")

    def instances(self) -> list[dict]:
        return self._call("GET", "/instances")

    def instance(self, instance_id: str) -> dict | None:
        """The instance, or ``None`` when Lambda doesn't know it (404): gone, or (just
        after a launch) not listed yet."""
        try:
            return self._call("GET", f"/instances/{instance_id}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def launch(self, instance_type: str, region: str, ssh_key_name: str, name: str,
               tag: str | None = None) -> str:
        body = {"region_name": region, "instance_type_name": instance_type,
                "ssh_key_names": [ssh_key_name], "name": name}
        if tag:
            body["tags"] = [{"key": RUN_TAG, "value": tag}]
        data = self._call("POST", "/instance-operations/launch", body)
        try:
            return str(data["instance_ids"][0])
        except (KeyError, IndexError, TypeError):
            raise ApiError(None, "POST /instance-operations/launch: no instance id in the "
                                 "answer") from None

    def terminate(self, ids: list[str]) -> list[str]:
        """The ids Lambda says it is terminating (its answer lists Instance objects)."""
        data = self._call("POST", "/instance-operations/terminate", {"instance_ids": ids})
        out = []
        for x in (data or {}).get("terminated_instances", []) if isinstance(data, dict) else []:
            out.append(str(x.get("id")) if isinstance(x, dict) else str(x))
        return out

    def ssh_keys(self) -> list[dict]:
        return self._call("GET", "/ssh-keys")


def is_ours(instance: dict) -> bool:
    return str(instance.get("name") or "").startswith(NAME_PREFIX)


def run_tag(instance: dict) -> str | None:
    for t in instance.get("tags") or []:
        if isinstance(t, dict) and t.get("key") == RUN_TAG:
            return t.get("value")
    return None


def live(instance: dict) -> bool:
    return instance.get("status") not in ENDED


def terminate_available(instance: dict) -> bool:
    """Lambda's ``actions.terminate.available`` (false while a VM has not launched)."""
    act = ((instance.get("actions") or {}).get("terminate") or {})
    return act.get("available", True) is not False


# -- money (pure) ---------------------------------------------------------------------


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spent(rows: list[dict], experiment: str) -> float:
    """Dollars charged to an experiment so far. A run still marked running counts at its
    worst case (that is what it was entered as), so a crashed runner can't free up cap."""
    return round(sum(float(r["cost_usd"] or 0) for r in rows if r["experiment"] == experiment), 4)


def spent_on_post(rows: list[dict], post: int) -> float:
    """Dollars charged to every experiment of a post (the ledger's ``post`` column)."""
    return round(sum(float(r["cost_usd"] or 0) for r in rows
                     if str(r.get("post") or "") == str(post)), 4)


def write_ledger_row(path: Path, row: dict) -> None:
    """Add ``row``, or replace the row with the same ``run``."""
    rows = [r for r in read_ledger(path) if r["run"] != row["run"]]
    rows.append({k: row.get(k, "") for k in LEDGER_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r.get(k) or "" for k in LEDGER_FIELDS} for r in rows)


def cost_usd(price_per_hour: float, seconds: float) -> float:
    return round(price_per_hour * seconds / 3600, 4)


@dataclass(frozen=True)
class Estimate:
    experiment: str
    instance_type: str
    price: float  # USD per hour, from the API now
    max_hours: float
    expected_hours: float
    spent: float
    cap: float
    regions: tuple[str, ...]  # with capacity now
    post: int = 0
    post_spent: float = 0.0
    post_cap: float = POST_CAP_USD

    @property
    def worst(self) -> float:
        """Rounded up to the cent: it is checked against the cap."""
        return math.ceil(round(self.price * self.max_hours * 100, 6)) / 100

    @property
    def expected(self) -> float:
        return round(self.price * self.expected_hours, 2)

    @property
    def remaining(self) -> float:
        return round(self.cap - self.spent, 2)

    @property
    def post_remaining(self) -> float:
        return round(self.post_cap - self.post_spent, 2)

    @property
    def within_cap(self) -> bool:
        return self.worst <= self.remaining and self.worst <= self.post_remaining

    def refusal(self) -> str | None:
        if self.worst > self.remaining:
            return (f"the worst case ${self.worst:.2f} is more than the ${self.remaining:.2f} "
                    f"left of {self.experiment}'s ${self.cap:.2f} cap")
        if self.worst > self.post_remaining:
            return (f"the worst case ${self.worst:.2f} is more than the "
                    f"${self.post_remaining:.2f} left of post {self.post}'s "
                    f"${self.post_cap:.2f} cap")
        return None

    def lines(self) -> list[str]:
        why = ("REFUSED: the worst case is more than the experiment's cap left"
               if self.worst > self.remaining else
               "REFUSED: the worst case is more than the post's cap left"
               if self.worst > self.post_remaining else "within both caps")
        return [
            f"experiment          {self.experiment} (post {self.post})",
            f"instance type       {self.instance_type}",
            f"price (API, now)    ${self.price:.2f}/h",
            f"capacity now        {', '.join(self.regions) or 'none'}",
            f"expected            {self.expected_hours:g} h = ${self.expected:.2f}",
            f"worst case          {self.max_hours:g} h (hard limit) = ${self.worst:.2f}",
            f"cap, experiment     ${self.cap:.2f}, spent ${self.spent:.2f}, "
            f"left ${self.remaining:.2f}",
            f"cap, post {self.post:<9} ${self.post_cap:.2f}, spent ${self.post_spent:.2f}, "
            f"left ${self.post_remaining:.2f}",
            "verdict             " + why,
        ]


def gpu_family(instance_type: str) -> str:
    """The GPU a Lambda type name names: ``gpu_1x_a100_sxm4`` -> ``a100``. Pure."""
    parts = instance_type.split("_")
    return parts[2] if len(parts) > 2 and parts[0] == "gpu" else instance_type


def same_gpu_lines(types: dict, instance_type: str) -> list[str]:
    """Every type in Lambda's live list with the experiment's GPU (any count, memory or
    form factor), cheapest first: name, price, Lambda's description (which states the
    memory, e.g. "1x A100 (40 GB SXM4)") and regions with capacity. Pure."""
    fam = gpu_family(instance_type)
    rows = sorted((v["instance_type"]["price_cents_per_hour"], k,
                   v["instance_type"].get("description", ""),
                   ", ".join(r["name"] for r in v.get("regions_with_capacity_available", []))
                   or "no capacity")
                  for k, v in types.items() if gpu_family(k) == fam)
    if not rows:
        return [f"no {fam} types in Lambda's list now"]
    lines = [f"{fam} types in Lambda's list now (the experiment's is marked *):"]
    for cents, name, desc, regions in rows:
        mark = "*" if name == instance_type else " "
        lines.append(f" {mark} {name:<26} ${cents / 100:>6.2f}/h  {desc}  [{regions}]")
    return lines


def make_estimate(exp: Experiment, types: dict, ledger: list[dict]) -> Estimate:
    if exp.instance_type not in types:
        fam = gpu_family(exp.instance_type)
        near = sorted(n for n in types if gpu_family(n) == fam
                      or exp.instance_type.split("_")[-1] in n)
        raise Refused(f"instance type {exp.instance_type} is not in Lambda's list now"
                      + (f"; similar: {', '.join(near)}" if near else ""))
    item = types[exp.instance_type]
    regions = tuple(r["name"] for r in item.get("regions_with_capacity_available", []))
    return Estimate(exp.name, exp.instance_type,
                    item["instance_type"]["price_cents_per_hour"] / 100, exp.max_hours,
                    exp.expected_hours, spent(ledger, exp.name), exp.cap_usd, regions,
                    exp.post, spent_on_post(ledger, exp.post), POST_CAP_USD)


# -- what runs on the instance -----------------------------------------------------------

SETUP_SCRIPT = r"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
cd ~/tinytools
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv --version
uv sync --frozen
uv run python - <<'PY'
import json, platform, subprocess, sys
from importlib.metadata import version
import torch
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU: Linux must take torch from the cu126 index (pyproject.toml)")
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()[0]
gpu, driver = [x.strip() for x in smi.split(",")]
print("TINYTOOLS_SETUP_JSON " + json.dumps({
    "gpu": gpu, "driver": driver, "cuda (torch build)": torch.version.cuda,
    "torch": torch.__version__, "transformers": version("transformers"),
    "bfcl-eval": version("bfcl-eval"), "python": platform.python_version(),
    "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip(),
    "os": platform.platform()}))
PY
"""
SETUP_MARK = "TINYTOOLS_SETUP_JSON "

# Started detached (nohup setsid), so a dropped SSH connection doesn't stop it; its pid is
# its session id, and stop() signals the whole session (GNU timeout moves itself and the
# commands into a process group of their own, but not out of the session). .run/start is
# the marker fetch compares against: only files newer than it are this run's. $1 is the
# hard limit in seconds; `timeout` stops the work then (it does not stop the bill). The
# commands see when that is as $TINYTOOLS_DEADLINE_EPOCH (milestone0 --deadline-epoch).
JOB_SCRIPT = """#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo $$ > .run/pid
touch .run/start
export PATH="$HOME/.local/bin:$PATH"
export TINYTOOLS_DEADLINE_EPOCH=$(( $(date +%s) + $1 ))
timeout --signal=TERM --kill-after=60 "$1" bash .run/commands.sh
echo $? > .run/exit.tmp && mv .run/exit.tmp .run/exit
"""

# The exit code; else "running" while job.sh lives; else "died" (killed, or the instance
# restarted: no exit code will ever come). The exit file is read again after the liveness
# check, since job.sh writes it just before it ends.
POLL_SCRIPT = (f"cd {REMOTE_DIR} || exit 1; if [ -f .run/exit ]; then cat .run/exit; "
               'elif kill -0 "$(cat .run/pid 2>/dev/null)" 2>/dev/null; then echo running; '
               f"elif [ -f .run/exit ]; then cat .run/exit; else echo {DIED}; fi; "
               "tail -c +{start} .run/log.txt")

STOP_SCRIPT = (f"cd {REMOTE_DIR} 2>/dev/null && [ -f .run/pid ] || exit 0; sid=$(cat .run/pid); "
               'pkill -TERM -s "$sid"; for i in $(seq 1 {wait}); do '
               'pgrep -s "$sid" >/dev/null || exit 0; sleep 1; done; '
               'pkill -KILL -s "$sid"; exit 0')


def commands_script(commands: tuple[str, ...]) -> str:
    """The experiment's commands, in order, stopping at the first that fails; each is echoed."""
    lines = ["set -euo pipefail"]
    for c in commands:
        lines.append(f"printf '+ %s\\n' {shlex.quote(c)}")
        lines.append(c)
    return "\n".join(lines) + "\n"


def parse_setup(output: str) -> dict:
    for line in output.splitlines():
        if line.startswith(SETUP_MARK):
            return json.loads(line[len(SETUP_MARK):])
    return {}


def run_dir_name(day: str, experiment: str, existing: set[str]) -> str:
    """``<date>-<exp>``, then ``-2``, ``-3`` for later runs the same day."""
    base = f"{day}-{experiment}"
    if base not in existing:
        return base
    k = 2
    while f"{base}-{k}" in existing:
        k += 1
    return f"{base}-{k}"


KILLED_STATUS = "killed (SIGKILL, e.g. out of memory)"


def exit_status(code: int, elapsed_s: float, limit_s: float) -> str:
    """run.json's status for the commands' exit code. 124 is GNU timeout's own. 137 is
    also what bash returns for a command SIGKILLed by anyone (the kernel's OOM killer),
    so it counts as the time limit only once the remote limit has passed (``timeout``
    sends its KILL ``--kill-after`` seconds after it); before that it was killed."""
    if code == 0:
        return "ok"
    if code == 124 or (code == KILLED and elapsed_s >= limit_s):
        return "time limit"
    if code == KILLED:
        return KILLED_STATUS
    return f"failed: a command exited {code}"


def remote_seconds(deadline: float, now: float, margin: float) -> int:
    """The remote `timeout`, so the work stops ``margin`` before the laptop's deadline."""
    return max(1, int(deadline - now - margin))


def fetch_budget(hard_end: float, now: float, terminate_allow: float,
                 cap: float = FETCH_TIMEOUT_S) -> float:
    """Seconds fetch may take: at most ``cap``, and never into the time kept for
    terminating before the instance's hard end. 0: skip the fetch."""
    return max(0.0, min(cap, hard_end - now - terminate_allow))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_listing(out: str) -> list[tuple[str, int]]:
    """``n<TAB>path`` lines (fetch's listing; n is 1 if the file changed after the last
    completed fetch's marker, else 0) -> (path, n). Pure."""
    rows = []
    for line in out.splitlines():
        size, sep, path = line.partition("\t")
        if sep and path.strip() and size.strip().isdigit():
            rows.append((path, int(size)))
    return rows


def mark_lines(pending: str, chunk: str) -> tuple[int, str]:
    """How many complete lines of ``pending + chunk`` are exactly :data:`CHECKPOINT_MARK`,
    and the unfinished last line to carry into the next chunk. Pure."""
    text = pending + chunk
    *lines, rest = text.split("\n")
    return sum(line.rstrip("\r") == CHECKPOINT_MARK for line in lines), rest[-4096:]


# -- the remote side over SSH ---------------------------------------------------------------


class SSHRemote:
    """The instance, reached with ssh and rsync (BatchMode: never a prompt; ``-n`` and no
    stdin unless a script is sent, so ssh never reads the terminal)."""

    def __init__(self, ip: str, key_path: str, known_hosts: Path) -> None:
        self.host = f"{SSH_USER}@{ip}"
        self.opts = ["-i", os.path.expanduser(key_path), "-o", "BatchMode=yes",
                     "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
                     "-o", f"UserKnownHostsFile={known_hosts}", "-o", "ServerAliveInterval=30",
                     "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=10"]

    def _ssh(self, command: str, timeout: float, stdin: str | None = None, text: bool = True):
        if stdin is None:
            return subprocess.run(["ssh", "-n", *self.opts, self.host, command],
                                  stdin=subprocess.DEVNULL, capture_output=True, text=text,
                                  timeout=timeout)
        return subprocess.run(["ssh", *self.opts, self.host, command], input=stdin,
                              capture_output=True, text=text, timeout=timeout)

    def ready(self) -> bool:
        try:
            return self._ssh("true", 30).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def push(self, src: Path, timeout: float) -> None:
        ssh = " ".join(shlex.quote(x) for x in ["ssh", *self.opts])
        r = subprocess.run(["rsync", "-az", "-e", ssh, f"{src}/", f"{self.host}:{REMOTE_DIR}/"],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"rsync to the instance failed: {r.stderr.strip()[-300:]}")

    def run(self, script: str, timeout: float) -> tuple[int, str]:
        r = self._ssh("bash -s", timeout, stdin=script)
        return r.returncode, r.stdout + r.stderr

    def start(self, seconds: int) -> None:
        # Not `cd x && nohup ... &`: that backgrounds the whole `&&` list in a subshell
        # which keeps this session's stdout open, so ssh would wait for the job to end.
        # Waits (up to 10 s) for job.sh to write its pid, so poll() can tell a job that
        # has not started yet from one that died.
        r = self._ssh(f"cd {REMOTE_DIR} || exit 1; rm -f .run/pid .run/start .run/exit; "
                      f"nohup setsid bash .run/job.sh {seconds} > .run/log.txt 2>&1 < /dev/null & "
                      "for i in $(seq 1 100); do [ -f .run/pid ] && break; sleep 0.1; done; "
                      "[ -f .run/pid ] && echo started", 60)
        if "started" not in r.stdout:
            raise RuntimeError(f"could not start the job: {r.stderr.strip()[-300:]}")

    def poll(self, offset: int) -> tuple[bytes, int | str | None]:
        """New log bytes, and the exit code (``None`` while running, :data:`DIED`)."""
        r = self._ssh(POLL_SCRIPT.format(start=offset + 1), 60, text=False)
        if r.returncode:
            raise RuntimeError("poll failed")
        head, _, rest = r.stdout.partition(b"\n")
        code = head.strip().decode()
        return rest, None if code == "running" else DIED if code == DIED else int(code)

    def stop(self, wait_s: float = STOP_WAIT_S) -> None:
        """TERM every process in the job's session, wait for them to end, then KILL."""
        self._ssh(STOP_SCRIPT.format(wait=max(1, int(wait_s))), wait_s + 30)

    def fetch(self, patterns: tuple[str, ...], dest: Path, timeout: float, mark: int = 1,
              since: int | None = None, have=()) -> tuple[list[str], list[str]]:
        """The files matching ``patterns`` that this run wrote (newer than .run/start), and
        those copied. First touches ``.run/fetch-<mark>``, before listing, so whatever is
        written after the listing is newer than it. A file in ``have`` (a fetch already
        copied it) is skipped unless its status changed after ``.run/fetch-<since>``, the
        marker of the last fetch that completed: ``find -cnewer``, the inode's change time,
        which a write and also a rename into place update (a file written whole and
        renamed keeps the temporary file's mtime). Replaces the skip by size, which kept a
        stale copy of a file rewritten at the same size (T8 review S6). GNU and BSD find.
        Returns (listed, copied)."""
        t0 = time.time()
        prev = f".run/fetch-{int(since)}" if since is not None else ""
        # c=1: changed after the last completed fetch began (or its marker is gone)
        newer = (f'if [ ! -f {prev} ] || [ -n "$(find "$f" -maxdepth 0 -cnewer {prev})" ]; '
                 'then c=1; fi; ') if prev else ""
        r = self._ssh(f"cd {REMOTE_DIR} && [ -f .run/start ] && touch .run/fetch-{int(mark)} && "
                      f"shopt -s nullglob && for f in {' '.join(patterns)}; do [ -f \"$f\" ] && "
                      '[ -n "$(find "$f" -maxdepth 0 -newer .run/start)" ] && '
                      f"{{ c=0; {newer}printf '%s\\t%s\\n' \"$c\" \"$f\"; }}; "
                      "done; true", min(60.0, timeout))
        listed = parse_listing(r.stdout)
        files = sorted({f for f, _ in listed})
        changed = {f for f, c in listed if c}
        todo = sorted(f for f in files if f not in set(have) or f in changed or since is None)
        if not todo:
            return files, []
        dest.mkdir(parents=True, exist_ok=True)
        # a tar stream over ssh: macOS ships openrsync, whose --files-from with a remote
        # source is not something to find out about mid-run
        tar = subprocess.run(["ssh", "-n", *self.opts, self.host,
                              f"cd {REMOTE_DIR} && tar -cf - " + " ".join(map(shlex.quote, todo))],
                             stdin=subprocess.DEVNULL, capture_output=True,
                             timeout=max(1.0, timeout - (time.time() - t0)), check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(dest, filter="data")
        return files, todo


# -- git ----------------------------------------------------------------------------------


def clean_commit(root: Path) -> str:
    """HEAD's commit, refusing a tree with uncommitted changes to tracked files."""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                              check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=no")
    except subprocess.CalledProcessError as e:
        raise Refused(f"git: {e.stderr.strip()}") from None
    if status.strip():
        raise Refused("the working tree has uncommitted changes (commit first, so the run "
                      "names a commit that is exactly what ran):\n" + status.rstrip())
    return commit


def export_tree(root: Path, commit: str, dest: Path) -> None:
    """The repo at ``commit`` (git archive: tracked files only) into ``dest``."""
    tar = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit],
                         capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")


# -- the run --------------------------------------------------------------------------------


@dataclass
class RunState:
    run_dir: Path | None = None
    instance_id: str | None = None
    instance_name: str = ""
    tag: str = ""
    region: str = ""
    launch_attempted: bool = False
    launch_refused: bool = False  # a 4xx: Lambda started nothing
    launch_uncertain: bool = False  # 5xx or no answer: it may have started one
    existed: bool = False  # an instance id was known (so it may have billed)
    seen: set = field(default_factory=set)  # ids Lambda has shown us
    t_launch: float | None = None
    t_end: float | None = None
    terminated: bool = False
    status: str = "not started"
    remote_exit: int | None = None
    versions: dict = field(default_factory=dict)
    fetched: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    have: set = field(default_factory=set)  # paths a fetch copied (the laptop has them)
    fetch_marks: int = 0  # fetches started: fetch N touches .run/fetch-N on the instance
    fetched_mark: int | None = None  # the marker of the last fetch that completed
    checkpoints: int = 0  # checkpoint fetches that completed


class Runner:
    """``run EXP``: every check, then up -> setup -> run -> fetch -> down."""

    def __init__(self, root: Path, exp: Experiment, exp_path: Path, api: LambdaAPI,
                 env: dict[str, str], *, remote_factory: Callable[[str], object],
                 approve: float | None = None, setup_script: str = SETUP_SCRIPT,
                 poll_s: float = POLL_S, boot_poll_s: float = BOOT_POLL_S,
                 reserve_s: float = RESERVE_S, margin_s: float = REMOTE_MARGIN_S,
                 confirm_s: float = CONFIRM_S, resend_s: float = RESEND_S,
                 ssh_wait_s: float = SSH_WAIT_S, boot_timeout_s: float = BOOT_TIMEOUT_S,
                 lost_launch_s: float = LOST_LAUNCH_S, status_check_s: float = STATUS_CHECK_S,
                 stop_wait_s: float = STOP_WAIT_S, setup_timeout_s: float = SETUP_TIMEOUT_S,
                 keep_awake: bool = True, stream=None,
                 ask: Callable[[str], str] | None = None) -> None:
        self.root, self.exp, self.exp_path, self.api, self.env = root, exp, exp_path, api, env
        self.remote_factory = remote_factory
        self.approve = approve
        self.setup_script = setup_script
        self.poll_s, self.boot_poll_s = poll_s, boot_poll_s
        self.reserve_s, self.margin_s, self.confirm_s = reserve_s, margin_s, confirm_s
        self.resend_s, self.ssh_wait_s, self.boot_timeout_s = resend_s, ssh_wait_s, boot_timeout_s
        self.lost_launch_s, self.status_check_s = lost_launch_s, status_check_s
        self.stop_wait_s = stop_wait_s
        self.setup_timeout_s = setup_timeout_s
        self.keep_awake = keep_awake
        self.stream = stream or sys.stderr
        self.ask = ask
        self.redact = api.redact
        self.state = RunState()
        self.remote = None
        self.price = 0.0
        self.commit = ""
        self._log = None
        self._caffeinate = None

    # -- output ------------------------------------------------------------------

    def say(self, msg: str) -> None:
        line = self.redact(f"[{time.strftime('%H:%M:%S')}] {msg}")
        for out in (self.stream, self._log):
            if out is None:
                continue
            try:
                out.write(line + "\n")
                out.flush()
            except (OSError, ValueError):
                pass  # a closed terminal must not stop the cleanup

    def _job_output(self, chunk: bytes) -> None:
        text = self.redact(chunk.decode("utf-8", "replace"))
        for out in (self.stream, self._log):
            try:
                out.write(text)
                out.flush()
            except (OSError, ValueError, AttributeError):
                pass

    # -- checks before launch ---------------------------------------------------------

    def preflight(self) -> Estimate:
        self.commit = clean_commit(self.root)
        est = make_estimate(self.exp, self.api.instance_types(),
                            read_ledger(self.root / "runs" / "ledger.csv"))
        for line in est.lines():
            self.say(line)
        if est.refusal():
            raise Refused(est.refusal())
        if not est.regions:
            raise Refused(f"no capacity for {est.instance_type} right now; try later")
        if self.approve is None:
            if self.ask is None:
                raise Refused("no approval: pass --approve USD (at least the worst case) or run "
                              "in a terminal")
            answer = self.ask(f"Launch {est.instance_type} for up to {est.max_hours:g} h "
                              f"(${est.worst:.2f})? Type yes: ")
            if answer.strip().lower() != "yes":
                raise Refused("not approved")
        elif est.worst > self.approve:
            raise Refused(f"the worst case ${est.worst:.2f} is more than the approved "
                          f"${self.approve:.2f}")
        running = [i for i in self.api.instances() if is_ours(i) and live(i)]
        if running:
            raise Refused("an instance of this repo is already up ("
                          + ", ".join(f"{i['id']} {i.get('status')}" for i in running)
                          + "); `down` it or let its run finish first")
        return est

    # -- the run ---------------------------------------------------------------------------

    def run(self) -> int:
        try:
            est = self.preflight()
        except (Refused, ApiError) as e:
            self.say(f"refused: {e}")
            return EXIT_REFUSED
        self.price = est.price
        runs = self.root / "runs"
        runs.mkdir(exist_ok=True)
        existing = {p.name for p in runs.iterdir()}
        st = self.state
        st.run_dir = runs / run_dir_name(time.strftime("%Y-%m-%d"), self.exp.name, existing)
        st.run_dir.mkdir()
        self._log = open(st.run_dir / "log.txt", "a", encoding="utf-8")
        self._log.write("\n".join(self.redact(line) for line in est.lines()) + "\n")
        self.say(f"run {st.run_dir.name}: commit {self.commit[:12]}")
        old = self._handlers()
        clean = None
        try:
            self._awake()
            region = self.env.get("LAMBDA_REGION")
            region = region if region in est.regions else est.regions[0]
            self._launch(region)
            ip = self._wait_active()
            self.remote = self.remote_factory(ip)
            self._wait_ssh()
            self._setup()
            self._job()
        except KeyboardInterrupt:
            st.status = "interrupted (Ctrl-C)"
        except Interrupted as e:
            st.status = f"interrupted ({e})"
        except TimeLimit:
            st.status = "time limit"
        except Exception as e:  # noqa: BLE001 - every failure ends in the same cleanup
            st.status = f"failed: {type(e).__name__}: {e}"
        finally:
            clean = self._cleanup(old)
        if not st.terminated or clean is False or (clean is None and st.launch_uncertain):
            self.say("EXIT 3: termination is not confirmed; an instance may be billing. Run "
                     "`scripts/lambda.sh list` and `down` it, or use the dashboard, NOW")
            return EXIT_UNCONFIRMED
        if st.status == "ok" and st.missing:
            self.say(f"EXIT 1: the commands succeeded, but nothing they wrote matched "
                     f"{', '.join(st.missing)}")
            return EXIT_FAILED
        return EXIT_OK if st.status == "ok" else EXIT_FAILED

    def _handlers(self) -> dict:
        def raise_interrupted(signum, frame):
            raise Interrupted(signal.Signals(signum).name)
        old = {}
        try:
            for sig in (signal.SIGTERM, signal.SIGHUP):
                old[sig] = signal.signal(sig, raise_interrupted)
            old[signal.SIGINT] = signal.signal(signal.SIGINT, signal.default_int_handler)
        except ValueError:  # not the main thread
            pass
        return old

    def _awake(self) -> None:
        """caffeinate: the Mac must not sleep while it is the one that terminates. In its
        own session, so the Ctrl-C that the terminal sends its foreground group (and a
        SIGHUP when the terminal closes) doesn't end it before the cleanup has run."""
        if self.keep_awake and shutil.which("caffeinate"):
            self._caffeinate = subprocess.Popen(
                ["caffeinate", "-i", "-s", "-w", str(os.getpid())], start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.say("caffeinate on: this Mac will not idle-sleep during the run")

    @property
    def hard_end(self) -> float:
        """When the instance's life must end (launch + max_hours)."""
        return self.state.t_launch + self.exp.max_hours * 3600

    @property
    def deadline(self) -> float:
        return self.hard_end - self.reserve_s

    def _check_deadline(self) -> None:
        if self.state.t_launch is not None and time.time() >= self.deadline:
            raise TimeLimit()

    def _record(self) -> None:
        """run.json and the ledger row, rewritten at each stage (so a crash leaves both)."""
        st = self.state
        ended = st.t_end or time.time()
        if st.t_launch is None or (st.terminated and not st.existed):
            hours, cost = 0.0, 0.0
        elif st.terminated:
            hours, cost = (ended - st.t_launch) / 3600, cost_usd(self.price, ended - st.t_launch)
        else:  # still up, or unconfirmed: charge the worst case
            hours, cost = self.exp.max_hours, cost_usd(self.price, self.exp.max_hours * 3600)
        stamp = lambda t: (dt.datetime.fromtimestamp(t, dt.timezone.utc)  # noqa: E731
                           .isoformat(timespec="seconds") if t else None)
        setup = self._setup_record()
        data = {
            "experiment": self.exp.name, "post": self.exp.post,
            "experiment_file": str(self.exp_path.relative_to(self.root)),
            "experiment_sha256": sha256_file(self.exp_path), "commit": self.commit,
            "commands": list(self.exp.commands),
            "instance": {"id": st.instance_id, "name": st.instance_name, "tag": st.tag,
                         "type": self.exp.instance_type, "region": st.region,
                         "price_usd_per_hour": self.price},
            "max_hours": self.exp.max_hours, "launched_at": stamp(st.t_launch),
            "ended_at": stamp(st.t_end), "hours": round(hours, 4), "cost_usd": cost,
            "cost_basis": ("API price at launch x (confirmed termination - launch request); "
                           "worst case while not confirmed. Lambda's invoice is authoritative."),
            "status": st.status, "terminated": st.terminated, "remote_exit_code": st.remote_exit,
            "versions": st.versions,
            "seeds": setup.get("seeds"), "determinism": setup.get("determinism"),
            "dtype": setup.get("dtype"), "experiment_setup": setup,
            "fetched": st.fetched, "not_found": st.missing, "checkpoints": st.checkpoints,
        }
        (st.run_dir / "run.json").write_text(self.redact(json.dumps(data, indent=2)) + "\n")
        write_ledger_row(self.root / "runs" / "ledger.csv", {
            "run": st.run_dir.name, "experiment": self.exp.name, "post": str(self.exp.post),
            "commit": self.commit[:12], "instance_type": self.exp.instance_type,
            "price_usd_per_hour": f"{self.price:.2f}",
            "launched_at": stamp(st.t_launch) or "", "ended_at": stamp(st.t_end) or "",
            "hours": f"{hours:.4f}", "cost_usd": f"{cost:.4f}",
            "status": self.redact(st.status)})

    def _setup_record(self) -> dict:
        rec = self.exp.setup_record
        path = self.state.run_dir / rec if rec else None
        if path and path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                return {"error": "unreadable"}
        return {}

    def _is_this_run(self, inst: dict) -> bool:
        st = self.state
        return bool((st.instance_name and inst.get("name") == st.instance_name)
                    or (st.tag and run_tag(inst) == st.tag)
                    or (st.instance_id and inst.get("id") == st.instance_id))

    def _launch(self, region: str) -> None:
        st = self.state
        st.region = region
        st.status = "launching"
        st.t_launch = time.time()  # billing may start from here: count it
        st.instance_name = instance_name(self.exp.name, st.t_launch)
        st.tag = f"{st.run_dir.name}-{secrets.token_hex(4)}"
        self._record()
        self.say(f"launching {self.exp.instance_type} in {region} as {st.instance_name}")
        st.launch_attempted = True
        # Uncertain from before the POST until a definite answer: Ctrl-C, SIGTERM/SIGHUP
        # (or anything else) mid-request leaves it set, so the cleanup looks for the
        # instance and an unconfirmed lookup exits 3, never 1.
        st.launch_uncertain = True
        key_name = self.env.get("LAMBDA_SSH_KEY_NAME", "")
        try:
            try:
                st.instance_id = self.api.launch(self.exp.instance_type, region, key_name,
                                                 st.instance_name, st.tag)
            except ApiError as e:
                if not refuses_tags(e):
                    raise
                self.say(f"the launch refused the run tag ({e}); launching again without it "
                         "(the run's instance is then found by its name only)")
                st.tag = ""
                st.instance_id = self.api.launch(self.exp.instance_type, region, key_name,
                                                 st.instance_name)
        except ApiError as e:
            if definite_refusal(e):
                st.launch_refused = True
                st.launch_uncertain = False
            raise
        st.launch_uncertain = False
        st.existed = True
        st.status = "running"
        self._record()
        self.say(f"instance {st.instance_id}: billing has started; hard limit "
                 f"{self.exp.max_hours:g} h")

    def _wait_active(self) -> str:
        st = self.state
        t0 = time.time()
        while True:
            self._check_deadline()
            try:
                inst = self.api.instance(st.instance_id)
            except ApiError as e:
                self.say(f"status check failed: {e}")
            else:
                # None (404): just launched and not listed yet; keep waiting
                if inst is not None:
                    st.seen.add(st.instance_id)
                    status = inst.get("status")
                    if status == "active" and inst.get("ip"):
                        self.say(f"active at {inst['ip']}")
                        return inst["ip"]
                    if status not in ("booting", "active"):
                        raise RuntimeError(f"instance status is {status!r}")
            if time.time() - t0 > self.boot_timeout_s:
                raise RuntimeError(f"not active after {self.boot_timeout_s / 60:g} minutes")
            time.sleep(self.boot_poll_s)

    def _wait_ssh(self) -> None:
        self.say("waiting for SSH")
        t0 = time.time()
        while not self.remote.ready():
            self._check_deadline()
            if time.time() - t0 > self.ssh_wait_s:
                raise RuntimeError(f"SSH did not answer within {self.ssh_wait_s / 60:g} minutes "
                                   "(key not in the agent, or registered from another machine?)")
            time.sleep(self.boot_poll_s)

    def _setup(self) -> None:
        st = self.state
        st.status = "setup"
        setup_end = time.time() + self.setup_timeout_s
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            export_tree(self.root, self.commit, tree)
            (tree / ".run").mkdir()
            (tree / ".run" / "job.sh").write_text(JOB_SCRIPT)
            (tree / ".run" / "commands.sh").write_text(commands_script(self.exp.commands))
            self.say(f"copying the repo at {self.commit[:12]} (tracked files only)")
            self._setup_step("copying the repo", setup_end,
                             lambda t: self.remote.push(tree, timeout=t))
        self.say("setup: uv, uv sync, GPU and versions")
        code, out = self._setup_step("the setup script", setup_end,
                                     lambda t: self.remote.run(self.setup_script, timeout=t))
        self._job_output(out.encode())
        st.versions = parse_setup(out)
        self._record()
        if code:
            raise RuntimeError(f"setup failed (exit {code})")

    def _setup_step(self, what: str, setup_end: float, fn: Callable[[float], object]):
        """``fn(timeout)``, the timeout being what is left of setup's own limit (and never
        past the deadline); running out is a failure that says which limit it was."""
        own = setup_end <= self.deadline
        left = min(setup_end, self.deadline) - time.time()
        if left <= 0:
            if own:
                raise RuntimeError(f"setup did not finish within {self.setup_timeout_s / 60:g} "
                                   f"minutes (before {what})")
            raise TimeLimit()
        try:
            return fn(max(1.0, left))
        except subprocess.TimeoutExpired:
            if own:
                raise RuntimeError(f"setup did not finish within {self.setup_timeout_s / 60:g} "
                                   f"minutes ({what} timed out)") from None
            raise TimeLimit() from None

    def _check_instance(self) -> None:
        """Between polls: is the instance still up and healthy? An API error is only noted."""
        try:
            inst = self.api.instance(self.state.instance_id)
        except ApiError as e:
            self.say(f"instance status check failed: {e}")
            return
        status = (inst or {}).get("status")
        if inst is None or status in ENDED or status in BROKEN:
            raise RuntimeError(f"instance status is {status or 'gone (404)'!r}")

    def _job(self) -> None:
        st = self.state
        secs = remote_seconds(self.deadline, time.time(), self.margin_s)
        st.status = "running commands"
        self.say(f"starting the commands; remote hard limit {secs} s")
        t_start = time.time()  # before the remote job starts: its elapsed time is at most ours
        self.remote.start(secs)
        offset, failures = 0, 0
        failing_since: float | None = None
        last_check = time.time()
        pending = ""  # the job output's unfinished last line
        while True:
            self._check_deadline()
            try:
                chunk, code = self.remote.poll(offset)
                failures, failing_since = 0, None
            except Exception as e:  # noqa: BLE001 - a network blip; keep trying a while
                failures += 1
                failing_since = failing_since or time.time()
                self.say(f"poll failed ({failures} in a row): {type(e).__name__}")
                if time.time() - failing_since > self.ssh_wait_s:
                    raise RuntimeError(
                        f"the instance stopped answering SSH: polls failed for "
                        f"{self.ssh_wait_s / 60:g} minutes in a row ({failures} polls)") from None
                chunk, code = b"", None
            if chunk:
                offset += len(chunk)
                self._job_output(chunk)
                marks, pending = mark_lines(pending, chunk.decode("utf-8", "replace"))
                if marks and code is None:
                    self._checkpoint()
            if code == DIED:
                raise RuntimeError("the job on the instance ended without an exit code "
                                   "(killed, or the instance restarted)")
            if code is not None:
                st.remote_exit = code
                st.status = exit_status(code, time.time() - t_start, secs)
                self.say(f"commands finished: exit {code}")
                return
            if time.time() - last_check >= self.status_check_s:
                last_check = time.time()
                self._check_instance()
            time.sleep(max(0.0, min(self.poll_s, self.deadline - time.time())))

    # -- always ----------------------------------------------------------------------------

    def _cleanup(self, old_handlers: dict) -> bool | None:
        """Stop, fetch, terminate, record, final check. Every step is guarded: whatever one
        of them raises, the next still runs. Returns the final check's answer (``True``:
        nothing of this run is up; ``False``: something is; ``None``: couldn't tell)."""
        st = self.state
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, signal.SIG_IGN)
            except ValueError:
                break
        self.say(f"status: {st.status}. Fetching and terminating; Ctrl-C is ignored until "
                 "Lambda confirms the instance is gone")
        if self.remote is not None:
            if st.status != "ok":
                self._guard("stopping the job", lambda: self.remote.stop(self.stop_wait_s))
            self._guard("fetch", self._fetch)
        ids: list[str] | None = None
        try:
            ids = self._ids_to_terminate()
        except Exception as e:  # noqa: BLE001
            self.say(f"could not work out what to terminate: {type(e).__name__}: {e}")
        if ids:
            st.existed = True
            try:
                st.terminated = self._terminate(ids)
            except Exception as e:  # noqa: BLE001 - not expected: terminate_and_confirm catches
                self.say(f"terminate raised {type(e).__name__}: {e}")
                st.terminated = False
            st.t_end = time.time()
        elif ids is not None:
            st.terminated = True  # nothing was started (or the launch started nothing)
            st.t_end = time.time()
        self._guard("writing run.json / ledger", self._record)
        clean = None
        try:
            clean = self._final_check()
        except Exception as e:  # noqa: BLE001
            self.say(f"final check raised {type(e).__name__}: {e}")
        if clean is None and st.launch_uncertain and not st.existed:
            # a launch whose answer never came, and no list since: it may be up and billing
            st.terminated = False
        self._guard("writing run.json / ledger", self._record)
        if self._caffeinate is not None:
            self._guard("stopping caffeinate", self._caffeinate.terminate)
        if self._log:
            self._log.close()
            self._log = None
        for sig, h in old_handlers.items():
            try:
                signal.signal(sig, h)
            except (ValueError, TypeError):
                pass
        return clean

    def _guard(self, what: str, fn: Callable[[], object]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - the cleanup goes on
            self.say(f"{what} failed: {type(e).__name__}: {e}")

    def _ids_to_terminate(self) -> list[str] | None:
        st = self.state
        if st.instance_id:
            return [st.instance_id]
        if not st.launch_attempted or st.launch_refused:
            return []  # no launch call, or Lambda refused it (4xx): nothing started
        found = self._find_by_name()  # None: couldn't tell
        if found:
            st.instance_id = found[0]
        return found

    def _checkpoint(self) -> None:
        """The job asked for a fetch (:data:`CHECKPOINT_MARK`): copy what it has written so
        far, while it goes on. Never past the deadline; a failure is said and the run goes on
        (the final fetch tries again)."""
        st = self.state
        budget = min(FETCH_TIMEOUT_S, self.deadline - time.time())
        if budget < CHECKPOINT_MIN_S:
            self.say("checkpoint: too close to the deadline to fetch; going on")
            return
        try:
            files, copied = self._remote_fetch(budget)
        except Exception as e:  # noqa: BLE001 - the job goes on; the final fetch tries again
            self.say(f"checkpoint fetch failed: {type(e).__name__}: {e}; going on")
            return
        st.checkpoints += 1
        self._set_fetched(list(st.have))
        self.say(f"checkpoint {st.checkpoints}: fetched {len(copied)} files "
                 f"({len(files) - len(copied)} unchanged) into {st.run_dir.relative_to(self.root)}")
        self._guard("writing run.json / ledger", self._record)

    def _remote_fetch(self, budget: float) -> tuple[list[str], list[str]]:
        """One fetch: marker N+1, files changed after the last completed fetch's marker
        (or not fetched yet). Only a fetch that completes moves ``fetched_mark``, so a
        failed one leaves its files to the next."""
        st = self.state
        st.fetch_marks += 1
        files, copied = self.remote.fetch(self.exp.fetch, st.run_dir, timeout=budget,
                                          mark=st.fetch_marks, since=st.fetched_mark,
                                          have=frozenset(st.have))
        st.fetched_mark = st.fetch_marks
        st.have.update(f for f in copied if (st.run_dir / f).is_file())
        return files, copied

    def _set_fetched(self, files: list[str]) -> None:
        st = self.state
        st.fetched = [{"path": f, "sha256": sha256_file(st.run_dir / f),
                       "bytes": (st.run_dir / f).stat().st_size}
                      for f in sorted(set(files)) if (st.run_dir / f).is_file()]

    def _fetch(self) -> None:
        st = self.state
        budget = fetch_budget(self.hard_end, time.time(),
                              min(TERMINATE_ALLOW_S, self.reserve_s / 2)) if st.t_launch else 0
        files: list[str] = []
        if budget <= 0:
            self.say("no time left to fetch before the hard limit: terminating without it")
        else:
            try:
                files, _ = self._remote_fetch(budget)
            except Exception as e:  # noqa: BLE001
                self.say(f"fetch failed: {type(e).__name__}: {e}")
        # what checkpoints fetched counts too (if this fetch failed, those copies are what's left)
        self._set_fetched([*files, *st.have])
        got = [f["path"] for f in st.fetched]
        st.missing = [p for p in self.exp.fetch if not any(Path(g).match(p) for g in got)]
        self.say(f"fetched {len(st.fetched)} files this run wrote into "
                 f"{st.run_dir.relative_to(self.root)}"
                 + (f"; nothing written matched {st.missing}" if st.missing else ""))

    def _find_by_name(self) -> list[str] | None:
        """After a launch call whose answer was lost (5xx, no answer): did it start one
        anyway? A new instance can take a while to appear in the list, so look for
        ``lost_launch_s``. ``None`` if the API never answered (then the run stays charged
        at its worst case and exits 3)."""
        st = self.state
        t0, answered = time.time(), False
        while True:
            try:
                found = [i["id"] for i in self.api.instances()
                         if self._is_this_run(i) and live(i)]
                answered = True
                if found:
                    st.seen.update(found)
                    self.say(f"the launch call's answer was lost, but {st.instance_name} is "
                             f"listed: {', '.join(found)}")
                    return found
            except ApiError as e:
                self.say(f"listing instances failed: {e}")
            if time.time() - t0 >= self.lost_launch_s:
                break
            time.sleep(self.boot_poll_s)
        if not answered:
            self.say("COULD NOT LIST INSTANCES after a lost launch: check the dashboard NOW")
            return None
        self.say(f"{st.instance_name} not listed after {self.lost_launch_s:.0f} s: taking it "
                 "that the launch started nothing (the final check looks once more)")
        return []

    def _terminate(self, ids: list[str]) -> bool:
        return terminate_and_confirm(self.api, ids, self.say, self.boot_poll_s, self.confirm_s,
                                     resend_s=min(self.resend_s, self.confirm_s / 4),
                                     seen=self.state.seen)

    def _final_check(self) -> bool | None:
        """List the account. An instance of this run that is still up is terminated here
        (a lost launch that appeared late); any other ``tinytools-*`` is reported."""
        st = self.state
        rows = None
        for _ in range(3):
            try:
                rows = self.api.instances()
                break
            except ApiError as e:
                self.say(f"listing instances failed: {e}")
                time.sleep(self.boot_poll_s)
        if rows is None:
            self.say("LIST FAILED: check the dashboard; an instance may be billing")
            return None
        mine = [i for i in rows if self._is_this_run(i) and live(i)]
        others = [i for i in rows if is_ours(i) and live(i) and not self._is_this_run(i)]
        clean = True
        if mine:
            ids = [i["id"] for i in mine]
            self.say(f"STILL UP (this run's): {', '.join(ids)}: terminating")
            st.seen.update(ids)
            st.existed = True
            ok = self._terminate(ids)
            if not st.instance_id:
                st.instance_id = ids[0]
            st.terminated = ok
            st.t_end = time.time()
            clean = ok
        for i in others:
            self.say(f"STILL UP: {i['id']} {i.get('name')} {i.get('status')}: not this run's; "
                     f"run `scripts/lambda.sh down {i['id']}` now")
            clean = False
        if clean:
            self.say("list: no instance of this repo is up")
        return clean


def terminate_and_confirm(api: LambdaAPI, ids: list[str], say: Callable[[str], None],
                          poll_s: float, confirm_s: float, *, resend_s: float | None = None,
                          seen=()) -> bool:
    """Terminate through the API and poll until Lambda reports every id ended
    (terminated or preempted), for up to ``confirm_s``.

    Terminate is retried after every failure; while an instance is still booting
    (``actions.terminate.available`` false) it waits; and after an accepted call it is
    re-sent every ``resend_s`` until confirmed, since a 200 is not proof. A 404 means gone
    only for an id Lambda has shown (``seen``, or a status read here): just after a
    launch, a new instance can be missing from the API for a while."""
    resend_s = min(RESEND_S, confirm_s / 4) if resend_s is None else resend_s
    seen = set(seen)
    t0 = time.time()
    last_sent: float | None = None
    failures, waiting_said = 0, False
    while True:
        pending, blocked = list(ids), []
        try:
            views = {i: api.instance(i) for i in ids}
        except Exception as e:  # noqa: BLE001 - ApiError, or anything else: keep going
            say(f"status check failed: {e}")
            views = None
        if views is not None:
            seen.update(i for i, v in views.items() if v is not None)
            pending = [i for i in ids
                       if (views[i] is None and i not in seen)
                       or (views[i] is not None and views[i].get("status") not in ENDED)]
            if not pending:
                say("terminated: Lambda confirms; billing has stopped")
                return True
            blocked = [i for i in pending
                       if views[i] is not None and not terminate_available(views[i])]
        sendable = [i for i in pending if i not in blocked]
        now = time.time()
        if sendable and (last_sent is None or now - last_sent >= resend_s):
            try:
                accepted = api.terminate(sendable)
                seen.update(accepted)  # Lambda named them: a 404 from now on means gone
                say(f"terminate {'re-sent' if last_sent else 'requested'} for "
                    f"{', '.join(sendable)}"
                    + ("" if set(sendable) <= set(accepted) else
                       f" (Lambda listed {', '.join(accepted) or 'none'} as terminating)"))
                last_sent = now
            except Exception as e:  # noqa: BLE001 - ApiError, or anything else: retry
                failures += 1
                say(f"terminate failed (try {failures}): {e}")
                if failures == TERMINATE_TRIES:
                    say(f"STILL NOT TERMINATED after {failures} tries: it may be billing. "
                        "The runner keeps trying; check the dashboard too")
        elif blocked and not sendable and not waiting_said:
            why = ((views[blocked[0]].get("actions") or {}).get("terminate") or {})
            say(f"{', '.join(blocked)}: Lambda can't terminate it until it has launched "
                f"({why.get('reason_code') or 'booting'}); waiting")
            waiting_said = True
        if time.time() - t0 >= confirm_s:
            say(f"TERMINATION OF {', '.join(ids)} NOT CONFIRMED after {confirm_s:.0f} s: it may "
                f"be billing. Run `scripts/lambda.sh down {ids[0]}` or use the dashboard NOW")
            return False
        time.sleep(poll_s)


# -- the other commands ------------------------------------------------------------------------


def cmd_types(api: LambdaAPI) -> int:
    types = api.instance_types()
    rows = sorted(((v["instance_type"]["price_cents_per_hour"], k,
                    ", ".join(r["name"] for r in v.get("regions_with_capacity_available", []))
                    or "-") for k, v in types.items()))
    print(f"{'type':<26} {'$/h':>7}  regions with capacity")
    for cents, name, regions in rows:
        print(f"{name:<26} {cents / 100:>7.2f}  {regions}")
    return 0


def cmd_estimate(api: LambdaAPI, root: Path, name: str) -> int:
    exp, _ = load_experiment(root, name)
    types = api.instance_types()
    for line in same_gpu_lines(types, exp.instance_type):  # e.g. A100 40 GB beside 80 GB
        print(line)
    est = make_estimate(exp, types, read_ledger(root / "runs" / "ledger.csv"))
    for line in est.lines():
        print(line)
    return EXIT_OK if est.within_cap else EXIT_REFUSED


def cmd_list(api: LambdaAPI) -> int:
    rows = api.instances()
    if not rows:
        print("no instances in this account")
    for i in rows:
        t = i.get("instance_type") or {}
        flag = "  <- tinytools" if is_ours(i) else ""
        print(f"{i['id']}  {i.get('status', '?'):<11} {t.get('name', '?'):<20} "
              f"${t.get('price_cents_per_hour', 0) / 100:.2f}/h  {i.get('name') or '-'}{flag}")
    return 0


def cmd_down(api: LambdaAPI, instance_id: str, yes: bool, ask=input,
             poll_s: float = BOOT_POLL_S, confirm_s: float = CONFIRM_S) -> int:
    inst = api.instance(instance_id)
    if inst is None or inst.get("status") in ENDED:
        print(f"{instance_id}: already ended or unknown")
        return EXIT_OK
    if not yes and ask(f"Terminate {instance_id} ({inst.get('name') or '-'})? Anything not "
                       "fetched is lost. Type yes: ").strip().lower() != "yes":
        print("not terminated")
        return EXIT_FAILED
    ok = terminate_and_confirm(api, [instance_id], lambda m: print(m, file=sys.stderr),
                               poll_s, confirm_s, seen={instance_id})
    return EXIT_OK if ok else EXIT_UNCONFIRMED


def _fingerprint(pub_text: str) -> str | None:
    r = subprocess.run(["ssh-keygen", "-lf", "-"], input=pub_text, capture_output=True, text=True)
    return r.stdout.split()[1] if r.returncode == 0 and r.stdout.split() else None


def doctor_checks(root: Path, env: dict[str, str], api: LambdaAPI) -> list[tuple[bool, str]]:
    """Everything a launch needs, checked without launching. Values are never shown."""
    out: list[tuple[bool, str]] = []
    out.append(((root / "lambda.env").exists(), "lambda.env present (gitignored)"))
    for k, line in zip(ENV_KEYS, describe_env(env), strict=True):
        out.append((bool(env.get(k)), line))
    for tool in ("ssh", "ssh-keygen", "rsync", "git"):
        out.append((shutil.which(tool) is not None, f"{tool} installed"))
    if sys.platform == "darwin":
        out.append((shutil.which("caffeinate") is not None, "caffeinate installed"))
    key = os.path.expanduser(env.get("LAMBDA_SSH_KEY", ""))
    mine = None
    if key:
        out.append((not key.endswith(".pub"), "LAMBDA_SSH_KEY is the private key (not .pub)"))
        out.append((os.path.isfile(key), "the private key file exists"))
        pub = key.removesuffix(".pub") + ".pub"
        if os.path.isfile(pub):
            mine = _fingerprint(Path(pub).read_text())
        elif os.path.isfile(key):
            r = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", key], capture_output=True,
                               text=True)
            mine = _fingerprint(r.stdout) if r.returncode == 0 else None
        out.append((mine is not None, "its public key is readable (fingerprint computed)"))
        if os.path.isfile(key):
            no_pass = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", key],
                                     capture_output=True).returncode == 0
            agent = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True).stdout
            out.append((no_pass or (mine is not None and mine in agent),
                        "usable without a prompt (no passphrase, or loaded in ssh-agent)"))
    try:
        types = api.instance_types()
        out.append((True, f"API reachable at {api.base} ({len(types)} instance types listed)"))
        keys = api.ssh_keys()
        named = [k for k in keys if k.get("name") == env.get("LAMBDA_SSH_KEY_NAME")]
        out.append((bool(named), "an SSH key with LAMBDA_SSH_KEY_NAME is registered at Lambda"))
        if named and mine:
            out.append((_fingerprint(named[0]["public_key"]) == mine,
                        "it is this laptop's key (fingerprints match)"))
        up = [i for i in api.instances() if is_ours(i) and live(i)]
        out.append((not up, f"no instance of this repo is up ({len(up)} up)"))
    except ApiError as e:
        out.append((False, f"API at {api.base}: {e}"))
    return out


def cmd_doctor(root: Path, env: dict[str, str], api: LambdaAPI) -> int:
    checks = doctor_checks(root, env, api)
    for ok, label in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {api.redact(label)}")
    bad = sum(not ok for ok, _ in checks)
    print("all checks pass; nothing was launched" if not bad else f"{bad} checks failed")
    return 0 if not bad else 1


def make_api(env: dict[str, str], min_interval: float | None = None,
             rate_wait: float = RATE_WAIT_S) -> LambdaAPI:
    """The API client. Requests to Lambda itself are spaced 1 s apart (its limit); a
    local fake is not paced unless asked. A 429 is retried either way."""
    base = env.get("LAMBDA_API_URL") or API_URL
    if min_interval is None:
        min_interval = API_MIN_INTERVAL_S if _is_lambda_host(base) else 0.0
    redact = Redactor([env.get("LAMBDA_API_KEY", ""), env.get("LAMBDA_SSH_KEY", "")])
    return LambdaAPI(base, env.get("LAMBDA_API_KEY", ""), redact, min_interval=min_interval,
                     rate_wait=rate_wait)


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    p = argparse.ArgumentParser(prog="scripts/lambda.sh", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("types")
    s = sub.add_parser("estimate")
    s.add_argument("experiment")
    s = sub.add_parser("run")
    s.add_argument("experiment")
    s.add_argument("--approve", type=float, default=None,
                   help="the approved worst case in USD; refuses if the estimate is higher")
    sub.add_parser("list")
    s = sub.add_parser("down")
    s.add_argument("instance_id")
    s.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)

    env = load_env(root)
    api = make_api(env)
    try:
        if args.command == "doctor":
            return cmd_doctor(root, env, api)
        if args.command == "types":
            return cmd_types(api)
        if args.command == "estimate":
            return cmd_estimate(api, root, args.experiment)
        if args.command == "list":
            return cmd_list(api)
        if args.command == "down":
            return cmd_down(api, args.instance_id, args.yes)
        exp, path = load_experiment(root, args.experiment)
        known = root / ".lambda" / "known_hosts"
        known.parent.mkdir(exist_ok=True)
        known.write_text("")  # a new instance each run: forget the last one's host key
        runner = Runner(root, exp, path, api, env,
                        remote_factory=lambda ip: SSHRemote(ip, env.get("LAMBDA_SSH_KEY", ""),
                                                            known),
                        approve=args.approve, ask=input if sys.stdin.isatty() else None)
        return runner.run()
    except (Refused, ApiError) as e:
        print(f"refused: {api.redact(str(e))}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
