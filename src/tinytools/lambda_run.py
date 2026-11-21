"""Run an experiment on Lambda Cloud from this laptop, as ``docs/lambda-playbook.md`` describes.

    scripts/lambda.sh doctor               settings, SSH key, agent, API; launches nothing
    scripts/lambda.sh types                instance types, prices, regions with capacity
    scripts/lambda.sh estimate EXP         price x hours, the experiment's cap left; refuses past it
    scripts/lambda.sh run EXP [--approve USD]
                                           up -> setup -> run -> fetch -> down, terminated always
    scripts/lambda.sh list                 every instance; flags this repo's
    scripts/lambda.sh down ID [--yes]      terminate one instance and wait until Lambda confirms

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
value from it; :class:`Redactor` scrubs them from every line written anyway.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
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
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://cloud.lambda.ai/api/v1"
NAME_PREFIX = "tinytools"  # every instance this repo launches is named tinytools-<exp>-<stamp>
SSH_USER = "ubuntu"
REMOTE_DIR = "tinytools"
ENV_KEYS = ("LAMBDA_API_KEY", "LAMBDA_SSH_KEY_NAME", "LAMBDA_SSH_KEY")
OPTIONAL_ENV = ("LAMBDA_API_URL", "LAMBDA_REGION")

RESERVE_S = 600  # kept back from max_hours for fetch + terminate
REMOTE_MARGIN_S = 30  # the remote `timeout` fires this long before the laptop's deadline
BOOT_TIMEOUT_S = 20 * 60
POLL_S = 15
BOOT_POLL_S = 10
TERMINATE_TRIES = 20
CONFIRM_S = 600
FETCH_TIMEOUT_S = 420
TIMED_OUT = (124, 137)  # GNU timeout: TERM sent (124), or KILL after --kill-after (137)
LEDGER_FIELDS = ("run", "experiment", "commit", "instance_type", "price_usd_per_hour",
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
    post: int | None = None
    setup_record: str | None = None


_TYPES = {"name": str, "post": int, "instance_type": str, "max_hours": (int, float),
          "expected_hours": (int, float), "cap_usd": (int, float), "commands": list,
          "fetch": list, "setup_record": str}
_REQUIRED = ("name", "instance_type", "max_hours", "expected_hours", "cap_usd", "commands",
             "fetch")


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
        if not FETCH_PATTERN.fullmatch(p) or p.startswith("/") or ".." in p.split("/"):
            raise ValueError(f"fetch pattern {p!r}: relative paths of [A-Za-z0-9_.*?/-] only, "
                             "no '..'")
    rec = d.get("setup_record")
    if rec is not None and not any(Path(rec).match(p) for p in fetch):
        raise ValueError(f"setup_record {rec!r} is not matched by any fetch pattern")
    return Experiment(d["name"], d["instance_type"], float(d["max_hours"]),
                      float(d["expected_hours"]), float(d["cap_usd"]), tuple(cmds), tuple(fetch),
                      d.get("post"), rec)


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
    """Replaces every secret value (and any ``Bearer ...`` token) in a string."""

    def __init__(self, secrets: list[str]) -> None:
        self.secrets = sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "[redacted]")
        return re.sub(r"(?i)bearer\s+(?!\[redacted\])\S+", "Bearer [redacted]", text)


# -- the API ---------------------------------------------------------------------------


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


def _error_text(body: bytes) -> str:
    try:
        err = json.loads(body).get("error", {})
        return f"{err.get('code', '')}: {err.get('message', '')}".strip(": ")
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")[:200]


class LambdaAPI:
    """The few Lambda Cloud API calls the runner needs. Errors are redacted."""

    def __init__(self, base: str, key: str, redact: Redactor, timeout: float = 30) -> None:
        self.base = base.rstrip("/")
        self._key = key
        self.redact = redact
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None):
        if not self._key:
            raise ApiError(None, "LAMBDA_API_KEY is not set (lambda.env)")
        req = urllib.request.Request(
            self.base + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                     "Content-Type": "application/json", "User-Agent": "tinytools-runner"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, self.redact(
                f"{method} {path}: HTTP {e.code}: {_error_text(e.read())}")) from None
        except (urllib.error.URLError, OSError) as e:
            raise ApiError(None, self.redact(f"{method} {path}: {e}")) from None
        try:
            return json.loads(payload)["data"]
        except (ValueError, KeyError, TypeError):
            raise ApiError(None, f"{method} {path}: unexpected response") from None

    def instance_types(self) -> dict:
        return self._call("GET", "/instance-types")

    def instances(self) -> list[dict]:
        return self._call("GET", "/instances")

    def instance(self, instance_id: str) -> dict | None:
        """The instance, or ``None`` once Lambda no longer knows it (404)."""
        try:
            return self._call("GET", f"/instances/{instance_id}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def launch(self, instance_type: str, region: str, ssh_key_name: str, name: str) -> str:
        data = self._call("POST", "/instance-operations/launch", {
            "region_name": region, "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key_name], "name": name})
        return data["instance_ids"][0]

    def terminate(self, ids: list[str]) -> None:
        self._call("POST", "/instance-operations/terminate", {"instance_ids": ids})

    def ssh_keys(self) -> list[dict]:
        return self._call("GET", "/ssh-keys")


def is_ours(instance: dict) -> bool:
    return str(instance.get("name") or "").startswith(NAME_PREFIX)


def live(instance: dict) -> bool:
    return instance.get("status") != "terminated"


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


def write_ledger_row(path: Path, row: dict) -> None:
    """Add ``row``, or replace the row with the same ``run``."""
    rows = [r for r in read_ledger(path) if r["run"] != row["run"]]
    rows.append({k: row.get(k, "") for k in LEDGER_FIELDS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


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
    def within_cap(self) -> bool:
        return self.worst <= self.remaining

    def lines(self) -> list[str]:
        return [
            f"experiment          {self.experiment}",
            f"instance type       {self.instance_type}",
            f"price (API, now)    ${self.price:.2f}/h",
            f"capacity now        {', '.join(self.regions) or 'none'}",
            f"expected            {self.expected_hours:g} h = ${self.expected:.2f}",
            f"worst case          {self.max_hours:g} h (hard limit) = ${self.worst:.2f}",
            f"cap                 ${self.cap:.2f}, spent ${self.spent:.2f}, "
            f"left ${self.remaining:.2f}",
            "verdict             " + ("within the cap" if self.within_cap else
                                      "REFUSED: the worst case is more than the cap left"),
        ]


def make_estimate(exp: Experiment, types: dict, ledger: list[dict]) -> Estimate:
    if exp.instance_type not in types:
        near = sorted(n for n in types if "h100" in n or exp.instance_type.split("_")[-1] in n)
        raise Refused(f"instance type {exp.instance_type} is not in Lambda's list now"
                      + (f"; similar: {', '.join(near)}" if near else ""))
    item = types[exp.instance_type]
    regions = tuple(r["name"] for r in item.get("regions_with_capacity_available", []))
    return Estimate(exp.name, exp.instance_type,
                    item["instance_type"]["price_cents_per_hour"] / 100, exp.max_hours,
                    exp.expected_hours, spent(ledger, exp.name), exp.cap_usd, regions)


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

# Started detached (nohup setsid), so a dropped SSH connection doesn't stop it. $1 is the
# hard limit in seconds; `timeout` stops the work then (it does not stop the bill).
JOB_SCRIPT = """#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo $$ > .run/pid
export PATH="$HOME/.local/bin:$PATH"
timeout --signal=TERM --kill-after=60 "$1" bash .run/commands.sh
echo $? > .run/exit.tmp && mv .run/exit.tmp .run/exit
"""


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


def remote_seconds(deadline: float, now: float, margin: float) -> int:
    """The remote `timeout`, so the work stops ``margin`` before the laptop's deadline."""
    return max(1, int(deadline - now - margin))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- the remote side over SSH ---------------------------------------------------------------


class SSHRemote:
    """The instance, reached with ssh and rsync (BatchMode: never a prompt)."""

    def __init__(self, ip: str, key_path: str, known_hosts: Path) -> None:
        self.host = f"{SSH_USER}@{ip}"
        self.opts = ["-i", os.path.expanduser(key_path), "-o", "BatchMode=yes",
                     "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
                     "-o", f"UserKnownHostsFile={known_hosts}", "-o", "ServerAliveInterval=30",
                     "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=10"]

    def _ssh(self, command: str, timeout: float, stdin: str | None = None):
        return subprocess.run(["ssh", *self.opts, self.host, command], input=stdin,
                              capture_output=True, text=True, timeout=timeout)

    def ready(self) -> bool:
        try:
            return self._ssh("true", 30).returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def push(self, src: Path, timeout: float) -> None:
        ssh = " ".join(shlex.quote(x) for x in ["ssh", *self.opts])
        r = subprocess.run(["rsync", "-az", "-e", ssh, f"{src}/", f"{self.host}:{REMOTE_DIR}/"],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"rsync to the instance failed: {r.stderr.strip()[-300:]}")

    def run(self, script: str, timeout: float) -> tuple[int, str]:
        r = self._ssh("bash -s", timeout, stdin=script)
        return r.returncode, r.stdout + r.stderr

    def start(self, seconds: int) -> None:
        # Not `cd x && nohup ... &`: that backgrounds the whole `&&` list in a subshell
        # which keeps this session's stdout open, so ssh would wait for the job to end.
        r = self._ssh(f"cd {REMOTE_DIR} || exit 1; nohup setsid bash .run/job.sh {seconds} "
                      "> .run/log.txt 2>&1 < /dev/null & echo started", 60)
        if "started" not in r.stdout:
            raise RuntimeError(f"could not start the job: {r.stderr.strip()[-300:]}")

    def poll(self, offset: int) -> tuple[bytes, int | None]:
        r = subprocess.run(["ssh", *self.opts, self.host,
                            f"cd {REMOTE_DIR} && (cat .run/exit 2>/dev/null || echo running) "
                            f"&& tail -c +{offset + 1} .run/log.txt"],
                           capture_output=True, timeout=60)
        if r.returncode:
            raise RuntimeError("poll failed")
        head, _, rest = r.stdout.partition(b"\n")
        code = head.strip().decode()
        return rest, None if code == "running" else int(code)

    def stop(self) -> None:
        self._ssh(f"cd {REMOTE_DIR} && [ -f .run/pid ] && kill -TERM -- -$(cat .run/pid)", 30)

    def fetch(self, patterns: tuple[str, ...], dest: Path, timeout: float) -> list[str]:
        r = self._ssh(f"cd {REMOTE_DIR} && shopt -s nullglob && for f in {' '.join(patterns)}; "
                      'do [ -f "$f" ] && echo "$f"; done; true', 60)
        files = [f for f in r.stdout.splitlines() if f.strip()]
        if not files:
            return []
        dest.mkdir(parents=True, exist_ok=True)
        # a tar stream over ssh: macOS ships openrsync, whose --files-from with a remote
        # source is not something to find out about mid-run
        tar = subprocess.run(["ssh", *self.opts, self.host,
                              f"cd {REMOTE_DIR} && tar -cf - " + " ".join(map(shlex.quote, files))],
                             capture_output=True, timeout=timeout, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            t.extractall(dest, filter="data")
        return files


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
    region: str = ""
    launch_attempted: bool = False
    existed: bool = False  # an instance id was known (so it may have billed)
    t_launch: float | None = None
    t_end: float | None = None
    terminated: bool = False
    status: str = "not started"
    remote_exit: int | None = None
    versions: dict = field(default_factory=dict)
    fetched: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


class Runner:
    """``run EXP``: every check, then up -> setup -> run -> fetch -> down."""

    def __init__(self, root: Path, exp: Experiment, exp_path: Path, api: LambdaAPI,
                 env: dict[str, str], *, remote_factory: Callable[[str], object],
                 approve: float | None = None, setup_script: str = SETUP_SCRIPT,
                 poll_s: float = POLL_S, boot_poll_s: float = BOOT_POLL_S,
                 reserve_s: float = RESERVE_S, margin_s: float = REMOTE_MARGIN_S,
                 confirm_s: float = CONFIRM_S, keep_awake: bool = True,
                 stream=None, ask: Callable[[str], str] | None = None) -> None:
        self.root, self.exp, self.exp_path, self.api, self.env = root, exp, exp_path, api, env
        self.remote_factory = remote_factory
        self.approve = approve
        self.setup_script = setup_script
        self.poll_s, self.boot_poll_s = poll_s, boot_poll_s
        self.reserve_s, self.margin_s, self.confirm_s = reserve_s, margin_s, confirm_s
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
        if not est.within_cap:
            raise Refused(f"the worst case ${est.worst:.2f} is more than the ${est.remaining:.2f}"
                          f" left of {self.exp.name}'s ${est.cap:.2f} cap")
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
            return 2
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
        self._awake()
        try:
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
            self._cleanup(old)
        return 0 if st.status == "ok" else 1

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
        """caffeinate: the Mac must not sleep while it is the one that terminates."""
        if self.keep_awake and shutil.which("caffeinate"):
            self._caffeinate = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
            self.say("caffeinate on: this Mac will not idle-sleep during the run")

    @property
    def deadline(self) -> float:
        return self.state.t_launch + self.exp.max_hours * 3600 - self.reserve_s

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
            "instance": {"id": st.instance_id, "name": st.instance_name,
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
            "fetched": st.fetched, "not_found": st.missing,
        }
        (st.run_dir / "run.json").write_text(self.redact(json.dumps(data, indent=2)) + "\n")
        write_ledger_row(self.root / "runs" / "ledger.csv", {
            "run": st.run_dir.name, "experiment": self.exp.name, "commit": self.commit[:12],
            "instance_type": self.exp.instance_type, "price_usd_per_hour": f"{self.price:.2f}",
            "launched_at": stamp(st.t_launch) or "", "ended_at": stamp(st.t_end) or "",
            "hours": f"{hours:.4f}", "cost_usd": f"{cost:.4f}", "status": st.status})

    def _setup_record(self) -> dict:
        rec = self.exp.setup_record
        path = self.state.run_dir / rec if rec else None
        if path and path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                return {"error": "unreadable"}
        return {}

    def _launch(self, region: str) -> None:
        st = self.state
        st.region = region
        st.instance_name = f"{NAME_PREFIX}-{self.exp.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        st.status = "launching"
        st.t_launch = time.time()  # billing may start from here: count it
        self._record()
        self.say(f"launching {self.exp.instance_type} in {region} as {st.instance_name}")
        st.launch_attempted = True
        st.instance_id = self.api.launch(self.exp.instance_type, region,
                                         self.env.get("LAMBDA_SSH_KEY_NAME", ""), st.instance_name)
        st.existed = True
        st.status = "running"
        self._record()
        self.say(f"instance {st.instance_id}: billing has started; hard limit "
                 f"{self.exp.max_hours:g} h")

    def _wait_active(self) -> str:
        t0 = time.time()
        while True:
            self._check_deadline()
            inst = self.api.instance(self.state.instance_id)
            status = (inst or {}).get("status")
            if status == "active" and inst.get("ip"):
                self.say(f"active at {inst['ip']}")
                return inst["ip"]
            if status not in ("booting", "active"):
                raise RuntimeError(f"instance status is {status!r}")
            if time.time() - t0 > BOOT_TIMEOUT_S:
                raise RuntimeError("not active after 20 minutes")
            time.sleep(self.boot_poll_s)

    def _wait_ssh(self) -> None:
        self.say("waiting for SSH")
        while not self.remote.ready():
            self._check_deadline()
            time.sleep(self.boot_poll_s)

    def _setup(self) -> None:
        st = self.state
        st.status = "setup"
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "tree"
            tree.mkdir()
            export_tree(self.root, self.commit, tree)
            (tree / ".run").mkdir()
            (tree / ".run" / "job.sh").write_text(JOB_SCRIPT)
            (tree / ".run" / "commands.sh").write_text(commands_script(self.exp.commands))
            self.say(f"copying the repo at {self.commit[:12]} (tracked files only)")
            self.remote.push(tree, timeout=max(1, self.deadline - time.time()))
        self.say("setup: uv, uv sync, GPU and versions")
        code, out = self.remote.run(self.setup_script, timeout=max(1, self.deadline - time.time()))
        self._job_output(out.encode())
        st.versions = parse_setup(out)
        self._record()
        if code:
            raise RuntimeError(f"setup failed (exit {code})")

    def _job(self) -> None:
        st = self.state
        secs = remote_seconds(self.deadline, time.time(), self.margin_s)
        st.status = "running commands"
        self.say(f"starting the commands; remote hard limit {secs} s")
        self.remote.start(secs)
        offset, failures = 0, 0
        while True:
            self._check_deadline()
            try:
                chunk, code = self.remote.poll(offset)
                failures = 0
            except Exception as e:  # noqa: BLE001 - a network blip; keep trying until the deadline
                failures += 1
                self.say(f"poll failed ({failures} in a row): {type(e).__name__}")
                chunk, code = b"", None
            if chunk:
                offset += len(chunk)
                self._job_output(chunk)
            if code is not None:
                st.remote_exit = code
                st.status = ("ok" if code == 0 else "time limit" if code in TIMED_OUT
                             else f"failed: a command exited {code}")
                self.say(f"commands finished: exit {code}")
                return
            time.sleep(max(0.0, min(self.poll_s, self.deadline - time.time())))

    # -- always ----------------------------------------------------------------------------

    def _cleanup(self, old_handlers: dict) -> None:
        st = self.state
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, signal.SIG_IGN)
            except ValueError:
                break
        self.say(f"status: {st.status}. Fetching and terminating; Ctrl-C is ignored until "
                 "Lambda confirms the instance is gone")
        if self.remote is not None and st.status != "ok":
            try:
                self.remote.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.remote is not None:
            self._fetch()
        ids: list[str] | None = [st.instance_id] if st.instance_id else []
        if st.launch_attempted and not ids:
            ids = self._find_by_name()  # None: couldn't tell
            if ids:
                st.instance_id = ids[0]
        if ids:
            st.existed = True
            st.terminated = self._terminate(ids)
            st.t_end = time.time()
        elif ids is not None:
            st.terminated = True  # nothing was started (or the launch started nothing)
            st.t_end = time.time()
        try:
            self._record()
        except Exception as e:  # noqa: BLE001
            self.say(f"could not write run.json / ledger: {e}")
        if self._caffeinate is not None:
            self._caffeinate.terminate()
        self._final_check()
        if self._log:
            self._log.close()
            self._log = None
        for sig, h in old_handlers.items():
            signal.signal(sig, h)

    def _fetch(self) -> None:
        st = self.state
        try:
            files = self.remote.fetch(self.exp.fetch, st.run_dir, timeout=FETCH_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            self.say(f"fetch failed: {type(e).__name__}: {e}")
            files = []
        st.fetched = [{"path": f, "sha256": sha256_file(st.run_dir / f),
                       "bytes": (st.run_dir / f).stat().st_size}
                      for f in sorted(files) if (st.run_dir / f).is_file()]
        got = [f["path"] for f in st.fetched]
        st.missing = [p for p in self.exp.fetch if not any(Path(g).match(p) for g in got)]
        self.say(f"fetched {len(st.fetched)} files into {st.run_dir.relative_to(self.root)}"
                 + (f"; nothing matched {st.missing}" if st.missing else ""))

    def _find_by_name(self) -> list[str] | None:
        """After a launch call that failed or whose answer was lost: did it start one anyway?
        ``None`` if the API can't be asked (then the run stays charged at its worst case)."""
        for _ in range(5):
            try:
                found = [i["id"] for i in self.api.instances()
                         if i.get("name") == self.state.instance_name and live(i)]
                self.say(f"the launch call failed; instances named {self.state.instance_name}: "
                         f"{len(found)}")
                return found
            except ApiError as e:
                self.say(f"listing instances failed: {e}")
                time.sleep(self.boot_poll_s)
        self.say("COULD NOT LIST INSTANCES after a failed launch: check the dashboard NOW")
        return None

    def _terminate(self, ids: list[str]) -> bool:
        return terminate_and_confirm(self.api, ids, self.say, self.boot_poll_s, self.confirm_s)

    def _final_check(self) -> None:
        try:
            left = [i for i in self.api.instances() if is_ours(i) and live(i)]
        except ApiError as e:
            self.say(f"LIST FAILED ({e}): check the dashboard; an instance may be billing")
            return
        if left:
            for i in left:
                self.say(f"STILL UP: {i['id']} {i.get('name')} {i.get('status')}: run "
                         f"`scripts/lambda.sh down {i['id']}` now")
        else:
            self.say("list: no instance of this repo is up")


def terminate_and_confirm(api: LambdaAPI, ids: list[str], say: Callable[[str], None],
                          poll_s: float, confirm_s: float) -> bool:
    """Terminate through the API (retrying), then poll until every id is terminated or gone."""
    for attempt in range(TERMINATE_TRIES):
        try:
            api.terminate(ids)
            say(f"terminate requested for {', '.join(ids)}")
            break
        except ApiError as e:
            say(f"terminate failed (try {attempt + 1}/{TERMINATE_TRIES}): {e}")
            try:
                if all((api.instance(i) or {"status": "terminated"})["status"] == "terminated"
                       for i in ids):
                    return True
            except ApiError:
                pass
            time.sleep(min(30.0, poll_s * 2 ** attempt))
    else:
        say(f"COULD NOT TERMINATE {', '.join(ids)}: it may still be billing. Run "
            f"`scripts/lambda.sh down {ids[0]}` or use the dashboard NOW")
        return False
    t0 = time.time()
    while time.time() - t0 < confirm_s:
        try:
            states = [(api.instance(i) or {"status": "terminated"})["status"] for i in ids]
        except ApiError as e:
            say(f"status check failed: {e}")
            states = ["?"]
        if all(s == "terminated" for s in states):
            say("terminated: Lambda confirms; billing has stopped")
            return True
        time.sleep(poll_s)
    say(f"termination of {', '.join(ids)} not confirmed after {confirm_s:.0f} s; "
        "check `scripts/lambda.sh list`")
    return False


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
    est = make_estimate(exp, api.instance_types(), read_ledger(root / "runs" / "ledger.csv"))
    for line in est.lines():
        print(line)
    return 0 if est.within_cap else 2


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
             poll_s: float = BOOT_POLL_S) -> int:
    inst = api.instance(instance_id)
    if inst is None or inst.get("status") == "terminated":
        print(f"{instance_id}: already terminated or unknown")
        return 0
    if not yes and ask(f"Terminate {instance_id} ({inst.get('name') or '-'})? Anything not "
                       "fetched is lost. Type yes: ").strip().lower() != "yes":
        print("not terminated")
        return 1
    ok = terminate_and_confirm(api, [instance_id], lambda m: print(m, file=sys.stderr),
                               poll_s, CONFIRM_S)
    return 0 if ok else 1


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
        out.append((True, f"API reachable ({len(types)} instance types listed)"))
        keys = api.ssh_keys()
        named = [k for k in keys if k.get("name") == env.get("LAMBDA_SSH_KEY_NAME")]
        out.append((bool(named), "an SSH key with LAMBDA_SSH_KEY_NAME is registered at Lambda"))
        if named and mine:
            out.append((_fingerprint(named[0]["public_key"]) == mine,
                        "it is this laptop's key (fingerprints match)"))
        up = [i for i in api.instances() if is_ours(i) and live(i)]
        out.append((not up, f"no instance of this repo is up ({len(up)} up)"))
    except ApiError as e:
        out.append((False, f"API: {e}"))
    return out


def cmd_doctor(root: Path, env: dict[str, str], api: LambdaAPI) -> int:
    checks = doctor_checks(root, env, api)
    for ok, label in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {api.redact(label)}")
    bad = sum(not ok for ok, _ in checks)
    print("all checks pass; nothing was launched" if not bad else f"{bad} checks failed")
    return 0 if not bad else 1


def make_api(env: dict[str, str]) -> LambdaAPI:
    redact = Redactor([env.get(k, "") for k in ENV_KEYS])
    return LambdaAPI(env.get("LAMBDA_API_URL") or API_URL, env.get("LAMBDA_API_KEY", ""), redact)


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
        return 2


if __name__ == "__main__":
    sys.exit(main())
