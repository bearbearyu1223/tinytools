"""Test doubles for the Lambda runner: a local HTTP fake of Lambda's API, and a "remote"
that is a local directory (the instance's side runs as local shell commands).

Not a test file itself (no ``test_`` prefix); ``test_lambda_run.py`` and
``lambda_driver.py`` import it.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

H100 = "gpu_1x_h100_sxm5"
NAME_MAX = 64


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


class FakeLambda:
    """Lambda's API, enough of it: instance types, instances, launch, terminate, ssh keys.

    Always, as the real API does: terminate answers with the Instance objects, status
    ``terminating``; launch checks the SSH key name (one name; one of ``ssh_keys`` when
    any are registered), the 64-character name limit and the region (one with capacity);
    every Instance carries ``actions.terminate``; error bodies carry a ``request_id``.

    An instance boots for ``boot_polls`` GETs of it, then is active; after terminate it is
    terminating for ``term_polls`` GETs, then terminated. ``fail`` queues errors per
    ``(method, path prefix)``: each entry is ``(status, body)`` and is used once.

    Modes (off by default):
      ``blocked_while_booting``  actions.terminate.available is false (reason
                                 ``vm-has-not-launched``) while booting, and terminate
                                 is refused with a 400 then
      ``rate_limit=(req_s, launch_s)``  429 ``global/rate-limited`` for a request less
                                 than ``req_s`` after the previous one, or a launch less
                                 than ``launch_s`` after the previous launch (Lambda: 1 s
                                 and 12 s)
      ``list_lag=N``             a new instance is missing from the first N reads of it
                                 (``/instances`` omits it; ``/instances/{id}`` is a 404)
      ``fate=(N, status)``       N GETs after going active, the instance turns ``status``
                                 (``unhealthy`` or ``preempted``)
      ``redirect_to=url``        every request answers 301 to ``url`` + path
      ``truncate[(method, path)] = n``  the next n such requests take effect, but the
                                 answer is cut off mid-body (http.client.IncompleteRead)
    """

    def __init__(self, key: str, *, price_cents: int = 329, capacity: bool = True,
                 ssh_keys: list[dict] | None = None, boot_polls: int = 2,
                 term_polls: int = 1, blocked_while_booting: bool = False,
                 rate_limit: tuple[float, float] | None = None, list_lag: int = 0,
                 fate: tuple[int, str] | None = None, redirect_to: str | None = None) -> None:
        self.key = key
        self.price_cents = price_cents
        self.capacity = capacity
        self.ssh_key_list = ssh_keys or []
        self.boot_polls, self.term_polls = boot_polls, term_polls
        self.blocked_while_booting = blocked_while_booting
        self.rate_limit = rate_limit
        self.list_lag = list_lag
        self.fate = fate
        self.redirect_to = redirect_to
        self.instances: dict[str, dict] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail: dict[tuple[str, str], list[tuple[int, dict]]] = {}
        self.truncate: dict[tuple[str, str], int] = {}
        self.limited = 0  # 429s sent
        self.launch_creates_on_error = False
        self.lock = threading.Lock()
        self._n = 0
        self._rid = 0
        self._last_request: float | None = None
        self._last_launch: float | None = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, status: int, payload: dict, cut: bool = False) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body) + (100 if cut else 0)))
                self.end_headers()
                self.wfile.write(body[:len(body) // 2] if cut else body)
                if cut:
                    self.close_connection = True

            def _handle(self, method: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                if fake.redirect_to:
                    with fake.lock:
                        fake.calls.append((method, self.path, body))
                    self.send_response(301)
                    self.send_header("Location", fake.redirect_to + self.path.removeprefix("/api/v1"))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                path = self.path.removeprefix("/api/v1")
                status, payload = fake.handle(method, path, body,
                                              self.headers.get("Authorization", ""))
                if status >= 400 and isinstance(payload, dict) and "request_id" not in payload:
                    with fake.lock:
                        fake._rid += 1
                        payload = {**payload, "request_id": f"req-{fake._rid:06d}"}
                self._send(status, payload, cut=fake._take_truncate(method, path))

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    # -- behaviour --------------------------------------------------------------------

    def _type(self) -> dict:
        return {"name": H100, "description": "1x H100 (80 GB SXM5)",
                "price_cents_per_hour": self.price_cents}

    def _regions(self) -> list[dict]:
        return [{"name": "us-east-1", "description": "Virginia"}] if self.capacity else []

    def _take_truncate(self, method: str, path: str) -> bool:
        with self.lock:
            n = self.truncate.get((method, path), 0)
            if n:
                self.truncate[(method, path)] = n - 1
            return bool(n)

    def _limited(self, method: str, path: str) -> bool:
        """The rate limit (only in that mode); records this request's time."""
        if not self.rate_limit:
            return False
        req_s, launch_s = self.rate_limit
        now = time.monotonic()
        last, self._last_request = self._last_request, now
        if last is not None and now - last < req_s:
            return True
        if method == "POST" and path == "/instance-operations/launch":
            if self._last_launch is not None and now - self._last_launch < launch_s:
                return True
        return False

    def handle(self, method: str, path: str, body, auth: str) -> tuple[int, dict]:
        with self.lock:
            if self._limited(method, path):  # not in `calls`: it never reached the API
                self.limited += 1
                return 429, _err("global/rate-limited", "too many requests")
            self.calls.append((method, path, body))
            if auth != f"Bearer {self.key}":
                return 401, _err("global/invalid-api-key", "bad key")
            for (m, prefix), queue in self.fail.items():
                if m == method and path.startswith(prefix) and queue:
                    status, payload = queue.pop(0)
                    if path == "/instance-operations/launch" and self.launch_creates_on_error:
                        self._create(body, hidden=self.list_lag)
                        self._last_launch = time.monotonic()
                    if payload == "echo-auth":  # a server that echoes the request's headers
                        payload = _err("x", f"bad request, auth={auth}")
                    return status, payload
            if method == "GET" and path == "/instance-types":
                return 200, {"data": {
                    H100: {"instance_type": self._type(),
                           "regions_with_capacity_available": self._regions()},
                    "gpu_1x_a10": {"instance_type": {"name": "gpu_1x_a10", "description": "A10",
                                                     "price_cents_per_hour": 75},
                                   "regions_with_capacity_available": self._regions()}}}
            if method == "GET" and path == "/ssh-keys":
                return 200, {"data": self.ssh_key_list}
            if method == "GET" and path == "/instances":
                shown = []
                for i in self.instances.values():
                    if i["hidden"] > 0:
                        i["hidden"] -= 1
                        continue
                    shown.append(self._view(i))
                return 200, {"data": shown}
            if method == "GET" and path.startswith("/instances/"):
                inst = self.instances.get(path.rsplit("/", 1)[1])
                if inst is not None and inst["hidden"] > 0:
                    inst["hidden"] -= 1
                    inst = None
                if inst is None:
                    return 404, _err("global/object-does-not-exist", "no such instance")
                self._advance(inst)
                return 200, {"data": self._view(inst)}
            if method == "POST" and path == "/instance-operations/launch":
                problem = self._launch_problem(body)
                if problem:
                    return 400, problem
                self._last_launch = time.monotonic()
                return 200, {"data": {"instance_ids": [self._create(body, hidden=self.list_lag)]}}
            if method == "POST" and path == "/instance-operations/terminate":
                ids = body.get("instance_ids") or []
                unknown = [i for i in ids if i not in self.instances]
                if unknown:
                    return 404, _err("global/object-does-not-exist",
                                     f"no such instance: {unknown[0]}")
                if self.blocked_while_booting and any(
                        self.instances[i]["status"] == "booting" for i in ids):
                    return 400, _err("global/invalid-parameters",
                                     "vm-has-not-launched: the instance can't be terminated "
                                     "until it has launched")
                out = []
                for i in ids:
                    inst = self.instances[i]
                    if inst["status"] not in ("terminated", "preempted", "terminating"):
                        inst["status"] = "terminating"
                        inst["polls"] = 0
                    out.append(self._view(inst))
                return 200, {"data": {"terminated_instances": out}}
            return 404, _err("not-found", path)

    def _launch_problem(self, body) -> dict | None:
        keys = body.get("ssh_key_names")
        if not (isinstance(keys, list) and len(keys) == 1 and isinstance(keys[0], str)
                and keys[0]):
            return _err("global/invalid-parameters", "ssh_key_names: exactly one key name")
        if self.ssh_key_list and keys[0] not in {k.get("name") for k in self.ssh_key_list}:
            return _err("global/invalid-parameters", "ssh_key_names: no such SSH key")
        name = body.get("name") or ""
        if len(name) > NAME_MAX:
            return _err("global/invalid-parameters", f"name: at most {NAME_MAX} characters")
        if body.get("region_name") not in {r["name"] for r in self._regions()}:
            return _err("instance-operations/launch/insufficient-capacity",
                        "Not enough capacity to fulfill launch request.")
        tags = body.get("tags", [])
        if not (isinstance(tags, list) and all(isinstance(t, dict) and set(t) == {"key", "value"}
                                               for t in tags)):
            return _err("global/invalid-parameters", "tags: a list of {key, value}")
        return None

    def _create(self, body: dict, hidden: int = 0) -> str:
        """An instance, as a launch makes it (``hidden``: reads it is missing from)."""
        self._n += 1
        iid = f"i-{self._n:04d}"
        self.instances[iid] = {"id": iid, "name": body["name"], "status": "booting", "polls": 0,
                               "region": body["region_name"], "type": body["instance_type_name"],
                               "tags": body.get("tags", []), "hidden": hidden,
                               "jupyter_token": "jt-should-never-print"}
        return iid

    def _advance(self, inst: dict) -> None:
        inst["polls"] += 1
        if inst["status"] == "booting" and inst["polls"] >= self.boot_polls:
            inst["status"] = "active"
            inst["polls"] = 0
        elif inst["status"] == "active" and self.fate and inst["polls"] >= self.fate[0]:
            inst["status"] = self.fate[1]
        elif inst["status"] == "terminating" and inst["polls"] >= self.term_polls:
            inst["status"] = "terminated"

    def _view(self, inst: dict) -> dict:
        booting = inst["status"] == "booting"
        blocked = self.blocked_while_booting and booting
        terminate = ({"available": False, "reason_code": "vm-has-not-launched",
                      "reason_description": "The instance has not launched yet."}
                     if blocked else {"available": True})
        return {"id": inst["id"], "name": inst["name"], "status": inst["status"],
                "ip": "127.0.0.1" if inst["status"] in ("active", "unhealthy") else None,
                "region": {"name": inst["region"]}, "instance_type": self._type(),
                "tags": inst["tags"], "actions": {"terminate": terminate},
                "jupyter_token": inst["jupyter_token"]}

    # -- what the tests read ----------------------------------------------------------

    def count(self, method: str, path: str) -> int:
        return sum(1 for m, p, _ in self.calls if m == method and p == path)

    def statuses(self) -> list[str]:
        return [i["status"] for i in self.instances.values()]


def session_pids(sid: int) -> list[int]:
    """Live (non-zombie) processes in session ``sid``, as ``pgrep -s`` (macOS has none)."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,stat="], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1].startswith("Z"):
            continue
        try:
            if os.getsid(int(parts[0])) == sid:
                pids.append(int(parts[0]))
        except OSError:
            pass
    return pids


def stop_session(sid: int, wait_s: float = 5.0) -> None:
    """What ``SSHRemote.stop`` does on the instance: TERM the session, wait, KILL."""
    for p in session_pids(sid):
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    t0 = time.time()
    while session_pids(sid) and time.time() - t0 < wait_s:
        time.sleep(0.05)
    for p in session_pids(sid):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


class LocalRemote:
    """The instance's side, run locally in ``root/tinytools`` (same interface as SSHRemote)."""

    def __init__(self, root: Path) -> None:
        self.dir = Path(root) / "tinytools"
        self.proc: subprocess.Popen | None = None

    def ready(self) -> bool:
        return True

    def push(self, src: Path, timeout: float) -> None:
        shutil.copytree(src, self.dir, dirs_exist_ok=True)

    def run(self, script: str, timeout: float) -> tuple[int, str]:
        r = subprocess.run(["bash", "-c", script], cwd=self.dir, capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout + r.stderr

    def start(self, seconds: int) -> None:
        log = open(self.dir / ".run" / "log.txt", "wb")
        self.proc = subprocess.Popen(["bash", ".run/job.sh", str(seconds)], cwd=self.dir,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()

    def poll(self, offset: int):
        exit_file = self.dir / ".run" / "exit"
        if exit_file.exists():
            code = int(exit_file.read_text())
        elif self.proc is not None and self.proc.poll() is None:
            code = None
        else:
            code = int(exit_file.read_text()) if exit_file.exists() else "died"
        with open(self.dir / ".run" / "log.txt", "rb") as f:
            f.seek(offset)
            return f.read(), code

    def stop(self, wait_s: float = 5.0) -> None:
        if self.proc is not None:
            self.proc.poll()  # reap job.sh if it has ended, so it isn't a zombie in the list
            stop_session(self.proc.pid, min(wait_s, 5.0))
            self.proc.poll()

    def fetch(self, patterns, dest: Path, timeout: float, mark: int = 1, since=None,
              have=()) -> tuple[list[str], list[str]]:
        """As SSHRemote.fetch: touch .run/fetch-<mark>; skip a file in ``have`` unless its
        status changed after .run/fetch-<since>."""
        start = self.dir / ".run" / "start"
        if not start.exists():
            return [], []
        (self.dir / ".run" / f"fetch-{mark}").touch()
        prev = self.dir / ".run" / f"fetch-{since}" if since is not None else None
        t0 = start.stat().st_mtime_ns
        files = sorted({str(p.relative_to(self.dir)) for pat in patterns for p in self.dir.glob(pat)
                        if p.is_file() and p.stat().st_mtime_ns > t0})
        self.copied = []
        for f in files:
            if (f in have and prev is not None and prev.exists()
                    and (self.dir / f).stat().st_ctime_ns <= prev.stat().st_mtime_ns):
                continue  # an earlier fetch has it, unchanged since
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.dir / f, dest / f)
            self.copied.append(f)
        return files, list(self.copied)


def timeout_shim(bin_dir: Path, enforce: bool = True) -> Path:
    """A ``timeout`` for the Mac (GNU's isn't installed): ``timeout [opts] SECS cmd...``.
    ``enforce=False`` ignores the limit, standing in for a remote timeout that failed."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "timeout"
    shim.write_text(f"""#!/usr/bin/env python3
import subprocess, sys
args = [a for a in sys.argv[1:] if not a.startswith("--")]
secs, cmd = float(args[0]), args[1:]
p = subprocess.Popen(cmd)
try:
    sys.exit(p.wait(timeout=secs if {enforce!r} else None))
except subprocess.TimeoutExpired:
    p.terminate()
    p.wait()
    sys.exit(124)
""")
    shim.chmod(0o755)
    return bin_dir


_PROC_SHIM = """#!/usr/bin/env python3
# `{name} [-SIGNAL] -s SID` for macOS, whose pgrep/pkill have no -s; anything else goes
# to the system's {name}.
import os, signal, subprocess, sys
args = sys.argv[1:]
sig = signal.SIGTERM
if args and args[0].startswith("-") and args[0] != "-s" and args[0][1:].isalpha():
    sig = getattr(signal, "SIG" + args[0][1:].upper())
    args = args[1:]
if args[:1] != ["-s"]:
    os.execv("/usr/bin/{name}", ["/usr/bin/{name}", *sys.argv[1:]])
sid = int(args[1])
out = subprocess.run(["ps", "-A", "-o", "pid=,stat="], capture_output=True, text=True).stdout
pids = []
for line in out.splitlines():
    parts = line.split()
    if len(parts) < 2 or parts[1].startswith("Z") or int(parts[0]) == os.getpid():
        continue
    try:
        if os.getsid(int(parts[0])) == sid:
            pids.append(int(parts[0]))
    except OSError:
        pass
for p in pids:
    if "{name}" == "pgrep":
        print(p)
    else:
        try:
            os.kill(p, sig)
        except OSError:
            pass
sys.exit(0 if pids else 1)
"""


def fake_ssh(bin_dir: Path, home: Path) -> Path:
    """An ``ssh`` that runs the remote command locally, in ``home`` (the instance's home),
    so :class:`tinytools.lambda_run.SSHRemote`'s own command strings are exercised; plus a
    ``setsid`` (macOS has none) that starts its command in a new session, and ``pgrep`` /
    ``pkill`` that understand ``-s SID`` as Ubuntu's do."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    ssh = bin_dir / "ssh"
    ssh.write_text(f"""#!/usr/bin/env python3
import os, subprocess, sys
args, i = sys.argv[1:], 0
while i < len(args) and args[i].startswith("-"):
    i += 2 if args[i] in ("-i", "-o", "-p", "-l") else 1
cmd = " ".join(args[i + 1:])  # args[i] is user@host
os.chdir({str(home)!r})
os.environ["HOME"] = {str(home)!r}
sys.exit(subprocess.run(["bash", "-c", cmd]).returncode)
""")
    setsid = bin_dir / "setsid"
    setsid.write_text("""#!/usr/bin/env python3
import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
""")
    made = [ssh, setsid]
    for name in ("pgrep", "pkill"):
        p = bin_dir / name
        p.write_text(_PROC_SHIM.replace("{name}", name))
        made.append(p)
    for f in made:
        f.chmod(0o755)
    return bin_dir
