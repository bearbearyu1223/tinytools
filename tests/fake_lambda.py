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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

H100 = "gpu_1x_h100_sxm5"


class FakeLambda:
    """Lambda's API, enough of it: instance types, instances, launch, terminate, ssh keys.

    An instance boots for ``boot_polls`` GETs, then is active; after terminate it is
    terminating for ``term_polls`` GETs, then terminated. ``fail`` queues errors per
    ``(method, path prefix)``: each entry is ``(status, body)`` and is used once.
    """

    def __init__(self, key: str, *, price_cents: int = 329, capacity: bool = True,
                 ssh_keys: list[dict] | None = None, boot_polls: int = 2,
                 term_polls: int = 1) -> None:
        self.key = key
        self.price_cents = price_cents
        self.capacity = capacity
        self.ssh_key_list = ssh_keys or []
        self.boot_polls, self.term_polls = boot_polls, term_polls
        self.instances: dict[str, dict] = {}
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail: dict[tuple[str, str], list[tuple[int, dict]]] = {}
        self.launch_creates_on_error = False
        self.lock = threading.Lock()
        self._n = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle(self, method: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                path = self.path.removeprefix("/api/v1")
                status, payload = fake.handle(method, path, body,
                                              self.headers.get("Authorization", ""))
                self._send(status, payload)

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

    def handle(self, method: str, path: str, body, auth: str) -> tuple[int, dict]:
        with self.lock:
            self.calls.append((method, path, body))
            if auth != f"Bearer {self.key}":
                return 401, {"error": {"code": "global/invalid-api-key", "message": "bad key"}}
            for (m, prefix), queue in self.fail.items():
                if m == method and path.startswith(prefix) and queue:
                    status, payload = queue.pop(0)
                    if path == "/instance-operations/launch" and self.launch_creates_on_error:
                        self._create(body)
                    if payload == "echo-auth":  # a server that echoes the request's headers
                        payload = {"error": {"code": "x", "message": f"bad request, auth={auth}"}}
                    return status, payload
            if method == "GET" and path == "/instance-types":
                regions = [{"name": "us-east-1", "description": "Virginia"}] if self.capacity else []
                return 200, {"data": {
                    H100: {"instance_type": self._type(), "regions_with_capacity_available": regions},
                    "gpu_1x_a10": {"instance_type": {"name": "gpu_1x_a10", "description": "A10",
                                                     "price_cents_per_hour": 75},
                                   "regions_with_capacity_available": regions}}}
            if method == "GET" and path == "/ssh-keys":
                return 200, {"data": self.ssh_key_list}
            if method == "GET" and path == "/instances":
                return 200, {"data": [self._view(i) for i in self.instances.values()]}
            if method == "GET" and path.startswith("/instances/"):
                inst = self.instances.get(path.rsplit("/", 1)[1])
                if inst is None:
                    return 404, {"error": {"code": "global/object-does-not-exist",
                                           "message": "no such instance"}}
                self._advance(inst)
                return 200, {"data": self._view(inst)}
            if method == "POST" and path == "/instance-operations/launch":
                return 200, {"data": {"instance_ids": [self._create(body)]}}
            if method == "POST" and path == "/instance-operations/terminate":
                for i in body["instance_ids"]:
                    if i in self.instances and self.instances[i]["status"] != "terminated":
                        self.instances[i]["status"] = "terminating"
                        self.instances[i]["polls"] = 0
                return 200, {"data": {"terminated_instances": body["instance_ids"]}}
            return 404, {"error": {"code": "not-found", "message": path}}

    def _create(self, body: dict) -> str:
        self._n += 1
        iid = f"i-{self._n:04d}"
        self.instances[iid] = {"id": iid, "name": body["name"], "status": "booting", "polls": 0,
                               "region": body["region_name"], "type": body["instance_type_name"],
                               "jupyter_token": "jt-should-never-print"}
        return iid

    def _advance(self, inst: dict) -> None:
        inst["polls"] += 1
        if inst["status"] == "booting" and inst["polls"] >= self.boot_polls:
            inst["status"] = "active"
        elif inst["status"] == "terminating" and inst["polls"] >= self.term_polls:
            inst["status"] = "terminated"

    def _view(self, inst: dict) -> dict:
        return {"id": inst["id"], "name": inst["name"], "status": inst["status"],
                "ip": "127.0.0.1" if inst["status"] == "active" else None,
                "region": {"name": inst["region"]}, "instance_type": self._type(),
                "jupyter_token": inst["jupyter_token"]}

    # -- what the tests read ----------------------------------------------------------

    def count(self, method: str, path: str) -> int:
        return sum(1 for m, p, _ in self.calls if m == method and p == path)

    def statuses(self) -> list[str]:
        return [i["status"] for i in self.instances.values()]


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
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr

    def start(self, seconds: int) -> None:
        log = open(self.dir / ".run" / "log.txt", "wb")
        self.proc = subprocess.Popen(["bash", ".run/job.sh", str(seconds)], cwd=self.dir,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()

    def poll(self, offset: int) -> tuple[bytes, int | None]:
        exit_file = self.dir / ".run" / "exit"
        code = int(exit_file.read_text()) if exit_file.exists() else None
        with open(self.dir / ".run" / "log.txt", "rb") as f:
            f.seek(offset)
            return f.read(), code

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)

    def fetch(self, patterns, dest: Path, timeout: float) -> list[str]:
        files = sorted({str(p.relative_to(self.dir)) for pat in patterns for p in self.dir.glob(pat)
                        if p.is_file()})
        for f in files:
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.dir / f, dest / f)
        return files


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


def fake_ssh(bin_dir: Path, home: Path) -> Path:
    """An ``ssh`` that runs the remote command locally, in ``home`` (the instance's home),
    so :class:`tinytools.lambda_run.SSHRemote`'s own command strings are exercised; plus a
    ``setsid`` (macOS has none) that starts its command in a new session."""
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
    for f in (ssh, setsid):
        f.chmod(0o755)
    return bin_dir
