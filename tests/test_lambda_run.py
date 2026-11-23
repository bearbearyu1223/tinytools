"""The Lambda runner: its pure pieces, and the whole flow against a local fake of the API.

Nothing here talks to Lambda. The "instance" is a local directory (``LocalRemote``);
the API is ``FakeLambda`` on 127.0.0.1. Every flow test checks that the instance ends
terminated and that no secret from the settings appears in any output or file.
"""

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fake_lambda import H100, FakeLambda, LocalRemote, fake_ssh, timeout_shim

from tinytools import lambda_run as lr

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-test-SECRET-0123456789abcdef"
KEY_NAME = "laptop-key-NAME-7f3a"
KEY_PATH = "/nonexistent/SECRETPATH/id_ed25519"
SECRETS = (KEY, KEY_NAME, KEY_PATH)
SETUP = "echo 'TINYTOOLS_SETUP_JSON {\"gpu\": \"fake GPU\", \"torch\": \"0\"}'"


# -- the experiment file ------------------------------------------------------------------

GOOD = """
name = "demo"
post = 9
instance_type = "gpu_1x_h100_sxm5"
max_hours = 2.5
expected_hours = 2
cap_usd = 15
commands = ["uv run tinytools milestone0"]
fetch = ["outputs/milestone0.txt", "outputs/milestone0-*.jsonl"]
"""


def test_parse_experiment_reads_a_good_file():
    e = lr.parse_experiment(GOOD, "demo")
    assert (e.instance_type, e.max_hours, e.expected_hours, e.cap_usd, e.post) == (
        "gpu_1x_h100_sxm5", 2.5, 2.0, 15.0, 9)
    assert e.commands == ("uv run tinytools milestone0",)
    assert e.setup_record is None


def test_the_milestone0_experiment_parses():
    e = lr.parse_experiment((ROOT / "experiments" / "milestone0.toml").read_text(), "milestone0")
    assert e.instance_type == "gpu_1x_a100_sxm4"  # the author, 2026-11-21: an A100
    assert e.expected_hours <= e.max_hours
    assert e.setup_record == "outputs/milestone0-setup.json"
    assert e.commands[-1] == ("uv run tinytools milestone0 --base-when-needed "
                              '--deadline-epoch "$TINYTOOLS_DEADLINE_EPOCH"')
    assert lr.DEADLINE_ENV == "TINYTOOLS_DEADLINE_EPOCH" and lr.DEADLINE_ENV in lr.JOB_SCRIPT
    assert (e.post, e.max_hours, e.cap_usd) == (9, 4.0, 10.0)
    for cents, ok in ((199, True), (250, True), (251, False)):  # worst $7.96 at $1.99/h
        types = {e.instance_type: {"instance_type": {"price_cents_per_hour": cents},
                                   "regions_with_capacity_available": [{"name": "us-east-1"}]}}
        assert lr.make_estimate(e, types, []).within_cap is ok
    # the fetch globs match this run's outputs, not the committed Mac --quick captures
    committed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "outputs"],
                               capture_output=True, text=True).stdout.split()
    quick = [f for f in committed if f.startswith("outputs/milestone0-quick")]
    assert quick and not [f for f in quick if any(Path(f).match(p) for p in e.fetch)]
    for f in ("outputs/milestone0-Qwen2.5-1.5B-Instruct-ours-replies.jsonl",
              "outputs/milestone0-Qwen2.5-3B-bfcl-replies.jsonl"):
        assert any(Path(f).match(p) for p in e.fetch)


@pytest.mark.parametrize("change, error", [
    (("cap_usd = 15", ""), "missing keys"),
    (("post = 9", "post = 9\ncolour = 1"), "unknown keys"),
    (('name = "demo"', 'name = "other"'), "differs from the file name"),
    (("max_hours = 2.5", "max_hours = 0"), "max_hours"),
    (("expected_hours = 2", "expected_hours = 3"), "expected_hours"),
    (("cap_usd = 15", "cap_usd = -1"), "cap_usd"),
    (("max_hours = 2.5", 'max_hours = "2.5"'), "wrong type"),
    (('commands = ["uv run tinytools milestone0"]', "commands = []"), "commands"),
    (('"outputs/milestone0.txt"', '"../secrets.txt"'), "no '..'"),
    (('"outputs/milestone0.txt"', '"/etc/passwd"'), "relative"),
    (('"outputs/milestone0.txt"', '"outputs/$(rm -rf ~)"'), "relative"),
    (("post = 9", 'post = 9\nsetup_record = "outputs/x.json"'), "not matched"),
])
def test_parse_experiment_rejects(change, error):
    with pytest.raises(ValueError, match=error.replace("(", r"\(")):
        lr.parse_experiment(GOOD.replace(*change), "demo")


# -- settings, secrets, money ------------------------------------------------------------


def test_parse_env():
    text = "# c\nexport LAMBDA_API_KEY='abc def'\nLAMBDA_SSH_KEY_NAME=\"k\"\nLAMBDA_SSH_KEY=~/x\nbad line\n"
    assert lr.parse_env(text) == {"LAMBDA_API_KEY": "abc def", "LAMBDA_SSH_KEY_NAME": "k",
                                  "LAMBDA_SSH_KEY": "~/x"}


def test_describe_env_shows_names_and_lengths_only():
    lines = lr.describe_env({"LAMBDA_API_KEY": KEY, "LAMBDA_SSH_KEY_NAME": KEY_NAME})
    assert lines[0] == f"LAMBDA_API_KEY: set ({len(KEY)} characters)"
    assert lines[2] == "LAMBDA_SSH_KEY: MISSING"
    assert not any(s in "\n".join(lines) for s in SECRETS)


def test_redactor():
    r = lr.Redactor([KEY, "", "ab"])
    assert r(f"x {KEY} y") == "x [redacted] y"
    assert r("Authorization: Bearer other-token") == "Authorization: Bearer [redacted]"
    assert r("ab stays (too short to be a secret)") == "ab stays (too short to be a secret)"


def test_estimate_and_the_cap():
    exp = lr.parse_experiment(GOOD, "demo")
    types = {H100: {"instance_type": {"price_cents_per_hour": 329},
                    "regions_with_capacity_available": [{"name": "us-east-1"}]}}
    e = lr.make_estimate(exp, types, [])
    assert (e.price, e.worst, e.expected, e.remaining, e.within_cap) == (3.29, 8.23, 6.58, 15.0, True)
    spent = [{"experiment": "demo", "cost_usd": "7.0"}, {"experiment": "other", "cost_usd": "9"}]
    e = lr.make_estimate(exp, types, spent)
    assert (e.spent, e.remaining, e.within_cap) == (7.0, 8.0, False)
    assert any("REFUSED" in line for line in e.lines())
    with pytest.raises(lr.Refused, match="not in Lambda's list"):
        lr.make_estimate(lr.parse_experiment(GOOD.replace("h100_sxm5", "h200"), "demo"), types, [])


def test_ledger_upsert_and_spent(tmp_path):
    p = tmp_path / "runs" / "ledger.csv"
    lr.write_ledger_row(p, {"run": "a", "experiment": "demo", "cost_usd": "8.2250", "status": "running"})
    lr.write_ledger_row(p, {"run": "b", "experiment": "demo", "cost_usd": "1.0", "status": "ok"})
    lr.write_ledger_row(p, {"run": "a", "experiment": "demo", "cost_usd": "2.5", "status": "ok"})
    rows = lr.read_ledger(p)
    assert [r["run"] for r in rows] == ["b", "a"]
    assert lr.spent(rows, "demo") == 3.5
    assert p.read_text().splitlines()[0] == ",".join(lr.LEDGER_FIELDS)


def test_cost_run_dir_and_remote_seconds():
    assert lr.cost_usd(3.29, 5400) == 4.935
    assert lr.run_dir_name("2026-11-21", "m0", set()) == "2026-11-21-m0"
    assert lr.run_dir_name("2026-11-21", "m0", {"2026-11-21-m0", "2026-11-21-m0-2"}) == "2026-11-21-m0-3"
    assert lr.remote_seconds(1000.0, 100.0, 30) == 870
    assert lr.remote_seconds(100.0, 100.0, 30) == 1


def test_commands_script_stops_at_the_first_failure(tmp_path):
    script = lr.commands_script(("echo one", "echo 'it''s'; false", "echo never"))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 1
    assert r.stdout.splitlines() == ["+ echo one", "one", "+ echo 'it''s'; false", "its"]


def test_parse_setup():
    assert lr.parse_setup("noise\nTINYTOOLS_SETUP_JSON {\"gpu\": \"H100\"}\n") == {"gpu": "H100"}
    assert lr.parse_setup("nothing") == {}


# -- the whole flow, against the fake API --------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t",
                    *args], check=True, capture_output=True)


def write_exp(repo: Path, commands: list[str], *, max_hours: float = 1.0,
              expected: float | None = None, cap: float = 15.0) -> None:
    fetch = ["outputs/demo.txt", "outputs/demo-*.jsonl", "outputs/demo-setup.json"]
    (repo / "experiments").mkdir(exist_ok=True)
    (repo / "experiments" / "demo.toml").write_text(
        f'name = "demo"\npost = 9\ninstance_type = "{H100}"\nmax_hours = {max_hours}\n'
        f"expected_hours = {expected or max_hours / 2}\ncap_usd = {cap}\n"
        f"commands = {json.dumps(commands)}\nfetch = {json.dumps(fetch)}\n"
        'setup_record = "outputs/demo-setup.json"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "experiment")


# Commands that write a file for each fetch pattern (a pattern that matches nothing
# written makes the run exit 1)
WRITES = ["mkdir -p outputs", "echo hi > outputs/demo.txt", "echo '{}' > outputs/demo-a.jsonl",
          "echo '{}' > outputs/demo-setup.json"]


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "README.md").write_text("demo repo\n")
    write_exp(r, WRITES)
    return r


# The real API's behaviour, scaled in time: terminate refused while booting, a new
# instance missing from the first read of it, and 429s for requests 0.05 s apart.
REALISTIC = dict(blocked_while_booting=True, list_lag=1, rate_limit=(0.05, 0.6))
RATE_WAIT = 0.05  # the client's 429 backoff, scaled the same way


@pytest.fixture(params=["plain", "realistic"])
def fake(request):
    kw = REALISTIC if request.param == "realistic" else {}
    with FakeLambda(KEY, boot_polls=2, term_polls=2, **kw) as f:
        yield f


@pytest.fixture(autouse=True)
def shim(tmp_path, monkeypatch):
    bin_dir = timeout_shim(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def env_for(fake: FakeLambda) -> dict:
    return {"LAMBDA_API_KEY": KEY, "LAMBDA_SSH_KEY_NAME": KEY_NAME, "LAMBDA_SSH_KEY": KEY_PATH,
            "LAMBDA_API_URL": fake.url}


def make_runner(repo, fake, tmp_path, *, remote_cls=LocalRemote, approve=100.0,
                setup_script=SETUP, reserve_s=1.0, margin_s=0.5):
    env = env_for(fake)
    exp, path = lr.load_experiment(repo, "demo")
    out = io.StringIO()
    runner = lr.Runner(repo, exp, path, lr.make_api(env, rate_wait=RATE_WAIT), env,
                       remote_factory=lambda ip: remote_cls(tmp_path / "remote"),
                       approve=approve, setup_script=setup_script, poll_s=0.05,
                       boot_poll_s=0.02, reserve_s=reserve_s, margin_s=margin_s, confirm_s=5,
                       keep_awake=False, stream=out)
    return runner, out


def everything_written(repo: Path, *texts: str) -> str:
    files = [p for p in (repo / "runs").rglob("*") if p.is_file()] if (repo / "runs").exists() else []
    return "\n".join([*texts, *(p.read_text(errors="replace") for p in files)])


def assert_no_secrets(text: str) -> None:
    for s in SECRETS:
        assert s not in text
    assert "jt-should-never-print" not in text


def run_json(repo: Path) -> dict:
    (path,) = list((repo / "runs").glob("*/run.json"))
    return json.loads(path.read_text())


def test_a_good_run_fetches_records_and_terminates(repo, fake, tmp_path):
    write_exp(repo, ["mkdir -p outputs", "echo result > outputs/demo.txt",
                     "echo '{\"a\": 1}' > outputs/demo-replies.jsonl",
                     "echo '{\"seeds\": [0], \"determinism\": \"on\", \"dtype\": \"bfloat16\"}' "
                     "> outputs/demo-setup.json"])
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 0
    assert fake.statuses() == ["terminated"]
    rj = run_json(repo)
    assert rj["status"] == "ok" and rj["terminated"] is True and rj["remote_exit_code"] == 0
    assert rj["commit"] == subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()
    assert rj["instance"]["price_usd_per_hour"] == 3.29
    assert rj["versions"] == {"gpu": "fake GPU", "torch": "0"}
    assert (rj["seeds"], rj["determinism"], rj["dtype"]) == ([0], "on", "bfloat16")
    run_dir = next((repo / "runs").glob("*-demo"))
    fetched = {f["path"]: f["sha256"] for f in rj["fetched"]}
    assert set(fetched) == {"outputs/demo.txt", "outputs/demo-replies.jsonl",
                            "outputs/demo-setup.json"}
    assert fetched["outputs/demo.txt"] == lr.sha256_file(run_dir / "outputs" / "demo.txt")
    assert (run_dir / "outputs" / "demo.txt").read_text() == "result\n"
    (row,) = lr.read_ledger(repo / "runs" / "ledger.csv")
    assert row["status"] == "ok" and 0 < float(row["cost_usd"]) < 0.01
    log = (run_dir / "log.txt").read_text()
    assert "+ echo result > outputs/demo.txt" in log and "billing has stopped" in log
    assert "no instance of this repo is up" in out.getvalue()
    assert_no_secrets(everything_written(repo, out.getvalue()))


def test_a_failing_command_still_terminates(repo, fake, tmp_path):
    write_exp(repo, ["echo start", "exit 3", "echo never"])
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 1
    assert fake.statuses() == ["terminated"]
    rj = run_json(repo)
    assert rj["status"] == "failed: a command exited 3" and rj["terminated"] is True
    log = next((repo / "runs").glob("*/log.txt")).read_text()
    assert "start" in log and "never\n" not in log.replace("echo never", "")
    assert_no_secrets(everything_written(repo, out.getvalue()))


def test_a_failing_setup_still_terminates(repo, fake, tmp_path):
    runner, out = make_runner(repo, fake, tmp_path, setup_script="echo broken; exit 5")
    assert runner.run() == 1
    assert fake.statuses() == ["terminated"]
    assert run_json(repo)["status"] == "failed: RuntimeError: setup failed (exit 5)"


class ExplodingRemote(LocalRemote):
    def push(self, src, timeout):
        raise OSError("disk full on the instance")


def test_an_exception_in_the_runner_still_terminates(repo, fake, tmp_path):
    runner, out = make_runner(repo, fake, tmp_path, remote_cls=ExplodingRemote)
    assert runner.run() == 1
    assert fake.statuses() == ["terminated"]
    assert run_json(repo)["status"] == "failed: OSError: disk full on the instance"


def test_the_laptop_deadline_terminates_when_the_remote_timeout_does_not(repo, fake, tmp_path,
                                                                          shim):
    timeout_shim(shim, enforce=False)  # the remote's `timeout` fails to stop the work
    write_exp(repo, ["sleep 60"], max_hours=0.002)  # 7.2 s of instance life
    runner, out = make_runner(repo, fake, tmp_path, reserve_s=3.0, margin_s=1.0)
    t0 = time.time()
    assert runner.run() == 1
    assert time.time() - t0 < 7.2 + 5
    assert fake.statuses() == ["terminated"]
    rj = run_json(repo)
    assert rj["status"] == "time limit" and rj["terminated"] is True
    assert rj["hours"] <= 0.002 + 5 / 3600


def test_the_remote_timeout_stops_the_work_and_the_instance_is_terminated(repo, fake, tmp_path):
    write_exp(repo, ["sleep 60"], max_hours=0.002)
    runner, out = make_runner(repo, fake, tmp_path, reserve_s=3.0, margin_s=2.0)
    assert runner.run() == 1
    assert fake.statuses() == ["terminated"]
    rj = run_json(repo)
    assert (rj["status"], rj["remote_exit_code"]) == ("time limit", 124)


def test_estimate_refuses_past_the_cap_and_nothing_launches(repo, fake, tmp_path, capsys):
    lr.write_ledger_row(repo / "runs" / "ledger.csv",
                        {"run": "earlier", "experiment": "demo", "cost_usd": "14.0000"})
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 2
    assert "refused: the worst case $3.29 is more than the $1.00 left" in out.getvalue()
    assert fake.count("POST", "/instance-operations/launch") == 0
    api = lr.make_api(env_for(fake), rate_wait=RATE_WAIT)
    assert lr.cmd_estimate(api, repo, "demo") == 2
    assert "REFUSED" in capsys.readouterr().out


def test_refuses_below_the_approved_amount_a_dirty_tree_and_a_running_instance(repo, fake,
                                                                                tmp_path):
    runner, out = make_runner(repo, fake, tmp_path, approve=3.0)
    assert runner.run() == 2 and "more than the approved $3.00" in out.getvalue()

    (repo / "README.md").write_text("changed\n")
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 2 and "uncommitted changes" in out.getvalue()
    _git(repo, "checkout", "README.md")

    fake._create({"name": "tinytools-demo-earlier", "region_name": "us-east-1",
                  "instance_type_name": H100})
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 2 and "already up" in out.getvalue()
    assert fake.count("POST", "/instance-operations/launch") == 0


def test_refuses_without_capacity_or_approval(repo, tmp_path):
    with FakeLambda(KEY, capacity=False) as f:
        runner, out = make_runner(repo, f, tmp_path)
        assert runner.run() == 2 and "no capacity" in out.getvalue()
    with FakeLambda(KEY) as f:
        runner, out = make_runner(repo, f, tmp_path, approve=None)  # no terminal to ask
        assert runner.run() == 2 and "no approval" in out.getvalue()
        assert f.count("POST", "/instance-operations/launch") == 0


def test_a_launch_whose_answer_is_lost_is_found_by_name_and_terminated(repo, fake, tmp_path):
    fake.fail[("POST", "/instance-operations/launch")] = [(500, {"error": {"code": "x",
                                                                           "message": "oops"}})]
    fake.launch_creates_on_error = True
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 1
    assert fake.statuses() == ["terminated"]
    rj = run_json(repo)
    assert rj["terminated"] is True and rj["instance"]["id"] == "i-0001"


def test_a_launch_that_starts_nothing_costs_nothing(repo, fake, tmp_path):
    fake.fail[("POST", "/instance-operations/launch")] = [
        (400, {"error": {"code": "instance-operations/launch/insufficient-capacity",
                         "message": "no capacity"}})]
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 1
    assert fake.instances == {} and fake.count("POST", "/instance-operations/terminate") == 0
    rj = run_json(repo)
    assert rj["terminated"] is True and rj["cost_usd"] == 0.0
    assert "insufficient-capacity" in rj["status"]


def test_terminate_is_retried_and_an_echoed_key_is_redacted(repo, fake, tmp_path):
    fake.fail[("POST", "/instance-operations/terminate")] = [(500, "echo-auth"), (503, "echo-auth")]
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 0
    assert fake.count("POST", "/instance-operations/terminate") == 3
    assert fake.statuses() == ["terminated"]
    text = everything_written(repo, out.getvalue())
    assert "auth=Bearer [redacted]" in text
    assert_no_secrets(text)


def test_unconfirmed_termination_is_charged_at_the_worst_case(repo, tmp_path):
    with FakeLambda(KEY, term_polls=10**9) as f:  # never reports terminated
        runner, out = make_runner(repo, f, tmp_path)
        runner.confirm_s = 0.3
        assert runner.run() == lr.EXIT_UNCONFIRMED  # 3: never 0 while it may be billing
        rj = run_json(repo)
        assert rj["terminated"] is False and rj["cost_usd"] == lr.cost_usd(3.29, 3600)
        assert "STILL UP" in out.getvalue()


def _driver(repo, fake, tmp_path, shim, commands):
    write_exp(repo, commands)
    settings = {"env": env_for(fake), "repo": str(repo), "experiment": "demo",
                "remote": str(tmp_path / "remote"), "setup_script": SETUP, "reserve_s": 1.0,
                "margin_s": 0.5, "rate_wait": RATE_WAIT}
    err = open(tmp_path / "driver.err", "w")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("lambda_driver.py")),
                             json.dumps(settings)], stdout=err, stderr=err,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)})
    remote_log = tmp_path / "remote" / "tinytools" / ".run" / "log.txt"
    t0 = time.time()
    while not (remote_log.exists() and "+ sleep" in remote_log.read_text()):
        assert proc.poll() is None, (tmp_path / "driver.err").read_text()
        assert time.time() - t0 < 30
        time.sleep(0.05)
    return proc


@pytest.mark.parametrize("sig, status", [(signal.SIGINT, "interrupted (Ctrl-C)"),
                                         (signal.SIGTERM, "interrupted (SIGTERM)")])
def test_ctrl_c_and_sigterm_terminate(repo, fake, tmp_path, shim, sig, status):
    proc = _driver(repo, fake, tmp_path, shim, ["sleep 60"])
    proc.send_signal(sig)
    assert proc.wait(timeout=30) == 1
    assert fake.statuses() == ["terminated"]
    rj = run_json(repo)
    assert rj["status"] == status and rj["terminated"] is True
    pid = int((tmp_path / "remote" / "tinytools" / ".run" / "pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):  # the remote job's process group is gone too
        os.killpg(pid, 0)
    assert_no_secrets(everything_written(repo, (tmp_path / "driver.err").read_text()))


def test_a_second_ctrl_c_does_not_stop_the_termination(repo, tmp_path, shim):
    with FakeLambda(KEY, term_polls=15) as f:  # confirming takes a while
        proc = _driver(repo, f, tmp_path, shim, ["sleep 60"])
        proc.send_signal(signal.SIGINT)
        t0 = time.time()
        while f.count("POST", "/instance-operations/terminate") == 0:
            assert time.time() - t0 < 30
            time.sleep(0.02)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 1
        assert f.statuses() == ["terminated"]
        assert run_json(repo)["terminated"] is True


def test_the_ssh_remote_end_to_end_through_a_fake_ssh(repo, fake, tmp_path, shim):
    """SSHRemote itself (rsync push, detached start, polling, tar fetch), with an `ssh`
    that runs each remote command locally in a fake home directory."""
    home = tmp_path / "home"
    fake_ssh(shim, home)
    write_exp(repo, ["mkdir -p outputs", "echo from-the-instance > outputs/demo.txt",
                     "echo '{}' > outputs/demo-a.jsonl", "echo '{}' > outputs/demo-setup.json",
                     "sleep 1", "echo done"])
    env = env_for(fake)
    exp, path = lr.load_experiment(repo, "demo")
    out = io.StringIO()
    runner = lr.Runner(repo, exp, path, lr.make_api(env, rate_wait=RATE_WAIT), env,
                       remote_factory=lambda ip: lr.SSHRemote(ip, KEY_PATH, tmp_path / "kh"),
                       approve=100.0, setup_script="cd ~/tinytools && " + SETUP, poll_s=0.1,
                       boot_poll_s=0.02, reserve_s=1.0, margin_s=0.5, confirm_s=5,
                       keep_awake=False, stream=out)
    assert runner.run() == 0, out.getvalue()
    run_dir = next((repo / "runs").glob("*-demo"))
    assert (run_dir / "outputs" / "demo.txt").read_text() == "from-the-instance\n"
    assert (home / "tinytools" / "README.md").exists()  # pushed by rsync
    assert "+ echo done" in (run_dir / "log.txt").read_text()
    rj = run_json(repo)
    assert [f["path"] for f in rj["fetched"]] == ["outputs/demo-a.jsonl", "outputs/demo-setup.json",
                                                  "outputs/demo.txt"]
    assert rj["not_found"] == []
    assert fake.statuses() == ["terminated"]


def test_the_ssh_remote_stops_the_job_on_ctrl_c(tmp_path, shim):
    home = tmp_path / "home"
    fake_ssh(shim, home)
    run = home / "tinytools" / ".run"
    run.mkdir(parents=True)
    (run / "job.sh").write_text(lr.JOB_SCRIPT)
    (run / "commands.sh").write_text(lr.commands_script(("sleep 60",)))
    remote = lr.SSHRemote("127.0.0.1", KEY_PATH, tmp_path / "kh")
    t0 = time.time()
    remote.start(100)
    assert time.time() - t0 < 5
    t0 = time.time()
    while b"+ sleep" not in remote.poll(0)[0]:
        assert time.time() - t0 < 10
        time.sleep(0.05)
    assert remote.poll(0)[1] is None
    t0 = time.time()
    remote.stop()
    assert time.time() - t0 < 5  # neither start nor stop waits for the job
    pgid = int((run / "pid").read_text())
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):  # job.sh, timeout and the command: all gone
        os.killpg(pgid, 0)


# -- the other commands --------------------------------------------------------------------


def _keypair(tmp_path, name):
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    return key, (tmp_path / f"{name}.pub").read_text()


def test_doctor_passes_and_shows_no_values(tmp_path, capsys):
    key, pub = _keypair(tmp_path, "id_test")
    (tmp_path / "lambda.env").write_text("not read by this test\n")
    with FakeLambda(KEY, ssh_keys=[{"id": "k", "name": KEY_NAME, "public_key": pub}]) as f:
        env = {"LAMBDA_API_KEY": KEY, "LAMBDA_SSH_KEY_NAME": KEY_NAME,
               "LAMBDA_SSH_KEY": str(key), "LAMBDA_API_URL": f.url}
        assert lr.cmd_doctor(tmp_path, env, lr.make_api(env)) == 0
        assert f.count("POST", "/instance-operations/launch") == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "fingerprints match" in out
    for s in (KEY, KEY_NAME, str(key)):
        assert s not in out


def test_doctor_catches_a_pub_path_and_a_key_registered_from_elsewhere(tmp_path, capsys):
    key, _ = _keypair(tmp_path, "id_here")
    _, other = _keypair(tmp_path, "id_elsewhere")
    with FakeLambda(KEY, ssh_keys=[{"id": "k", "name": KEY_NAME, "public_key": other}]) as f:
        env = {"LAMBDA_API_KEY": KEY, "LAMBDA_SSH_KEY_NAME": KEY_NAME,
               "LAMBDA_SSH_KEY": str(key), "LAMBDA_API_URL": f.url}
        assert lr.cmd_doctor(tmp_path, env, lr.make_api(env)) == 1
        env["LAMBDA_SSH_KEY"] = str(key) + ".pub"
        assert lr.cmd_doctor(tmp_path, env, lr.make_api(env)) == 1
    out = capsys.readouterr().out
    assert "FAIL  it is this laptop's key (fingerprints match)" in out
    assert "FAIL  LAMBDA_SSH_KEY is the private key (not .pub)" in out
    assert "FAIL  lambda.env present" in out


def test_doctor_with_a_wrong_api_key(tmp_path, capsys):
    with FakeLambda("the-real-key-000") as f:
        env = {"LAMBDA_API_KEY": KEY, "LAMBDA_API_URL": f.url}
        assert lr.cmd_doctor(tmp_path, env, lr.make_api(env)) == 1
    out = capsys.readouterr().out
    assert "HTTP 401" in out and KEY not in out


def test_list_flags_ours_and_down_terminates(fake, capsys):
    a = fake._create({"name": "tinytools-milestone0-x", "region_name": "r", "instance_type_name": H100})
    fake._create({"name": "someone-else", "region_name": "r", "instance_type_name": H100})
    api = lr.make_api(env_for(fake), rate_wait=RATE_WAIT)
    lr.cmd_list(api)
    out = capsys.readouterr().out
    assert out.count("<- tinytools") == 1 and "jt-should-never-print" not in out
    assert lr.cmd_down(api, a, yes=False, ask=lambda q: "no") == 1
    assert fake.instances[a]["status"] == "booting"
    assert lr.cmd_down(api, a, yes=True, poll_s=0.02) == 0
    assert fake.instances[a]["status"] == "terminated"
    assert lr.cmd_down(api, a, yes=True, poll_s=0.02) == 0  # already gone: nothing to do


# -- checkpoint fetches (post 9's main run: fetch after each seed pair) ---------------------


def test_mark_lines_and_listing_are_parsed_exactly():
    assert lr.mark_lines("", "a\nTINYTOOLS_FETCH_NOW\nb") == (1, "b")
    assert lr.mark_lines("TINYTOOLS_FE", "TCH_NOW\n") == (1, "")  # split across chunks
    assert lr.mark_lines("", "+ echo TINYTOOLS_FETCH_NOW\n") == (0, "")  # the echoed command
    assert lr.mark_lines("", "TINYTOOLS_FETCH_NOW\r\nTINYTOOLS_FETCH_NOW\n") == (2, "")
    assert lr.parse_listing("12\toutputs/a b.txt\n  7\tx\nbad line\n\t\n") == [
        ("outputs/a b.txt", 12), ("x", 7)]


def test_a_checkpoint_fetches_mid_run_and_the_final_fetch_skips_unchanged_files(repo, fake,
                                                                               tmp_path):
    write_exp(repo, ["mkdir -p outputs", "echo first > outputs/demo-a.jsonl",
                     "echo TINYTOOLS_FETCH_NOW", "sleep 1",
                     "echo later > outputs/demo.txt", "echo '{}' > outputs/demo-setup.json"])
    runner, out = make_runner(repo, fake, tmp_path)
    assert runner.run() == 0, out.getvalue()
    rj = run_json(repo)
    assert rj["checkpoints"] == 1
    assert {f["path"] for f in rj["fetched"]} == {"outputs/demo-a.jsonl", "outputs/demo.txt",
                                                  "outputs/demo-setup.json"}
    assert "checkpoint 1: fetched 1 files" in out.getvalue()
    assert runner.remote.copied == ["outputs/demo-setup.json", "outputs/demo.txt"]  # a skipped
    assert fake.statuses() == ["terminated"]


class LastFetchFails(LocalRemote):
    calls = 0

    def fetch(self, *a, **k):
        LastFetchFails.calls += 1
        if LastFetchFails.calls > 1:
            raise RuntimeError("the network went away")
        return super().fetch(*a, **k)


def test_what_a_checkpoint_fetched_survives_a_failed_final_fetch(repo, fake, tmp_path):
    LastFetchFails.calls = 0
    write_exp(repo, ["mkdir -p outputs", "echo pair1 > outputs/demo-a.jsonl",
                     "echo TINYTOOLS_FETCH_NOW", "sleep 1", "echo pair2 > outputs/demo-b.jsonl",
                     "echo hi > outputs/demo.txt", "echo '{}' > outputs/demo-setup.json"])
    runner, out = make_runner(repo, fake, tmp_path, remote_cls=LastFetchFails)
    assert runner.run() == 1  # the final fetch failed: demo.txt never arrived
    rj = run_json(repo)
    assert [f["path"] for f in rj["fetched"]] == ["outputs/demo-a.jsonl"]
    run_dir = next((repo / "runs").glob("*-demo"))
    assert (run_dir / "outputs" / "demo-a.jsonl").read_text() == "pair1\n"
    assert "fetch failed: RuntimeError" in out.getvalue()
    assert fake.statuses() == ["terminated"]


def test_a_failed_checkpoint_does_not_stop_the_run(repo, fake, tmp_path):
    class FirstFetchFails(LocalRemote):
        n = 0

        def fetch(self, *a, **k):
            FirstFetchFails.n += 1
            if FirstFetchFails.n == 1:
                raise RuntimeError("blip")
            return super().fetch(*a, **k)

    write_exp(repo, [*WRITES, "echo TINYTOOLS_FETCH_NOW", "sleep 1", "echo done"])
    runner, out = make_runner(repo, fake, tmp_path, remote_cls=FirstFetchFails)
    assert runner.run() == 0, out.getvalue()
    assert "checkpoint fetch failed: RuntimeError: blip; going on" in out.getvalue()
    assert run_json(repo)["checkpoints"] == 0 and len(run_json(repo)["fetched"]) == 3


def test_the_ssh_remote_checkpoint_and_skip_through_a_fake_ssh(repo, fake, tmp_path, shim):
    home = tmp_path / "home"
    fake_ssh(shim, home)
    write_exp(repo, ["mkdir -p outputs", "echo pair-one > outputs/demo-a.jsonl",
                     "echo TINYTOOLS_FETCH_NOW", "sleep 1", "echo more >> outputs/demo-a.jsonl",
                     "echo x > outputs/demo.txt", "echo '{}' > outputs/demo-setup.json"])
    env = env_for(fake)
    exp, path = lr.load_experiment(repo, "demo")
    out = io.StringIO()
    runner = lr.Runner(repo, exp, path, lr.make_api(env, rate_wait=RATE_WAIT), env,
                       remote_factory=lambda ip: lr.SSHRemote(ip, KEY_PATH, tmp_path / "kh"),
                       approve=100.0, setup_script="cd ~/tinytools && " + SETUP, poll_s=0.1,
                       boot_poll_s=0.02, reserve_s=1.0, margin_s=0.5, confirm_s=5,
                       keep_awake=False, stream=out)
    assert runner.run() == 0, out.getvalue()
    run_dir = next((repo / "runs").glob("*-demo"))
    # appended to after the checkpoint: its size changed, so the final fetch copied it again
    assert (run_dir / "outputs" / "demo-a.jsonl").read_text() == "pair-one\nmore\n"
    assert run_json(repo)["checkpoints"] == 1
    assert "checkpoint 1: fetched 1 files (0 unchanged)" in out.getvalue()


def test_a_same_size_rewrite_after_a_checkpoint_is_fetched_again(repo, fake, tmp_path, shim):
    """T8 review S6: the final fetch copies a file a checkpoint has if it changed after
    that checkpoint's marker (.run/fetch-N, find -cnewer), whatever its size; files that
    did not change are not copied again. Through the real SSHRemote and a fake ssh."""
    home = tmp_path / "home"
    fake_ssh(shim, home)
    write_exp(repo, ["mkdir -p outputs", "echo 'verdict: AAAA' > outputs/demo-a.jsonl",
                     "echo same > outputs/demo-b.jsonl", "echo TINYTOOLS_FETCH_NOW", "sleep 1",
                     "echo 'verdict: BBBB' > outputs/demo-a.jsonl.tmp",
                     "mv outputs/demo-a.jsonl.tmp outputs/demo-a.jsonl",  # written whole, renamed
                     "echo x > outputs/demo.txt", "echo '{}' > outputs/demo-setup.json"])
    env = env_for(fake)
    exp, path = lr.load_experiment(repo, "demo")
    out = io.StringIO()
    runner = lr.Runner(repo, exp, path, lr.make_api(env, rate_wait=RATE_WAIT), env,
                       remote_factory=lambda ip: lr.SSHRemote(ip, KEY_PATH, tmp_path / "kh"),
                       approve=100.0, setup_script="cd ~/tinytools && " + SETUP, poll_s=0.1,
                       boot_poll_s=0.02, reserve_s=1.0, margin_s=0.5, confirm_s=5,
                       keep_awake=False, stream=out)
    assert runner.run() == 0, out.getvalue()
    run_dir = next((repo / "runs").glob("*-demo"))
    assert (run_dir / "outputs" / "demo-a.jsonl").read_text() == "verdict: BBBB\n"
    assert "checkpoint 1: fetched 2 files (0 unchanged)" in out.getvalue()
    assert (runner.state.fetch_marks, runner.state.fetched_mark) == (2, 2)
    marks = sorted(p.name for p in (home / "tinytools" / ".run").glob("fetch-*"))
    assert marks == ["fetch-1", "fetch-2"]


def test_a_failed_fetch_leaves_its_files_to_the_next(repo, fake, tmp_path):
    """Only a fetch that completes moves the marker the next one compares against."""
    class FailsAfterMarking(LocalRemote):
        n = 0
        seen: list = []

        def fetch(self, patterns, dest, timeout, mark=1, since=None, have=()):
            FailsAfterMarking.n += 1
            FailsAfterMarking.seen.append((mark, since, sorted(have)))
            if FailsAfterMarking.n == 2:  # the second checkpoint: marked, then the network
                (self.dir / ".run" / f"fetch-{mark}").touch()
                raise RuntimeError("blip")
            return super().fetch(patterns, dest, timeout, mark, since, have)

    FailsAfterMarking.n, FailsAfterMarking.seen = 0, []
    write_exp(repo, ["mkdir -p outputs", "echo one > outputs/demo-a.jsonl",
                     "echo TINYTOOLS_FETCH_NOW", "sleep 1", "echo two > outputs/demo-a.jsonl",
                     "echo TINYTOOLS_FETCH_NOW", "sleep 1",
                     "echo x > outputs/demo.txt", "echo '{}' > outputs/demo-setup.json"])
    runner, out = make_runner(repo, fake, tmp_path, remote_cls=FailsAfterMarking)
    assert runner.run() == 0, out.getvalue()
    # the final fetch (mark 3) compares against fetch-1, the last that completed
    assert [(m, s) for m, s, _ in FailsAfterMarking.seen] == [(1, None), (2, 1), (3, 1)]
    run_dir = next((repo / "runs").glob("*-demo"))
    assert (run_dir / "outputs" / "demo-a.jsonl").read_text() == "two\n"
