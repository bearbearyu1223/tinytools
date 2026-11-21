"""The runner's spend safety (review tinytools-runner-review-1, findings 1-4 and 6-15),
against the fake API's realistic modes. Nothing here talks to Lambda."""

import io
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fake_lambda import H100, FakeLambda, LocalRemote, timeout_shim

from tinytools import lambda_run as lr

KEY = "sk-test-SECRET-0123456789abcdef"
KEY_NAME = "laptop-key-NAME-7f3a"
KEY_PATH = "/nonexistent/SECRETPATH/id_ed25519"
SETUP = "echo 'TINYTOOLS_SETUP_JSON {\"gpu\": \"fake GPU\"}'"
LAUNCH = ("POST", "/instance-operations/launch")
TERMINATE = ("POST", "/instance-operations/terminate")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t",
                    *args], check=True, capture_output=True)


def write_exp(repo: Path, commands: list[str], *, name: str = "demo", post: int = 9,
              max_hours: float = 1.0, cap: float = 15.0) -> None:
    fetch = ["outputs/demo.txt", "outputs/demo-*.jsonl"]
    (repo / "experiments").mkdir(exist_ok=True)
    (repo / "experiments" / f"{name}.toml").write_text(
        f'name = "{name}"\npost = {post}\ninstance_type = "{H100}"\nmax_hours = {max_hours}\n'
        f"expected_hours = {max_hours / 2}\ncap_usd = {cap}\n"
        f"commands = {json.dumps(commands)}\nfetch = {json.dumps(fetch)}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "experiment")


# Commands that write a file for each fetch pattern (a pattern that matches nothing
# written makes the run exit 1)
WRITES = ["mkdir -p outputs", "echo hi > outputs/demo.txt", "echo '{}' > outputs/demo-a.jsonl"]


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "README.md").write_text("demo repo\n")
    write_exp(r, WRITES)
    return r


@pytest.fixture(autouse=True)
def shim(tmp_path, monkeypatch):
    bin_dir = timeout_shim(tmp_path / "bin")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def env_for(url: str) -> dict:
    return {"LAMBDA_API_KEY": KEY, "LAMBDA_SSH_KEY_NAME": KEY_NAME, "LAMBDA_SSH_KEY": KEY_PATH,
            "LAMBDA_API_URL": url}


def make_runner(repo, fake, tmp_path, *, remote_cls=LocalRemote, exp="demo", **kw):
    env = env_for(fake.url)
    e, path = lr.load_experiment(repo, exp)
    out = io.StringIO()
    settings = dict(approve=100.0, setup_script=SETUP, poll_s=0.05, boot_poll_s=0.02,
                    reserve_s=1.0, margin_s=0.5, confirm_s=5.0, keep_awake=False, stream=out,
                    lost_launch_s=1.0)
    settings.update(kw)
    runner = lr.Runner(repo, e, path, lr.make_api(env, rate_wait=0.05), env,
                       remote_factory=lambda ip: remote_cls(tmp_path / "remote"), **settings)
    return runner, out


def run_json(repo: Path) -> dict:
    (path,) = list((repo / "runs").glob("*/run.json"))
    return json.loads(path.read_text())


# -- 1. nothing escapes _call; the cleanup survives anything ------------------------------


def test_a_cut_off_terminate_answer_is_retried(repo, tmp_path):
    """The terminate takes effect but its answer is cut off mid-body (IncompleteRead,
    an http.client.HTTPException): the runner keeps going and confirms."""
    with FakeLambda(KEY, boot_polls=1, term_polls=3) as fake:
        fake.truncate[TERMINATE] = 1
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_OK
        assert fake.statuses() == ["terminated"]
        assert "IncompleteRead" in out.getvalue()
        assert run_json(repo)["terminated"] is True


def test_call_turns_every_failure_into_an_api_error():
    api = lr.LambdaAPI("notaurl", KEY, lr.Redactor([KEY]))  # urllib raises ValueError
    with pytest.raises(lr.ApiError) as e:
        api.instances()
    assert e.value.status is None
    with FakeLambda(KEY) as fake:
        fake.truncate[("GET", "/instances")] = 1
        with pytest.raises(lr.ApiError, match="IncompleteRead"):
            lr.make_api(env_for(fake.url)).instances()


def test_a_crashing_remote_stop_and_fetch_do_not_stop_the_terminate(repo, tmp_path):
    class Broken(LocalRemote):
        def stop(self, wait_s=5.0):
            raise RuntimeError("stop exploded")

        def fetch(self, *a, **k):
            raise KeyError("fetch exploded")

    write_exp(repo, ["exit 4"])
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, remote_cls=Broken)
        assert runner.run() == lr.EXIT_FAILED
        assert fake.statuses() == ["terminated"]
        assert "stopping the job failed: RuntimeError: stop exploded" in out.getvalue()


# -- 2. terminate: wait while booting, retry until confirmed, re-send ------------------------


def test_terminate_waits_for_a_booting_instance_without_hammering(repo, tmp_path):
    """Boot never finishes within the runner's boot timeout; Lambda refuses terminate
    until the VM has launched (actions.terminate.available false). The runner waits,
    sends terminate once it may, and confirms."""
    with FakeLambda(KEY, boot_polls=40, term_polls=1, blocked_while_booting=True) as fake:
        runner, out = make_runner(repo, fake, tmp_path, boot_timeout_s=0.05)
        assert runner.run() == lr.EXIT_FAILED
        assert fake.statuses() == ["terminated"]
        assert fake.count(*TERMINATE) == 1  # not sent while Lambda said it can't be
        assert "vm-has-not-launched" in out.getvalue()
        assert run_json(repo)["terminated"] is True


def test_terminate_keeps_trying_past_twenty_failures(monkeypatch):
    monkeypatch.setattr(lr.time, "sleep", lambda s: None)
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        iid = fake._create({"name": "tinytools-demo-x", "region_name": "us-east-1",
                            "instance_type_name": H100})
        fake.fail[TERMINATE] = [(503, {"error": {"code": "x", "message": "busy"}})] * 30
        said = []
        assert lr.terminate_and_confirm(lr.make_api(env_for(fake.url)), [iid], said.append,
                                        0.0, 30)
        assert fake.count(*TERMINATE) == 31
        assert any("STILL NOT TERMINATED after 20 tries" in s for s in said)


def test_terminate_is_resent_until_it_takes():
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        iid = fake._create({"name": "tinytools-demo-x", "region_name": "us-east-1",
                            "instance_type_name": H100})
        fake.fail[TERMINATE] = [(200, {"data": {"terminated_instances": []}})] * 2
        said = []
        assert lr.terminate_and_confirm(lr.make_api(env_for(fake.url)), [iid], said.append,
                                        0.01, 5, resend_s=0.05)
        assert fake.count(*TERMINATE) == 3
        assert sum("re-sent" in s for s in said) == 2


def test_a_404_for_an_instance_never_seen_is_not_taken_as_gone():
    """Just after a launch Lambda may not list the instance yet: a 404 then is not proof
    it is gone. The id is terminated anyway, and confirmed once it shows. (Here
    terminate's answer names no instance; an id it names counts as seen: next test.)"""
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        iid = fake._create({"name": "tinytools-demo-x", "region_name": "us-east-1",
                            "instance_type_name": H100}, hidden=3)
        api = lr.make_api(env_for(fake.url))
        real = api.terminate
        api.terminate = lambda ids: (real(ids), [])[1]
        assert lr.terminate_and_confirm(api, [iid], lambda m: None, 0.01, 5)
        assert fake.statuses() == ["terminated"]


def test_an_id_named_in_terminates_answer_is_gone_at_its_next_404():
    """Lambda's terminate answer lists the Instance objects it is terminating: such an
    id has been shown, so a 404 afterwards means it has left the API (round 2, R2-2)."""
    with FakeLambda(KEY, boot_polls=1, term_polls=10**9) as fake:
        iid = fake._create({"name": "tinytools-demo-x", "region_name": "us-east-1",
                            "instance_type_name": H100}, hidden=10**6)
        said = []
        t0 = time.time()
        assert lr.terminate_and_confirm(lr.make_api(env_for(fake.url)), [iid], said.append,
                                        0.01, 30)
        assert time.time() - t0 < 5
        assert fake.count(*TERMINATE) == 1 and said[-1].startswith("terminated: Lambda confirms")


# -- 3. lost launches ------------------------------------------------------------------------


def test_a_launch_cut_off_mid_answer_is_found_late_and_terminated(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1, list_lag=3) as fake:
        fake.truncate[LAUNCH] = 1
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_FAILED
        assert fake.statuses() == ["terminated"]
        rj = run_json(repo)
        assert rj["terminated"] is True and rj["instance"]["id"] == "i-0001"
        assert rj["cost_usd"] > 0


def test_the_final_check_terminates_this_runs_instance_found_after_the_lookup(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1, list_lag=1) as fake:
        fake.fail[LAUNCH] = [(502, {"error": {"code": "x", "message": "bad gateway"}})]
        fake.launch_creates_on_error = True
        runner, out = make_runner(repo, fake, tmp_path, lost_launch_s=0.0)
        assert runner.run() == lr.EXIT_FAILED
        assert fake.statuses() == ["terminated"]
        assert "STILL UP (this run's): i-0001: terminating" in out.getvalue()
        rj = run_json(repo)
        assert rj["terminated"] is True and rj["instance"]["id"] == "i-0001"


def test_a_definite_4xx_launch_is_not_looked_for(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, ssh_keys=[{"id": "k", "name": "another-key",
                                                  "public_key": "x"}]) as fake:
        runner, out = make_runner(repo, fake, tmp_path, lost_launch_s=60.0)
        t0 = time.time()
        assert runner.run() == lr.EXIT_FAILED
        assert time.time() - t0 < 10
        assert fake.instances == {}
        assert fake.count("GET", "/instances") == 2  # preflight and the final check only
        rj = run_json(repo)
        assert rj["cost_usd"] == 0.0 and "no such SSH key" in rj["status"]


def test_the_launch_carries_a_tag_unique_to_the_run(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, _ = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_OK
        (body,) = [b for m, p, b in fake.calls if (m, p) == LAUNCH]
        assert body["tags"] == [{"key": lr.RUN_TAG, "value": run_json(repo)["instance"]["tag"]}]
        assert run_json(repo)["instance"]["tag"].startswith(runner.state.run_dir.name)


def test_a_launch_that_refuses_tags_is_retried_without_them(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        fake.fail[LAUNCH] = [(400, {"error": {"code": "global/invalid-parameters",
                                              "message": "unknown field: tags"}})]
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_OK
        assert fake.count(*LAUNCH) == 2 and fake.statuses() == ["terminated"]
        assert "tags" not in [b for m, p, b in fake.calls if (m, p) == LAUNCH][-1]


# -- 4. exit code 3 ------------------------------------------------------------------------


def test_down_exits_3_when_not_confirmed():
    with FakeLambda(KEY, boot_polls=1, term_polls=10**9) as fake:
        iid = fake._create({"name": "tinytools-x", "region_name": "us-east-1",
                            "instance_type_name": H100})
        api = lr.make_api(env_for(fake.url))
        assert lr.cmd_down(api, iid, yes=True, poll_s=0.01, confirm_s=0.2) == lr.EXIT_UNCONFIRMED


# -- 6. fetch: only what the run wrote ------------------------------------------------------


def test_fetch_takes_rewritten_files_and_leaves_untouched_committed_ones(repo, tmp_path):
    (repo / "outputs").mkdir()
    (repo / "outputs" / "demo.txt").write_text("made on the Mac\n")
    (repo / "outputs" / "demo-old.jsonl").write_text("{}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "Mac captures")
    write_exp(repo, ["echo from-the-run > outputs/demo.txt"])
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        # the commands succeeded, but demo-*.jsonl matched nothing written: exit 1
        assert runner.run() == lr.EXIT_FAILED
        assert "nothing they wrote matched outputs/demo-*.jsonl" in out.getvalue()
    rj = run_json(repo)
    assert [f["path"] for f in rj["fetched"]] == ["outputs/demo.txt"]
    assert rj["not_found"] == ["outputs/demo-*.jsonl"]
    run_dir = next((repo / "runs").glob("*-demo"))
    assert (run_dir / "outputs" / "demo.txt").read_text() == "from-the-run\n"


def test_fetch_budget_leaves_time_to_terminate():
    assert lr.fetch_budget(hard_end=1000.0, now=0.0, terminate_allow=300) == lr.FETCH_TIMEOUT_S
    assert lr.fetch_budget(hard_end=1000.0, now=600.0, terminate_allow=300) == 100.0
    assert lr.fetch_budget(hard_end=1000.0, now=800.0, terminate_allow=300) == 0.0


# -- 7. redirects and the base URL ------------------------------------------------------------


def test_the_default_base_and_a_redirect_is_an_error_naming_the_location(capsys, tmp_path):
    assert lr.API_URL == "https://cloud.lambda.com/api/v1"
    assert lr.make_api({}).base == lr.API_URL
    assert lr.make_api({}).min_interval == lr.API_MIN_INTERVAL_S  # Lambda: paced 1 req/s
    with FakeLambda(KEY) as target, FakeLambda(KEY, redirect_to=target.url) as old:
        env = {**env_for(old.url), "LAMBDA_SSH_KEY": ""}
        assert lr.cmd_doctor(tmp_path, env, lr.make_api(env)) == 1
        assert target.calls == []
        for m, p in (LAUNCH, TERMINATE):
            with pytest.raises(lr.ApiError) as e:
                lr.make_api(env)._call(m, p, {"instance_ids": []})
            assert e.value.status == 301 and f"redirects to {target.url}" in str(e.value)
        assert target.calls == []
    out = capsys.readouterr().out
    assert f"API at {old.url}" in out and "redirects to" in out and KEY not in out


def test_doctor_reports_the_base_it_reached(tmp_path, capsys):
    with FakeLambda(KEY) as fake:
        lr.cmd_doctor(tmp_path, env_for(fake.url), lr.make_api(env_for(fake.url)))
    assert f"API reachable at {fake.url} (2 instance types listed)" in capsys.readouterr().out


def test_lambda_request_id_is_kept_in_errors():
    with FakeLambda(KEY) as fake:
        with pytest.raises(lr.ApiError, match=r"no such instance \(request_id req-\d+\)"):
            lr.make_api(env_for(fake.url))._call("GET", "/instances/i-9999")
    assert lr._error_text(b'{"error": {"code": "c", "message": "m", "request_id": "r1"}}') == \
        "c: m (request_id r1)"
    assert lr._error_text(b"oops", {"X-Request-Id": "r2"}) == "oops (request_id r2)"


# -- rate limits ------------------------------------------------------------------------------


def test_the_runner_copes_with_rate_limits(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1, rate_limit=(0.2, 2.4)) as fake:
        fake._last_launch = time.monotonic()  # someone launched just before
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_OK, out.getvalue()
        assert fake.limited > 0 and fake.statuses() == ["terminated"]


def test_pacing_keeps_under_the_rate_limit():
    with FakeLambda(KEY, rate_limit=(0.2, 12.0)) as fake:
        api = lr.make_api(env_for(fake.url), min_interval=0.25)
        for _ in range(4):
            api.instances()
        assert fake.limited == 0


# -- 9. liveness, instance status, SSH limit ----------------------------------------------------


@pytest.mark.parametrize("status", ["unhealthy", "preempted"])
def test_an_instance_that_turns_bad_mid_run_ends_the_run(repo, tmp_path, status):
    write_exp(repo, ["sleep 30"])
    with FakeLambda(KEY, boot_polls=1, term_polls=1, fate=(2, status)) as fake:
        runner, out = make_runner(repo, fake, tmp_path, status_check_s=0.1)
        t0 = time.time()
        assert runner.run() == lr.EXIT_FAILED
        assert time.time() - t0 < 15
        rj = run_json(repo)
        assert f"instance status is '{status}'" in rj["status"] and rj["terminated"] is True
        # preempted is already ended: nothing to terminate; unhealthy is terminated
        assert fake.count(*TERMINATE) == (0 if status == "preempted" else 1)
        assert fake.statuses() == ["preempted" if status == "preempted" else "terminated"]


def test_ssh_that_never_answers_is_given_up_on(repo, tmp_path):
    class Silent(LocalRemote):
        def ready(self):
            return False

    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, remote_cls=Silent, ssh_wait_s=0.2)
        assert runner.run() == lr.EXIT_FAILED
        assert "SSH did not answer" in run_json(repo)["status"]
        assert fake.statuses() == ["terminated"]


def test_ssh_never_reads_the_terminal(monkeypatch, tmp_path):
    seen = []

    def fake_run(args, **kw):
        seen.append((args, kw))
        return subprocess.CompletedProcess(args, 0, "running\n" if kw.get("text") else b"0\n", "")

    monkeypatch.setattr(lr.subprocess, "run", fake_run)
    r = lr.SSHRemote("203.0.113.9", KEY_PATH, tmp_path / "kh")
    r.ready()
    r.poll(0)
    r.stop()
    r.run("echo hi", 5)
    for args, kw in seen[:3]:
        assert "-n" in args and kw.get("stdin") is subprocess.DEVNULL
    args, kw = seen[3]
    assert kw.get("input") == "echo hi" and "-n" not in args
    assert "pkill -TERM -s" in seen[2][0][-1]


# -- 11. redaction --------------------------------------------------------------------------------


def test_the_redactor_scrubs_key_paths_in_every_form_but_not_non_secrets():
    home = os.path.expanduser("~")
    r = lr.make_api({"LAMBDA_API_KEY": KEY, "LAMBDA_SSH_KEY_NAME": "h100",
                     "LAMBDA_SSH_KEY": "~/.ssh/review-key"}).redact
    assert r(f"ssh -i {home}/.ssh/review-key host") == "ssh -i [redacted] host"
    assert r("ssh -i ~/.ssh/review-key host") == "ssh -i [redacted] host"
    assert r(f"key {KEY}") == "key [redacted]"
    # the key's name at Lambda is not a secret: an h100 run's lines stay readable
    assert r("gpu_1x_h100_sxm5 in us-east-1") == "gpu_1x_h100_sxm5 in us-east-1"


# -- 14. preempted counts as ended ------------------------------------------------------------


def test_preempted_is_not_live():
    assert not lr.live({"status": "preempted"}) and not lr.live({"status": "terminated"})
    assert lr.live({"status": "unhealthy"}) and lr.live({"status": "terminating"})


# -- 15. caps per post --------------------------------------------------------------------------


def test_the_cap_is_summed_per_post(repo, tmp_path):
    lr.write_ledger_row(repo / "runs" / "ledger.csv",
                        {"run": "r1", "experiment": "other", "post": "9", "cost_usd": "48.0000"})
    lr.write_ledger_row(repo / "runs" / "ledger.csv",
                        {"run": "r2", "experiment": "x", "post": "10", "cost_usd": "40.0000"})
    with FakeLambda(KEY) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_REFUSED
        assert "left of post 9's $50.00 cap" in out.getvalue()
        assert fake.count(*LAUNCH) == 0
    rows = lr.read_ledger(repo / "runs" / "ledger.csv")
    assert lr.spent_on_post(rows, 9) == 48.0 and lr.spent_on_post(rows, 10) == 40.0


def test_post_is_required_and_names_fit_lambdas_limit():
    good = ('name = "demo"\npost = 9\ninstance_type = "x"\nmax_hours = 1\n'
            'expected_hours = 1\ncap_usd = 5\ncommands = ["true"]\nfetch = ["a"]\n')
    assert lr.parse_experiment(good, "demo").post == 9
    with pytest.raises(ValueError, match="missing keys \\['post'\\]"):
        lr.parse_experiment(good.replace("post = 9\n", ""), "demo")
    long = "x" * 40
    with pytest.raises(ValueError, match="at most 64"):
        lr.parse_experiment(good.replace('"demo"', f'"{long}"'), long)
    assert len(lr.instance_name("milestone0", time.time())) <= lr.NAME_MAX
