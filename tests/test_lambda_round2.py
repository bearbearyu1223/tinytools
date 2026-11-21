# ruff: noqa: F811  (pytest fixtures imported from test_lambda_spend)
"""The runner after the second spend-safety review (tinytools-runner-review-1, round 2,
R2-1..R2-7). Fake API and local processes only; nothing talks to Lambda."""

import time

import pytest
from fake_lambda import H100, FakeLambda, LocalRemote
from test_lambda_spend import (  # noqa: F401  (fixtures)
    KEY, LAUNCH, WRITES, env_for, make_runner, repo, run_json, shim, write_exp)

from tinytools import lambda_run as lr

# -- R2-1. polls that keep failing end the run ---------------------------------------------


class SilentAfterStart(LocalRemote):
    def poll(self, offset):
        raise RuntimeError("poll failed")


def test_polls_failing_longer_than_the_ssh_limit_raise_and_terminate(repo, tmp_path):
    write_exp(repo, ["sleep 1"], max_hours=0.02)  # deadline ~70 s away
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, remote_cls=SilentAfterStart,
                                  ssh_wait_s=1.0)
        t0 = time.time()
        assert runner.run() == lr.EXIT_FAILED
        assert time.time() - t0 < 15
        assert fake.statuses() == ["terminated"]
    status = run_json(repo)["status"]
    assert "stopped answering SSH" in status and "polls failed for" in status


def test_a_poll_that_recovers_resets_the_clock(repo, tmp_path):
    class Flaky(LocalRemote):
        n = 0

        def poll(self, offset):
            Flaky.n += 1
            if Flaky.n % 10:  # nine failures, then one answer: never 1 s of failures in a row
                raise RuntimeError("poll failed")
            return super().poll(offset)

    write_exp(repo, [*WRITES, "sleep 2"], max_hours=0.02)
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, remote_cls=Flaky, ssh_wait_s=1.0,
                                  poll_s=0.02)
        assert runner.run() == lr.EXIT_OK, out.getvalue()[-600:]


# -- R2-3. a launch interrupted mid-request ----------------------------------------------------


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), lr.Interrupted("SIGTERM"),
                                 lr.Interrupted("SIGHUP")], ids=["SIGINT", "SIGTERM", "SIGHUP"])
def test_an_interrupted_launch_is_looked_for_and_never_exits_1(repo, tmp_path, exc):
    """The POST took effect, the answer never arrived, the lookup saw nothing (list lag)
    and the final list failed: exit 3, charged at the worst case."""
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, lost_launch_s=0.1)

        def launch(*a, **k):
            assert runner.state.launch_uncertain  # set before the request goes out
            fake._create({"name": a[3], "region_name": a[1], "instance_type_name": a[0]},
                         hidden=10**6)
            raise exc

        runner.api.launch = launch
        real_final = runner._final_check

        def final_check():
            runner.api.instances = lambda: (_ for _ in ()).throw(lr.ApiError(503, "HTTP 503"))
            return real_final()

        runner._final_check = final_check
        assert runner.run() == lr.EXIT_UNCONFIRMED
    rj = run_json(repo)
    assert rj["terminated"] is False
    assert rj["cost_usd"] == lr.cost_usd(runner.price, runner.exp.max_hours * 3600)


def test_a_definite_refusal_clears_the_uncertainty(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, ssh_keys=[{"id": "k", "name": "another-key",
                                                  "public_key": "x"}]) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_FAILED
        assert not runner.state.launch_uncertain and runner.state.launch_refused
        assert run_json(repo)["cost_usd"] == 0.0


def test_a_successful_launch_clears_the_uncertainty(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_OK
        assert not runner.state.launch_uncertain


# -- R2-4. exit 137 ------------------------------------------------------------------------


def test_exit_status():
    assert lr.exit_status(0, 5, 100) == "ok"
    assert lr.exit_status(124, 100, 100) == "time limit"
    assert lr.exit_status(124, 3, 100) == "time limit"  # GNU timeout's own code
    assert lr.exit_status(137, 160, 100) == "time limit"  # KILL after --kill-after
    assert lr.exit_status(137, 100, 100) == "time limit"
    assert lr.exit_status(137, 99.9, 100) == lr.KILLED_STATUS
    assert lr.exit_status(137, 3, 100) == "killed (SIGKILL, e.g. out of memory)"
    assert lr.exit_status(3, 200, 100) == "failed: a command exited 3"


def test_a_sigkilled_command_is_recorded_as_killed(repo, tmp_path):
    write_exp(repo, ["bash -c 'kill -KILL $$'"], max_hours=0.05)
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_FAILED
    rj = run_json(repo)
    assert rj["status"] == lr.KILLED_STATUS and rj["remote_exit_code"] == 137


# -- R2-5 (runner side). the commands are told the deadline ------------------------------------


def test_the_commands_see_when_the_remote_limit_fires(repo, tmp_path):
    write_exp(repo, [*WRITES, f'echo "${lr.DEADLINE_ENV}" > outputs/demo-deadline.jsonl'],
              max_hours=0.05)
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        t0 = time.time()
        assert runner.run() == lr.EXIT_OK, out.getvalue()[-600:]
        secs = lr.remote_seconds(runner.deadline, t0, runner.margin_s)
    epoch = int((tmp_path / "remote" / "tinytools" / "outputs" / "demo-deadline.jsonl")
                .read_text())
    assert abs(epoch - (t0 + secs)) < 15
    assert epoch <= runner.deadline


# -- R2-6. a fetch pattern that matched nothing -------------------------------------------------


def test_commands_that_wrote_nothing_fetched_exit_1(repo, tmp_path):
    write_exp(repo, ["echo hi"])
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_FAILED
        assert fake.statuses() == ["terminated"]
    assert "nothing they wrote matched outputs/demo.txt, outputs/demo-*.jsonl" in out.getvalue()
    assert run_json(repo)["not_found"] == ["outputs/demo.txt", "outputs/demo-*.jsonl"]


# -- R2-7. setup's own limit; the tag refusal ---------------------------------------------------


def test_setup_has_its_own_time_limit(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, setup_script="sleep 30",
                                  setup_timeout_s=1.0)
        t0 = time.time()
        assert runner.run() == lr.EXIT_FAILED
        assert time.time() - t0 < 15
        assert fake.statuses() == ["terminated"]
    assert "setup did not finish within" in run_json(repo)["status"]


def test_a_slow_push_counts_against_the_setup_limit(repo, tmp_path):
    class SlowPush(LocalRemote):
        def push(self, src, timeout):
            assert timeout <= 1.0
            time.sleep(1.2)
            super().push(src, timeout)

    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        runner, out = make_runner(repo, fake, tmp_path, remote_cls=SlowPush, setup_timeout_s=1.0)
        assert runner.run() == lr.EXIT_FAILED
    assert "setup did not finish within" in run_json(repo)["status"]


def _err(status, code, message):
    return lr.ApiError(status, f"HTTP {status}: {code}: {message}", code=code,
                       api_message=message)


def test_the_tag_refusal_matches_the_error_code_and_field():
    bad = "global/invalid-parameters"
    assert lr.refuses_tags(_err(400, bad, "unknown field: tags"))
    assert lr.refuses_tags(_err(400, bad, "tags: a list of {key, value}"))
    assert not lr.refuses_tags(_err(400, bad, "vm-has-not-launched: stage 2"))
    assert not lr.refuses_tags(_err(400, bad, "ssh_key_names: no such SSH key 'vintage'"))
    assert not lr.refuses_tags(_err(400, "instance-operations/launch/insufficient-capacity",
                                    "no capacity for tags"))
    assert not lr.refuses_tags(_err(500, bad, "unknown field: tags"))
    assert not lr.refuses_tags(lr.ApiError(400, "HTTP 400: unknown field: tags"))  # no code


def test_a_refusal_that_only_mentions_tag_in_passing_is_not_retried(repo, tmp_path):
    with FakeLambda(KEY, boot_polls=1, term_polls=1) as fake:
        fake.fail[LAUNCH] = [(400, {"error": {"code": "global/invalid-parameters",
                                              "message": "region: not in this stage of tagging"}})]
        runner, out = make_runner(repo, fake, tmp_path)
        assert runner.run() == lr.EXIT_FAILED
        assert fake.count(*LAUNCH) == 1 and fake.instances == {}


def test_the_api_error_carries_the_code_and_message():
    with FakeLambda(KEY, boot_polls=1) as fake:
        api = lr.make_api(env_for(fake.url))
        with pytest.raises(lr.ApiError) as e:
            api.launch(H100, "us-east-1", "", "tinytools-x")
        assert e.value.status == 400 and e.value.code == "global/invalid-parameters"
        assert e.value.api_message == "ssh_key_names: exactly one key name"
