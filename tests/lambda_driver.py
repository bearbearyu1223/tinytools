"""Run the Lambda runner in its own process against the fake API, so a test can send it
SIGINT or SIGTERM. Usage: python lambda_driver.py SETTINGS_JSON (see test_lambda_run.py)."""

import json
import sys
from pathlib import Path

from fake_lambda import LocalRemote

from tinytools import lambda_run

s = json.loads(sys.argv[1])
env = s["env"]
api = lambda_run.make_api(env, rate_wait=s.get("rate_wait", 1.0))
repo = Path(s["repo"])
exp, path = lambda_run.load_experiment(repo, s["experiment"])
runner = lambda_run.Runner(
    repo, exp, path, api, env, remote_factory=lambda ip: LocalRemote(Path(s["remote"])),
    approve=100.0, setup_script=s["setup_script"], poll_s=0.05, boot_poll_s=0.02,
    reserve_s=s["reserve_s"], margin_s=s["margin_s"], confirm_s=10, keep_awake=True)
sys.exit(runner.run())
