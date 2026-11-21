#!/usr/bin/env bash
# Run tinytools experiments on Lambda Cloud from this laptop. See docs/lambda-playbook.md.
#
#   scripts/lambda.sh doctor | types | estimate EXP | run EXP [--approve USD] | list | down ID
#
# The runner is src/tinytools/lambda_run.py (standard library only). It is exec'd as the
# project's own Python, with no `uv run` process in between, so Ctrl-C reaches the runner
# itself and its cleanup (fetch, terminate) is not cut short by a parent exiting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -x .venv/bin/python ]] || uv sync --quiet
exec .venv/bin/python -m tinytools.lambda_run "$@"
