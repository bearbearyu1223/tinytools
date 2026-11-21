# Running tinytools experiments on Lambda Cloud

This is the procedure every GPU run in this repo follows, from "I want to run X" to "the
numbers are in `outputs/` and the instance is gone". It exists because a cloud GPU bills by
the hour whether or not anything useful is happening, and because a result is only worth
publishing if someone else can rerun it from what's written down.

> Status: design. The commands below are the interface the runner is being built to.
> Ones marked *(planned)* don't exist yet.

## The rules

1. **Nothing launches without an approved estimate.** Every run starts with `estimate`, which
   prints the instance type, the live hourly price from Lambda's API, the expected hours and the
   cost. The author approves that number; the run is refused if it would push the experiment's
   total past its cap.
2. **Billing stops only when the instance is terminated through the API.** Shutting the machine
   down from inside does not stop the bill. So the runner terminates from the laptop: on
   success, on any error, on Ctrl-C, and when the hard time limit is reached.
3. **Every run ends with a check that nothing is left running.** `list` must show no instances
   tagged for this repo. If it doesn't, terminate by hand before anything else.
4. **Every run leaves a record.** `runs/<date>-<experiment>/run.json`: the experiment, git
   commit, instance type, price, start and end time, cost, package versions, seeds,
   determinism settings, and the sha256 of every file fetched back.
5. **Secrets never leave `lambda.env`.** It's gitignored. The runner reads keys from it and never
   prints them; logs show key names and lengths only.
6. **Numbers in a post come from fetched captures, never from a terminal scrollback.**

## One-time setup

| step | how | what goes wrong |
|---|---|---|
| Lambda account and API key | dashboard, API keys | - |
| `lambda.env` | `cp lambda.env.example lambda.env`, fill in `LAMBDA_API_KEY`, `LAMBDA_SSH_KEY_NAME`, `LAMBDA_SSH_KEY` | the SSH key path must be the private key, not the `.pub` |
| SSH key registered with Lambda | dashboard, SSH keys, from this laptop's public key | a key registered from another machine won't match |
| key loaded | `ssh-add <key>` | a passphrase key not in the agent makes every step hang |
| check | `scripts/lambda.sh doctor` *(planned)* | checks all of the above without launching |

## An experiment is a file

Each experiment is `experiments/<name>.toml` *(planned)*, committed before it runs:

```toml
name = "milestone0"
post = 9
instance_type = "gpu_1x_h100_sxm5"   # checked against the live list at launch
max_hours = 2.5                      # hard limit: the runner terminates at this point
cap_usd = 15                         # refuse to launch past this total for the experiment
commands = [
  "uv run tinytools milestone0",
]
fetch = ["outputs/milestone0.txt", "outputs/milestone0-*.jsonl"]
```

The file is the recipe a post prints: same commands, same settings, same commit.

## A run, step by step

```
scripts/lambda.sh estimate milestone0   # price, hours, cost, remaining cap
scripts/lambda.sh run milestone0        # up -> setup -> run -> fetch -> down, all in one
scripts/lambda.sh list                  # must show nothing running
```

`run` does, in order (each step is also its own subcommand, for recovery):

| step | what happens |
|---|---|
| `up` | launch one instance of the type, tagged with the repo and experiment; wait until SSH answers |
| `setup` | copy the repo at the recorded commit; `uv sync` (torch from the CUDA 12.6 index on Linux); print GPU, driver, versions |
| `run` | the experiment's commands, under `caffeinate` on the laptop and a remote `timeout` of `max_hours`; progress streams to `runs/.../log.txt` |
| `fetch` | copy the `fetch` files back, record their sha256 |
| `down` | terminate through the API, then poll until Lambda reports it terminated |

A `trap` on exit runs `fetch` (best effort) and `down` whatever happened.

## Determinism and reproducing a run

- CUDA runs set `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; captures print both, and the dtype.
- Before trusting a new kind of run, run it twice and diff the captures. If they differ, the
  post publishes one frozen capture and says which numbers moved, by how much.
- Mac (fp32) and CUDA (bf16) numbers are never mixed in one comparison.

## When something goes wrong

| symptom | cause and fix |
|---|---|
| no capacity for the type | Lambda is out of that GPU; `estimate` shows availability; wait or pick another type, and re-approve the estimate |
| torch says no CUDA | wrong wheel; Linux must take torch from the cu126 index (see `pyproject.toml`) |
| SSH hangs | key not in the agent, or registered from another machine |
| laptop slept or lost network mid-run | the instance keeps billing: run `list` and `down` as soon as you're back; the remote `timeout` stops the work, not the bill |
| run finished but `list` shows an instance | terminate it by hand, then find out why `down` didn't |

## After a run

1. `list` shows nothing.
2. Read the capture; commit it with `run.json` (not the logs with timestamps, not `lambda.env`).
3. Add the run's cost to the experiment's ledger (`runs/ledger.csv` *(planned)*).
4. If the numbers go into a post, the post's recipe quotes the experiment file and the commit.
