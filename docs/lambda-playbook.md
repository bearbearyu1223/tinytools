# Running tinytools experiments on Lambda Cloud

This is the procedure every GPU run in this repo follows, from "I want to run X" to "the
numbers are in `outputs/` and the instance is gone". It exists because a cloud GPU bills by
the hour whether or not anything useful is happening, and because a result is only worth
publishing if someone else can rerun it from what's written down.

> Status: built and tested against a local fake of Lambda's API (`tests/test_lambda_run.py`);
> not yet run against real Lambda. Milestone 0 is its first real run: fix this page where
> reality differs.

The runner is `src/tinytools/lambda_run.py` (standard library only), called through
`scripts/lambda.sh`. It is Python rather than bash so the API key travels in a request
header (a `curl` command line would show it to `ps`), the cleanup can ignore a second Ctrl-C
while it terminates, and the whole flow is tested against a fake API.

## The rules

1. **Nothing launches without an approved estimate.** `run` prints the instance type, the
   live hourly price from Lambda's API, the expected and worst-case hours and cost, and what
   is left of the experiment's cap. It refuses if the worst case (price x `max_hours`) is more
   than what's left. It launches only when the author approves: typing `yes` at the prompt,
   or `--approve USD` with a figure at least the worst case.
2. **Billing stops only when the instance is terminated through the API.** Shutting the machine
   down from inside does not stop the bill. So the runner terminates from the laptop: on
   success, on any error, on Ctrl-C or SIGTERM/SIGHUP, and when the hard time limit is reached.
3. **Every run ends with a check that nothing is left running.** `run` lists the account's
   instances last and prints `STILL UP` for any named `tinytools-*`. If it does, `down` it by
   hand before anything else.
4. **Every run leaves a record.** `runs/<date>-<experiment>/run.json`, and a row in
   `runs/ledger.csv` (below).
5. **Secrets never leave `lambda.env`.** It's gitignored. The runner reads keys from it and never
   prints them; `doctor` shows key names and lengths only, and every line the runner writes
   (terminal, log, `run.json`, ledger) passes through a redactor that replaces any value from
   `lambda.env` and any `Bearer ...` token. The instance never receives the API key.
6. **Numbers in a post come from fetched captures, never from a terminal scrollback.**

## One-time setup

| step | how | what goes wrong |
|---|---|---|
| Lambda account and API key | dashboard, API keys | - |
| `lambda.env` | `cp lambda.env.example lambda.env`, fill in `LAMBDA_API_KEY`, `LAMBDA_SSH_KEY_NAME`, `LAMBDA_SSH_KEY` | the SSH key path must be the private key, not the `.pub` |
| SSH key registered with Lambda | dashboard, SSH keys, from this laptop's public key | a key registered from another machine won't match |
| key loaded | `ssh-add --apple-use-keychain <key>` | a passphrase key not in the agent makes every SSH step fail (the runner never prompts) |
| check | `scripts/lambda.sh doctor` | checks all of the above without launching |

`doctor` checks: `lambda.env` exists; each key is set (name and length shown); ssh,
ssh-keygen, rsync, git and caffeinate are installed; the key path is not a `.pub`; the
private key exists and its public half is readable; it is usable without a prompt (no
passphrase, or loaded in ssh-agent); the API answers with this key; a key named
`LAMBDA_SSH_KEY_NAME` is registered at Lambda and its fingerprint is this laptop's key's;
no `tinytools-*` instance is up. Optional settings: `LAMBDA_REGION` (default: the first
region with capacity), `LAMBDA_API_URL` (default `https://cloud.lambda.ai/api/v1`).

## An experiment is a file

Each experiment is `experiments/<name>.toml`, committed before it runs:

```toml
name = "milestone0"
post = 9
instance_type = "gpu_1x_h100_sxm5"   # checked against the live list at launch
max_hours = 2.5                      # hard limit on the instance's life, launch to termination
expected_hours = 2.0                 # the estimate's expected figure (basis in a comment)
cap_usd = 15                         # refuse to launch past this total for the experiment
commands = [
  "uv run tinytools toolace-convert",
  "uv run tinytools sft-data --seeds 1",
  "uv run tinytools milestone0",
]
fetch = ["outputs/milestone0.txt", "outputs/milestone0-*.jsonl", "outputs/milestone0-setup.json"]
setup_record = "outputs/milestone0-setup.json"   # optional: merged into run.json
```

(The block shows the shape; `experiments/milestone0.toml` is the file that runs.) The file
is the recipe a post prints: same commands, same settings, same commit. The parser
(`lambda_run.parse_experiment`) rejects unknown keys, `expected_hours` above `max_hours`, and
fetch patterns that are absolute, contain `..`, or use characters outside `[A-Za-z0-9_.*?/-]`.
`data/` is not in git, so an experiment's commands rebuild what it needs on the instance.

## A run, step by step

```
scripts/lambda.sh estimate milestone0   # price, hours, cost, remaining cap
scripts/lambda.sh run milestone0        # checks, then up -> setup -> run -> fetch -> down
scripts/lambda.sh list                  # must show nothing flagged "<- tinytools"
```

Before launching, `run` refuses (exit 2, nothing started) if: the working tree has
uncommitted changes to tracked files; the type isn't in Lambda's live list; no region has
capacity; the worst case is past the cap left; the author doesn't approve; or an instance
named `tinytools-*` is already up.

| step | what happens |
|---|---|
| up | launch one instance of the type, named `tinytools-<exp>-<stamp>`; the ledger gets a row at the worst-case cost *before* the launch call; wait until it is active and SSH answers |
| setup | copy the repo at the recorded commit (`git archive`: tracked files only) with rsync; install uv if missing; `uv sync --frozen` (torch from the CUDA 12.6 index on Linux); print GPU, driver and versions; fail if torch can't see the GPU |
| run | the commands, in order, stopping at the first that fails, detached on the instance (`nohup setsid`, so a dropped SSH connection doesn't stop them) under a remote `timeout`; the laptop polls every 15 s, streams the output to the terminal and `runs/<date>-<exp>/log.txt`, and keeps the Mac awake with `caffeinate -i -s` |
| fetch | copy the `fetch` files into `runs/<date>-<exp>/` (same relative paths), record each one's sha256 and size; patterns that matched nothing are listed |
| down | terminate through the API (retried up to 20 times), then poll until Lambda reports it terminated |

`down <id>` and `list` also work on their own, for recovery.

**The time limit.** `max_hours` bounds the instance's life from the launch request to
termination, so the worst-case cost is exactly price x `max_hours`. The commands must stop
10 minutes (`RESERVE_S`) before that, leaving time to fetch and terminate: the remote
`timeout` fires 30 s before that deadline, and the laptop stops waiting at the deadline
whatever the instance is doing.

## How termination is guaranteed

The run is a `try` whose `finally` always fetches (best effort) and then terminates. At the
start of the cleanup, SIGINT, SIGTERM and SIGHUP are ignored until Lambda confirms
termination, so a second Ctrl-C can't cut it short. Each case below is a test against the
fake API (`tests/test_lambda_run.py`).

| what happens | what the runner does | status in run.json |
|---|---|---|
| all commands succeed | fetch, terminate, confirm | `ok` |
| a command exits non-zero | later commands don't run; fetch, terminate | `failed: a command exited N` |
| setup fails, or anything raises | the exception is caught; fetch, terminate | `failed: <error>` |
| Ctrl-C (or SIGTERM / SIGHUP) | stop the remote job, fetch, terminate; further Ctrl-C ignored | `interrupted (...)` |
| the remote `timeout` fires | the job exits 124; fetch, terminate | `time limit` |
| the remote `timeout` doesn't fire | the laptop's deadline passes; stop the job, fetch, terminate | `time limit` |
| the launch call fails or its answer is lost | look the instance up by its unique name and terminate it if it exists | `failed: <error>` |
| the terminate call fails | retried up to 20 times with backoff, then `COULD NOT TERMINATE` printed | `terminated: false` |
| termination isn't confirmed | `STILL UP` printed by the final list; the run is charged its worst case | `terminated: false` |

**Not covered: the laptop itself stopping** (power loss, `kill -9`, the lid closed on
battery, the network gone for good). Then nothing terminates the instance: the remote
`timeout` stops the work, not the bill. The ledger already holds the run at its worst-case
cost, so the cap stays honest. When back: `scripts/lambda.sh list`, then `down <id>`. (Giving
the instance the API key so it can terminate itself is option (b), to decide before post 10's
long runs.)

## The record

`runs/<date>-<exp>/run.json` (rewritten at each stage, so a crash leaves a partial one):
experiment, post, experiment file and its sha256, commit, commands; instance id, name, type,
region and the API's price at launch; `max_hours`, launch and end times, hours, cost and how
it was computed; status, whether termination was confirmed, the commands' exit code;
versions from setup (GPU, driver, CUDA build, torch, transformers, bfcl-eval, python, uv,
OS); seeds, determinism and dtype from the experiment's `setup_record`, and the whole
record; every fetched file's path, sha256 and size, and the patterns that matched nothing.

`runs/ledger.csv`: one row per run (`run, experiment, commit, instance_type,
price_usd_per_hour, launched_at, ended_at, hours, cost_usd, status`). The row is written at
the worst-case cost before the launch call and replaced with the measured cost
(price x hours from launch request to confirmed termination) once termination is confirmed.
A launch that started nothing costs 0. `estimate` and `run` subtract the experiment's rows
from its cap. Lambda's invoice is authoritative; the ledger is the runner's own count.

`runs/*/log.txt` is gitignored (timestamps, progress); `run.json`, the fetched captures and
`ledger.csv` are committed with the post's numbers.

## Determinism and reproducing a run

- CUDA runs set `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; captures print both, and the dtype.
- Before trusting a new kind of run, run it twice and diff the captures. If they differ, the
  post publishes one frozen capture and says which numbers moved, by how much. Milestone 0
  runs one model's greedy dev pass twice in one process and prints which replies differ.
- Mac (fp32) and CUDA (bf16) numbers are never mixed in one comparison.

## Milestone 0's estimate

`experiments/milestone0.toml`'s hours come from this arithmetic: Mac timings and reasoning
about the H100, **not an H100 measurement**. Milestone 0 itself prints seconds per item and
per training step, from which the main run's estimate is redone.

Generation is one item at a time with Hugging Face `generate` (the path every capture so far
used), so its cost is decode steps x seconds per step. At batch 1 (or one group of 8) a
0.5B-3B model's step is bounded by kernel-launch and Python overhead, not FLOPs, so the step
time is taken as 15-30 ms on the H100 for every size (25 ms central). Decode steps per model:

| pass | items | steps per item (basis) | instruct | base |
|---|---|---|---|---|
| our dev tasks, greedy | 200 | instruct ~46 (Mac 0.5B: 41 on test, 46 on dev); base 512 | 9,200 | 102,400 |
| our dev tasks, 8 samples at T=1 | 200 | the longest of 8: instruct ~140 (Mac 0.5B: 3x greedy's time); base 512 | 28,000 | 102,400 |
| BFCL dev slice, greedy | 124 | instruct ~68 (Mac 0.5B, all items); base ~425 (Mac quick) | 8,432 | 52,700 |
| total per model | | | 45,632 | 257,500 |

**Base models don't stop.** A base model's config stops only at `<|endoftext|>`; the scorer
adds `<|im_end|>` (`bfcl.stop_ids`), but `--quick` on the Mac shows Qwen2.5-0.5B base writing
the call's JSON without `<tool_call>` tags and then repeating it until the 512-token cap, on
every item. So a base model costs ~5.6x an instruct model.

| scope | 15 ms/step | 25 ms/step | 30 ms/step |
|---|---|---|---|
| all six models (the toml as written) | 4.1 h | 6.7 h | 8.0 h |
| `--base-when-needed`, an instruct model qualifies | 0.9 h | 1.3 h | 1.5 h |
| `--base-when-needed`, base 1.5B needed | 2.0 h | 3.1 h | 3.7 h |
| `--base-when-needed`, both base models needed | 3.1 h | 4.9 h | 5.8 h |

Each includes the second greedy pass of 1.5B instruct (9,200 steps) and ~18 minutes of
boot, `uv sync`, data preparation, model downloads and loads, the training probe, fetch and
termination. At the 2025 list price of $3.29/h for 1x H100 SXM (the API's live price
decides), all six models cost ~$22 expected and $26.32 worst case at `max_hours = 8`, past the
$15 cap, so `estimate` refuses the file as written until the author changes the scope or
the cap (the toml's comment lists both options).

## When something goes wrong

| symptom | cause and fix |
|---|---|
| no capacity for the type | Lambda is out of that GPU; `estimate` shows availability; wait or pick another type, and re-approve the estimate |
| torch says no CUDA | wrong wheel; Linux must take torch from the cu126 index (see `pyproject.toml`); setup fails and the instance is terminated |
| SSH never answers | key not in the agent, or registered from another machine (`doctor` checks both); the runner waits until its deadline, then terminates |
| laptop slept or lost network mid-run | the instance keeps billing: run `list` and `down` as soon as you're back; the remote `timeout` stops the work, not the bill |
| run finished but `list` shows an instance | terminate it by hand (`down <id>`), then find out why `down` didn't |

## After a run

1. `list` shows nothing flagged.
2. Read the capture; copy what the post uses from `runs/<date>-<exp>/outputs/` to `outputs/`;
   commit it with `run.json` and `runs/ledger.csv` (not the log, not `lambda.env`).
3. If the numbers go into a post, the post's recipe quotes the experiment file and the commit.
