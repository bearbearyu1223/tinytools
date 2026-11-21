# Running tinytools experiments on Lambda Cloud

This is the procedure every GPU run in this repo follows, from "I want to run X" to "the
numbers are in `outputs/` and the instance is gone". It exists because a cloud GPU bills by
the hour whether or not anything useful is happening, and because a result is only worth
publishing if someone else can rerun it from what's written down.

> Status: built and tested against a local fake of Lambda's API (`tests/test_lambda_run.py`,
> `tests/test_lambda_spend.py`), including the real API's behaviour the first version
> missed (spend-safety review, 2026-11-21): terminate answers with Instance objects,
> terminate is refused while a VM boots, 429 rate limits, `unhealthy` and `preempted`
> statuses, a new instance missing from the list for a while, redirects, and answers cut
> off mid-body. Not yet run against real Lambda. Milestone 0 is its first real run: fix
> this page where reality differs.

The runner is `src/tinytools/lambda_run.py` (standard library only), called through
`scripts/lambda.sh`. It is Python rather than bash so the API key travels in a request
header (a `curl` command line would show it to `ps`), the cleanup can ignore a second Ctrl-C
while it terminates, and the whole flow is tested against a fake API.

## The rules

1. **Nothing launches without an approved estimate.** `run` prints the instance type, the
   live hourly price from Lambda's API, the expected and worst-case hours and cost, and what
   is left of the experiment's cap and of the post's cap ($50 per post, summed over all the
   post's experiments). It refuses if the worst case (price x `max_hours`) is more than
   either. It launches only when the author approves: typing `yes` at the prompt, or
   `--approve USD` with a figure at least the worst case.
2. **Billing stops only when the instance is terminated through the API.** Shutting the machine
   down from inside does not stop the bill. So the runner terminates from the laptop: on
   success, on any error, on Ctrl-C or SIGTERM/SIGHUP, and when the hard time limit is reached,
   and it keeps trying until Lambda confirms.
3. **Every run ends with a check that nothing is left running.** `run` lists the account's
   instances last. An instance of this run that is still up (a lost launch that showed up
   late) is terminated there; any other `tinytools-*` is printed as `STILL UP`. If termination
   is not confirmed, or another `tinytools-*` instance is up, `run` exits **3**: `down` the
   instance by hand before anything else.
4. **Every run leaves a record.** `runs/<date>-<experiment>/run.json`, and a row in
   `runs/ledger.csv` (below).
5. **Secrets never leave `lambda.env`.** It's gitignored. The runner reads keys from it and never
   prints them; `doctor` shows key names and lengths only, and every line the runner writes
   (terminal, log, `run.json`, ledger) passes through a redactor that replaces the API key,
   any `Bearer ...` token, and the SSH key's path (as written, `~`-expanded and absolute).
   Nothing else is redacted: the SSH key's *name* at Lambda is not a secret, and scrubbing a
   short name like `h100` would garble every line containing it. The instance never receives
   the API key.
6. **Numbers in a post come from fetched captures, never from a terminal scrollback.**

Exit codes of `run`: **0** done and terminated; **1** the run failed (or succeeded but a
`fetch` pattern matched nothing it wrote), its instance confirmed terminated; **2** refused before launching (nothing started); **3** termination not confirmed,
or another `tinytools-*` instance still up (an instance may be billing). `down` exits 3 too
when it can't confirm.

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
passphrase, or loaded in ssh-agent); the API answers with this key, and **at which base URL**;
a key named `LAMBDA_SSH_KEY_NAME` is registered at Lambda and its fingerprint is this
laptop's key's; no `tinytools-*` instance is up. Optional settings: `LAMBDA_REGION` (default:
the first region with capacity), `LAMBDA_API_URL` (default `https://cloud.lambda.com/api/v1`,
the base in Lambda's API reference at docs.lambda.com).

**Redirects are never followed.** urllib would send the `Authorization` header on to the new
host and turn a POST (launch, terminate) into a body-less GET. A 3xx is an error that names
its `Location`; if that is Lambda's new address, set `LAMBDA_API_URL` to it. `doctor` shows it.

**Rate limits.** Lambda allows 1 request per second and 1 launch per 12 seconds. Requests to
Lambda are spaced 1 s apart, and a 429 is retried after 1, 2, 4, 8, 15, 15 s (or its
`Retry-After`). Errors keep Lambda's `request_id`, for a support ticket.

## An experiment is a file

Each experiment is `experiments/<name>.toml`, committed before it runs:

```toml
name = "milestone0"
post = 9                             # required: the $50 cap is summed per post
instance_type = "gpu_1x_a100_sxm4"   # checked against the live list at launch
max_hours = 4.0                      # hard limit on the instance's life, launch to termination
expected_hours = 1.2                 # the estimate's expected figure (basis in a comment)
cap_usd = 10                         # refuse to launch past this total for the experiment
commands = [
  "uv run tinytools toolace-convert",
  "uv run tinytools sft-data --seeds 1",
  "uv run tinytools milestone0 --base-when-needed",
]
fetch = ["outputs/milestone0.txt", "outputs/milestone0-setup.json",
         "outputs/milestone0-Qwen2.5-*-replies.jsonl"]
setup_record = "outputs/milestone0-setup.json"   # optional: merged into run.json
```

(The block shows the shape; `experiments/milestone0.toml` is the file that runs.) The file
is the recipe a post prints: same commands, same settings, same commit. The parser
(`lambda_run.parse_experiment`) rejects unknown keys, a missing `post`, `expected_hours` above
`max_hours`, a name that makes the instance name longer than Lambda's 64 characters, and
fetch patterns that are absolute, contain `..`, or use characters outside `[A-Za-z0-9_.*?/-]`.
Keep fetch patterns to the files the commands write: a pattern like `milestone0-*.jsonl` also
matches the committed `milestone0-quick-*` Mac captures. `data/` is not in git, so an
experiment's commands rebuild what it needs on the instance.

## A run, step by step

```
scripts/lambda.sh estimate milestone0   # every type of its GPU, price, hours, cost, caps
scripts/lambda.sh run milestone0        # checks, then up -> setup -> run -> fetch -> down
scripts/lambda.sh list                  # must show nothing flagged "<- tinytools"
```

Before launching, `run` refuses (exit 2, nothing started) if: the working tree has
uncommitted changes to tracked files; the type isn't in Lambda's live list; no region has
capacity; the worst case is past the experiment's or the post's cap left; the author doesn't
approve; or an instance named `tinytools-*` is already up.

| step | what happens |
|---|---|
| up | launch one instance of the type, named `tinytools-<exp>-<stamp>` and tagged `tinytools-run=<run>-<random>` (if Lambda refuses the tag, a 4xx with error code `global/invalid-parameters` whose message names the `tags` field, launched again without it; any other refusal is final); the ledger gets a row at the worst-case cost *before* the launch call; wait until it is active (a 404 just after the launch means "not listed yet"; 20 minutes at most), then until SSH answers (10 minutes at most) |
| setup | copy the repo at the recorded commit (`git archive`: tracked files only) with rsync; install uv if missing; `uv sync --frozen` (torch from the CUDA 12.6 index on Linux); print GPU, driver and versions; fail if torch can't see the GPU. Copying and the setup script together get 25 minutes (`SETUP_TIMEOUT_S`); past that the run fails and the instance is terminated |
| run | the commands, in order, stopping at the first that fails, detached on the instance (`nohup setsid`, so a dropped SSH connection doesn't stop them) under a remote `timeout`; `job.sh` touches `.run/start` first and exports `TINYTOOLS_DEADLINE_EPOCH`, the Unix time the remote `timeout` fires, for commands that can stop themselves in time (`milestone0 --deadline-epoch`). The laptop polls every 15 s (each poll also checks that `job.sh` is alive: one that died without an exit code ends the run; polls that fail for 10 minutes in a row, SSH gone, end the run), asks Lambda for the instance's status every 5 minutes (`unhealthy`, `preempted`, `terminated` or gone ends the run), streams the output to the terminal and `runs/<date>-<exp>/log.txt`, and keeps the Mac awake with `caffeinate -i -s` (in its own session, so Ctrl-C doesn't end it) |
| fetch | copy the `fetch` files **written during the run** (newer than `.run/start`) into `runs/<date>-<exp>/` (same relative paths), record each one's sha256 and size; patterns that matched nothing written are listed, and make `run` exit 1 even when every command succeeded. A committed file the commands didn't rewrite is not fetched. Fetching takes at most 7 minutes, and never the last 5 minutes before the hard limit (kept for terminating) |
| down | terminate through the API until Lambda reports the instance `terminated` or `preempted`: retried after every failure, not sent while Lambda says it can't (`actions.terminate.available` false, `vm-has-not-launched`, while it boots), re-sent every 60 s after a call that was accepted but didn't take. An id Lambda named in terminate's answer counts as seen, so a later 404 for it means it is gone. Up to 2 hours; if still not confirmed, the final check lists the account and, if this run's instance is still up, makes a second attempt of up to 2 hours (so up to 4 hours in all: **the laptop must stay awake and online throughout**); then exit 3 |

`down <id>` and `list` also work on their own, for recovery.

**The time limit.** `max_hours` bounds the instance's life from the launch request to
termination, so the worst-case cost is price x `max_hours` as long as Lambda accepts the
terminate. The commands must stop 10 minutes (`RESERVE_S`) before that, leaving time to stop,
fetch and terminate: the remote `timeout` fires 30 s before that deadline, and the laptop
stops waiting at the deadline whatever the instance is doing.

**Stopping the job.** `job.sh` runs in a session of its own (`setsid`), and GNU `timeout`
moves itself and the commands into a new process group, so signalling `job.sh`'s group
misses them. `stop` signals the whole session (`pkill -TERM -s`), waits up to 20 s, then
`pkill -KILL -s`, before anything is fetched. ssh is run with `-n` and no stdin except when a
script is sent, so it never reads the terminal.

## How termination is guaranteed

The run is a `try` whose `finally` always stops the job, fetches (best effort), terminates,
records and lists. Each step is guarded: whatever one raises, the next still runs. Every
API failure, an `http.client` protocol error or a cut-off answer included, is an `ApiError`.
At the start of the cleanup, SIGINT, SIGTERM and SIGHUP are ignored until Lambda confirms
termination, so a second Ctrl-C can't cut it short. Each case below is a test against the
fake API (`tests/test_lambda_run.py`, `tests/test_lambda_spend.py`).

| what happens | what the runner does | status in run.json | exit |
|---|---|---|---|
| all commands succeed | fetch, terminate, confirm | `ok` | 0 |
| a command exits non-zero | later commands don't run; fetch, terminate | `failed: a command exited N` | 1 |
| setup fails, or anything raises | the exception is caught; fetch, terminate | `failed: <error>` | 1 |
| Ctrl-C (or SIGTERM / SIGHUP) | stop the remote job, fetch, terminate; further Ctrl-C ignored | `interrupted (...)` | 1 |
| the remote `timeout` fires | the job exits 124 (137 if `--kill-after` had to KILL); fetch, terminate | `time limit` | 1 |
| a command is SIGKILLed before the limit (the OOM killer) | exit 137 before the remote seconds have passed; fetch, terminate | `killed (SIGKILL, e.g. out of memory)` | 1 |
| the remote `timeout` doesn't fire | the laptop's deadline passes; stop the job, fetch, terminate | `time limit` | 1 |
| `job.sh` dies with no exit code | the next poll notices; stop what's left, fetch, terminate | `failed: ... ended without an exit code` | 1 |
| the instance turns `unhealthy`, `preempted` or is gone | the status check notices; terminate (a preempted one is already ended) | `failed: ... instance status is ...` | 1 |
| SSH never answers | after 10 minutes, terminate | `failed: ... SSH did not answer ...` | 1 |
| SSH stops answering mid-job | polls fail 10 minutes in a row: stop (best effort), fetch, terminate | `failed: ... stopped answering SSH ...` | 1 |
| setup takes over 25 minutes | the copy or the setup script is cut off; terminate | `failed: ... setup did not finish within 25 minutes ...` | 1 |
| the commands succeed but a `fetch` pattern matched nothing they wrote | fetch the rest, terminate; the patterns are named | `ok` (`not_found` lists them) | 1 |
| terminate called while it boots | wait until Lambda allows it, then terminate | as above | 1 |
| the launch is refused (4xx) | nothing started; no lookup; cost 0 | `failed: <error>` | 1 |
| the launch call fails (5xx) or its answer is lost, or Ctrl-C / SIGTERM / SIGHUP arrives during it | the launch counts as uncertain from before the request until a definite answer; list every ~10 s for 2 minutes, looking for the run's name or tag; terminate it if found; the final check looks once more | `failed: <error>` or `interrupted (...)` | 1; 3 if no list ever answered (charged the worst case) |
| the terminate call fails | retried until confirmed (a loud line after 20 failures) | - | - |
| termination isn't confirmed after 2 hours | `NOT CONFIRMED` printed; the final check lists, and if the instance is still up tries again for up to 2 more hours (keep the laptop awake); then the run is charged its worst case | `terminated: false` | 3 |

**Not covered: the laptop itself stopping** (power loss, `kill -9`, the lid closed on
battery, the network gone for good). Then nothing terminates the instance: the remote
`timeout` stops the work, not the bill. The ledger already holds the run at its worst-case
cost, so the caps stay honest. When back: `scripts/lambda.sh list`, then `down <id>`. (Giving
the instance the API key so it can terminate itself is option (b), to decide before post 10's
long runs.)

## The record

`runs/<date>-<exp>/run.json` (rewritten at each stage, so a crash leaves a partial one):
experiment, post, experiment file and its sha256, commit, commands; instance id, name, tag,
type, region and the API's price at launch; `max_hours`, launch and end times, hours, cost
and how it was computed; status, whether termination was confirmed, the commands' exit code;
versions from setup (GPU, driver, CUDA build, torch, transformers, bfcl-eval, python, uv,
OS); seeds, determinism and dtype from the experiment's `setup_record`, and the whole
record; every fetched file's path, sha256 and size, and the patterns that matched nothing
written during the run.

`runs/ledger.csv`: one row per run (`run, experiment, post, commit, instance_type,
price_usd_per_hour, launched_at, ended_at, hours, cost_usd, status`; the status redacted
like every other line). The row is written at the worst-case cost before the launch call and
replaced with the measured cost (price x hours from launch request to confirmed termination)
once termination is confirmed. A launch that started nothing costs 0. `estimate` and `run`
subtract the experiment's rows from its cap and the post's rows from the $50 post cap.
Lambda's invoice is authoritative; the ledger is the runner's own count.

`runs/*/log.txt` is gitignored (timestamps, progress); `run.json`, the fetched captures and
`ledger.csv` are committed with the post's numbers.

## Determinism and reproducing a run

- CUDA runs set `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; captures print both, and the dtype.
- Generation is the batched engine (`tinytools.engine`): the batch size is printed in
  Setup and recorded in every replies row, and items are sorted by prompt length, so a
  run's batches are fixed by the items and the batch size. Rounding in a forward pass
  can depend on the batch's shape, so a capture is reproduced at the same batch size.
  Sampled replies draw from per-reply generators (seed, item id, sample number), so
  their random numbers don't depend on the batch at all (`engine-check` measures how
  often a reply changes with the batch size anyway).
- Before trusting a new kind of run, run it twice and diff the captures. If they differ, the
  post publishes one frozen capture and says which numbers moved, by how much. Milestone 0
  runs one model's greedy dev pass twice in one process and prints which replies differ.
- Mac (fp32) and CUDA (bf16) numbers are never mixed in one comparison.

## Milestone 0's estimate

`experiments/milestone0.toml`'s hours come from this arithmetic: a Mac measurement of the
batched engine and reasoning about the A100, **not an A100 measurement**. Milestone 0
itself prints seconds per item and per training step, and records each model's batched
decode steps and seconds, from which the main run's estimate is redone.

**The instance** (the author, 2026-11-21): an A100, not an H100 (an H100 is more than
0.5-3B models need at these batch sizes); 80 GB if Lambda has one, for the 3B
full-fine-tune probe. `gpu_1x_a100_sxm4` is a 1x A100 SXM4 **40 GB** (tinyvlm ran on it
and torch reported A100-SXM4-40GB); no 1x A100 80 GB type name is known here without the
live API. `estimate` therefore lists every A100 type in Lambda's live list (memory from
Lambda's description, price, regions with capacity) above its usual lines: if a 1x A100
80 GB type has capacity, put its name in the experiment file before `run`. On 40 GB the
3B full fine-tune may not fit (bf16 weights, gradients and AdamW states are ~24 GB before
activations); the probe then prints "out of memory" for that row and carries on.

**Generation is batched** (`tinytools.engine`): 64 sequences per forward pass on CUDA, an
item's 8 samples in one batch (8 tasks x 8). A batch runs until its longest row stops, so
its cost is decode steps (the longest reply) x seconds per step, and short replies wait.
Batches per model:

| pass | items | batches of 64 rows |
|---|---|---|
| our dev tasks, greedy | 200 | 4 |
| our dev tasks, 8 samples at T=1 | 200 x 8 | 25 (8 tasks each) |
| BFCL dev slice, greedy | 40 / 20 / 20 / 20 / 24 | 5 (one per category) |
| the repeat model's second greedy pass (1.5B instruct) | 200 | 4 |
| total | | 34 (38 for 1.5B instruct) |

**Steps per batch: every batch is assumed to run to the 512-token cap.** On the Mac
(`outputs/engine-check.txt`, 0.5B instruct, batches of 8) a batch's longest reply set its
length and only 34-48% of row-steps produced a counted token; with 64 rows per batch the
longest is often a reply that runs to the cap (T=1 samples, and BFCL replies, do), and a
base model's replies always do. So the cap is the bound, not a typical case: an instruct
model will usually take less. A base model costs the same as an instruct one here
(every batch runs to the cap either way), where one at a time it cost ~5.6x.

**Step time: 35 ms per batched decode step, assumed, 70 ms as the pessimistic case.**
One at a time, a 0.5-3B model's decode step is bound by kernel launches and Python
overhead (the H100 estimate assumed 15-30 ms). At 64 rows a step also reads the weights
(3-6 GB in bf16 for 1.5B-3B, ~2-4 ms at the A100's ~1.5 TB/s) and the KV cache (64 rows
x ~2k tokens x ~29-37 KB per token for 1.5B-3B, ~2.5-3 ms), and the sampler adds an
integer cumulative sum over 64 x 152k probabilities and one host sync. The Mac shows why
not to take a batch's step as free: there a padded batch's step grows with every row
(16 rows: 53 ms unpadded, 130 ms padded), so batching sped generation up only 1.2x
(`engine-check`); on the A100 the per-row cost is a small share of the step. Prefill (64
prompts of ~0.4-2k tokens) is ~1-2 s a batch, included in the ~1 minute per model
allowed for loading.

| scope (`--base-when-needed`) | decode steps | 35 ms/step | 70 ms/step |
|---|---|---|---|
| an instruct model qualifies (1.5B, 3B and 0.5B instruct scored) | 54,272 | 1.0 h | 1.5 h |
| base 1.5B needed | 71,680 | 1.2 h | 1.9 h |
| both base models needed | 89,088 | 1.4 h | 2.2 h |

Each includes ~24 minutes of boot, `uv sync`, data preparation, model downloads and
loads, the training-step probe, fetch and termination, and ~1 minute per model of loading
and prefill. `expected_hours = 1.2` is the middle row at 35 ms/step; even the worst row
at 70 ms/step fits `max_hours = 4`. The worst case is 4 h x the live price; with
`cap_usd = 10`, `estimate` refuses above $2.50/h ($7.96 at $1.99/h).

The deadline logic stays as a guard. Before each model `milestone0` estimates its time
conservatively from what was measured so far: a model of the same kind at the slowest
measured seconds per item-pass; the first base model as its batches (`batch_count`) each
running to the 512-token cap at the slowest measured seconds per greedy decode step, plus
loading; x 1.5, and 2 minutes kept for writing. A model that can't finish is not started:
the rule section is printed from what was measured, with the verdict "incomplete" if the
rule reaches the unscored model, and `milestone0` exits 1 (so `run` exits 1, with every
file fetched). SIGTERM (the time limit, or the run stopped) does the same with whatever
was measured, and exits 143. The training probe is not started with under 10 minutes
left. The next step is then the author's.

The earlier H100 estimate (one item at a time, 15-30 ms per step, 3.1 h expected, $20 cap)
is replaced by this one.

## The model rule Milestone 0 prints

A candidate qualifies when all four hold (`tinytools.milestone0`; the author's decision of
2026-11-21 after the review found the three-part rule could pick a model with no right call):

- (a) our dev tasks, greedy, right by execution match: at most 70% (all 200 tasks);
- (b) BFCL dev slice, call categories, right: at most 80%;
- (c) mixed groups (some of 8 samples at T=1 right, some wrong): at least 10% of the dev
  tasks that need a call (choose+fill and tricky arguments: 160 of the 200);
- (d) greedy right by execution match: at least 20% of the dev tasks that need a call.

(c) and (d) leave out the 40 "no tool fits" tasks, where declining is right, so a model that
never makes a right call can't qualify. The capture prints every criterion for every model
with its count, denominator, threshold and met / not met, and the verdict and its sentence
are built from those numbers.

## When something goes wrong

| symptom | cause and fix |
|---|---|
| no capacity for the type | Lambda is out of that GPU; `estimate` shows availability; wait or pick another type, and re-approve the estimate |
| torch says no CUDA | wrong wheel; Linux must take torch from the cu126 index (see `pyproject.toml`); setup fails and the instance is terminated |
| SSH never answers | key not in the agent, or registered from another machine (`doctor` checks both); the runner gives up after 10 minutes and terminates |
| `doctor`: "redirects to ..." | Lambda's API moved; if the `Location` is Lambda's, set `LAMBDA_API_URL` to its base |
| exit 3 / `NOT CONFIRMED` | `scripts/lambda.sh list`, `down <id>`; if that fails too, the dashboard, then find out why |
| laptop slept or lost network mid-run | the instance keeps billing: run `list` and `down` as soon as you're back; the remote `timeout` stops the work, not the bill |
| run finished but `list` shows an instance | terminate it by hand (`down <id>`), then find out why `down` didn't |

## After a run

1. `list` shows nothing flagged, and `run` exited 0 or 1 (never 3).
2. Read the capture; copy what the post uses from `runs/<date>-<exp>/outputs/` to `outputs/`;
   commit it with `run.json` and `runs/ledger.csv` (not the log, not `lambda.env`).
3. If the numbers go into a post, the post's recipe quotes the experiment file and the commit.
