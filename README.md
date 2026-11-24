# tinytools

Companion project for posts 9-12 of the LLM Architecture Refresh series: teaching a
small model to call tools (SFT on ToolACE, scored on BFCL), then RL with execution
rewards. Local only until post 9 publishes. So far: data, evaluation, the SFT training sets, a
batched generation engine, Milestone 0 (below) with its Lambda runner, and post 9's SFT trainer,
merged-adapter scoring and report (`sft-*`, "Post 9's main run" below).

```
uv sync
uv run tinytools info               # device, versions, pinned bfcl-eval
uv run tinytools toolace-stats      # count ToolACE; check the conversion and masking
uv run tinytools toolace-convert    # write data/toolace-en-qwen.jsonl
uv run tinytools bfcl-score --model Qwen/Qwen2.5-0.5B-Instruct \
    --categories simple,multiple,parallel,parallel_multiple,irrelevance --limit 20
uv run tinytools bfcl-score --split dev --samples 8 --temperature 1.0
uv run tinytools bfcl-rescore --replies outputs/bfcl-<...>-replies.jsonl  # no model
uv run tinytools tasks              # our tool tasks -> data/tasks/, counts (r2a)
uv run tinytools check-demo         # three calls checked both ways (r2b)
uv run tinytools score --split test --limit 30              # greedy, 30 per kind
uv run tinytools score --split test --limit 10 --samples 8 --temperature 1.0
uv run tinytools rescore --replies outputs/score-<...>-replies.jsonl      # no model
uv run tinytools sft-data --seeds 1,2,3  # SFT sets -> data/sft/ (r4a, r4b)
uv run tinytools milestone0 --quick      # Milestone 0 end to end, 0.5B, a few items (Mac check)
uv run tinytools milestone0 --base-when-needed  # instruct models, base only if the rule needs it (A100)
uv run tinytools sft-check --quick      # template, sha256, lengths, memory, merge checks (0.5B)
uv run tinytools sft-score --untrained --quick
uv run tinytools sft-score --untrained --noise-floor --quick  # r6a's baseline (seed 1)
uv run tinytools sft-pair --seed 1 --quick   # train real, score, train control, score
uv run tinytools sft-train --arm real --seed 1 --repeat --quick  # determinism check
uv run tinytools sft-report --quick     # r4c-r6a from outputs/main-quick/ (no model)
uv run tinytools sft-check --profile base --quick  # the base arm (plan 3f): every sft-* command
                                         # takes --profile base (0.5B base in quick mode)
scripts/lambda.sh doctor                 # Lambda settings, SSH key, API base URL; launches nothing
scripts/lambda.sh estimate milestone0    # price x hours against the experiment's and the post's cap
scripts/lambda.sh run milestone0 --approve 8  # up -> setup -> run -> fetch -> down (docs/lambda-playbook.md)
scripts/lambda.sh list                   # after every run: no tinytools instance may be left
uv run pytest -q && uvx ruff check --select F,E9,B
```

Every command prints a 74-column report on stdout (progress on stderr only) and
writes the same report to `outputs/`. `bfcl-score` also writes every reply, its
outcome and BFCL's error type to `outputs/bfcl-<model>-<mode>-limit<N>-replies.jsonl`.
`data/` (converted ToolACE) and `bfcl_project/` (bfcl-eval's scratch directories)
are gitignored.

GPU runs go through `scripts/lambda.sh` and `experiments/<name>.toml`, as
`docs/lambda-playbook.md` describes: an approved estimate within the experiment's cap and
the post's ($50, summed over its experiments), a hard time limit, termination through
Lambda's API whatever happens (retried until Lambda confirms; exit code 3 if it never
does), only the files the run wrote fetched, and a record in `runs/<date>-<exp>/run.json`
and `runs/ledger.csv`. Settings and keys live in `lambda.env` (gitignored; see
`lambda.env.example`); the API base defaults to `https://cloud.lambda.com/api/v1`
(`LAMBDA_API_URL` overrides it) and redirects are never followed.

Milestone 0's model rule: a candidate qualifies when (a) greedy right on our dev tasks
<= 70%, (b) BFCL dev call categories right <= 80%, (c) mixed groups of 8 at T=1 >= 10% of
the dev tasks that need a call, and (d) greedy right on the dev tasks that need a call
>= 20%. The capture prints each with its count, denominator, threshold and met / not.

Environment: `TINYTOOLS_DEVICE=cpu|mps|cuda` overrides the device (default cuda >
mps > cpu); `TINYTOOLS_QUIET=1` silences progress.

## Milestone 0: choosing the model

Before anything is trained, `milestone0` scores Qwen2.5 1.5B and 3B (instruct; base models only if
the rule reaches them; 0.5B as a comparison row) on the **dev** split, which no published number
uses, and applies a rule fixed in advance. It picks the smaller candidate that still leaves room
to learn: (a) greedy right on at most 70% of our dev tasks, (b) BFCL dev call items right at most
80%, (c) mixed groups (some of 8 samples right, some wrong) on at least 10% of the dev tasks that
need a call, (d) at least 20% of those tasks right. It also runs one model twice to check
reproducibility, and times a few LoRA and full fine-tuning steps per size. Run on Lambda:
`experiments/milestone0.toml` (A100 40 GB, at most 4 h, cap $10); output `outputs/milestone0.txt`.

The verdict is one of: the model that qualifies, none (no candidate meets all four), or
`incomplete` (time ran out before a model the rule needed was scored; exit 1).

### Running it

**On the Mac, a check only.** `uv run tinytools milestone0 --quick` runs every step on 0.5B with a
few items in about 4 minutes (`outputs/milestone0-quick.txt`). The full command runs on a Mac too,
but takes about 9 hours in fp32, the 3B full fine-tuning probe runs out of memory, and its timings
say nothing about the GPU; the published verdict comes from the A100.

**On Lambda.** Once (see `docs/lambda-playbook.md`, "One-time setup"): `lambda.env` filled in, the
laptop's SSH key registered with Lambda, and, if the key has a passphrase, loaded into the agent:

```
ssh-add --apple-use-keychain ~/.ssh/id_ed25519   # the key LAMBDA_SSH_KEY points to
ssh-add -l                                       # one key listed
```

Then, from this folder, with the working tree committed (`run` refuses uncommitted changes, so
the run's commit is the code that ran):

```
scripts/lambda.sh doctor                        # every setting checked; launches nothing
scripts/lambda.sh estimate milestone0           # live A100 price, expected and worst-case cost
scripts/lambda.sh run milestone0 --approve 8    # approve up to $8; refused if the worst case is more
scripts/lambda.sh list                          # afterwards: no "<- tinytools" instance left
```

`run` launches one instance, copies the repo at the current commit, installs with `uv sync`,
then runs the experiment's commands in order: `toolace-convert` and `sft-data --seeds 1` (the
data isn't in git), `engine-check --quick` (the batched engine against the one-at-a-time one, on
CUDA bf16), and `milestone0 --base-when-needed`. Expected about 1.2 h (1.0-2.2 h depending on
which models the rule reaches and the A100's real step time).

**Watching.** `run` streams the output in its own terminal; leave that one open. From a second
terminal, `tail -f runs/*-milestone0/log.txt` shows the same stream: the laptop pulls the
instance's log every 15 s, so lines arrive in chunks. `run` prints `active at <ip>`; with it,
`ssh ubuntu@<ip> 'tail -f tinytools/.run/log.txt'` is the live log and `ssh ubuntu@<ip>
nvidia-smi` the GPU. What appears, in order: the runner's steps (launching, active, SSH, copy,
setup with the GPU and versions); the data commands; the engine check; then for each model
`pass N: <model>: <set>, <n> items` with progress at each quarter of a pass (a pass can be quiet
for minutes), the training-step probe (`training run N: <size> LoRA r16 | full`, loss and
s/step), each model's tables as it finishes, and at the end "The rule (fixed before these
numbers)" with each condition's count, threshold and met / not met, and the verdict.

**Stopping.** When the commands finish, fail, hit the time limit or are interrupted (Ctrl-C in
the `run` terminal), the runner fetches what the run wrote and terminates the instance through
Lambda's API, retrying until Lambda confirms: the last line is `terminated: Lambda confirms;
billing has stopped`. Termination is sent from the laptop, so keep it plugged in, awake and
online, and the `run` terminal open, until then: if the laptop stops, nothing terminates the
instance (shutting down its OS doesn't stop the bill). Exit codes: 0 done; 1 something failed
or a file wasn't fetched (instance terminated); 2 refused before launching (nothing started);
3 termination not confirmed, an instance may be billing: `scripts/lambda.sh list`, then
`scripts/lambda.sh down <id>`, or the Lambda dashboard. Lines starting `EXIT 3`, `STILL UP` or
`LIST FAILED` mean the same.

**Afterwards.** Everything lands in `runs/<date>-milestone0/`: `log.txt`, `run.json` (instance,
price, hours, cost, commit, versions, status, whether termination was confirmed), and the fetched
files under `outputs/` (`milestone0.txt` with the verdict, `milestone0-setup.json`, the replies
files, `engine-check-quick.txt`, and the data captures). `runs/ledger.csv` gets one row per run.

## Versions

| package | pin | why |
|---|---|---|
| torch | 2.13.0 | same as tinyvlm (post 7); Linux takes the CUDA 12.6 build (pyproject explains) |
| transformers | 5.14.1 | same as tinyvlm |
| bfcl-eval | **2025.12.17** | the BFCL leaderboard page (last updated 2026-04-12, read 2026-11-20) says: "Models are evaluated using commit f7cf735 ... To reproduce the results, please either checkout our codebase at this checkpoint, or install the PyPI package `pip install bfcl-eval==2025.12.17`." Latest on PyPI is 2026.3.23; we pin the leaderboard's. The package is `bfcl-eval`; `bfcl` on PyPI is unrelated. |
| numpy | 1.26.4 | forced by bfcl-eval (`numpy==1.26.4`); torch 2.13 and transformers 5.14.1 accept it |
| soundfile | any | bfcl-eval loads every model handler; one imports qwen-agent 0.0.34, which imports soundfile without declaring it |

bfcl-eval pulls in about 140 packages (API clients for every provider, faiss,
sentence-transformers, tree-sitter). None conflicted with the torch/transformers pins.

**vLLM (later, Linux only).** bfcl-eval's `oss-eval-vllm` extra pins `vllm==0.8.5`,
which pins its own torch. When the Lambda harness is built, add our own extra in
`pyproject.toml` (a commented placeholder is there) with a `sys_platform == 'linux'`
marker, and check its torch against 2.13.0 first.

## ToolACE conversion

Source: HF `Team-ACE/ToolACE` (Apache-2.0), `data.json` at revision `6bda777c88d2`
(pinned in `toolace.py`). Each row is `{"system", "conversations": [{"from", "value"}]}`.

* **Language.** A row is Chinese if the first 200 characters of its system prompt
  contain a CJK ideograph. Those rows are dropped.
* **Tool list.** The JSON list after "...in JSON format that you can invoke:". Rows
  whose tools are written another way (ToolACE's format-variation rows, which also
  use other call syntaxes) are counted and dropped; `toolace-stats` counts them by
  format, detected from the list's first characters (`toolace.tool_list_format`):
  YAML-like lines, a Markdown list, a LaTeX tabular, an HTML table, JSON keyed
  `tool_name` / `definition` instead of `name` / `description`, XML-like tags, no
  tool list at all, and "other" (0 in this revision).
* **Calls.** `[Name(arg=value, ...), ...]`, parsed by `calls.py`. Names may contain
  spaces, slashes, dots, parentheses and apostrophes (`Daily Forecast (10 days)`),
  so names are matched against the tool list, longest first.
* **Output format.** Qwen2.5's own template: tools passed as
  `{"type": "function", "function": <ToolACE tool, unchanged>}`; calls as assistant
  `tool_calls`; tool results as `tool` messages (verbatim).
* **System prompt.** ToolACE's is fixed boilerplate, the tool list, the bracket-syntax
  instruction, and in about half the rows one context line ("The current time is ..."
  or "Today is ..."). We keep only that line, after Qwen's default system sentence, so
  a training prompt has the shape a BFCL prompt has (Qwen's sentence, then the tools
  block) and still carries the date relative requests need. Rows without one get the
  template's default. The boilerplate's "if no function fits / arguments are missing,
  point it out" instruction is dropped; the declining turns themselves stay.
* **Masking.** Each record carries the rendered `text` and `train_spans`: the
  character span of every assistant turn, from after `<|im_start|>assistant\n`
  through its `<|im_end|>`. Everything else (system, user, tool) is masked.
  `toolace-stats` tokenizes every row and reports any token straddling a span edge.

## BFCL scoring

`bfcl-score` uses bfcl-eval's test items, reference answers, Qwen `<tool_call>`
decoder (`QwenFCHandler.decode_ast`) and per-item verdict functions (the ones
`bfcl evaluate` runs, which call `ast_checker`). Ours: generation with HF
transformers on the local device, the prompt from the model's own chat template, and
the outcome labels (`classify.py`, one place; `score` uses the same). Tasks that
need a call (BFCL's AST categories; our choose+fill and tricky arguments):

| outcome | meaning |
|---|---|
| no call | no `<tool_call>` tag |
| unparseable | a tag, but some tagged block doesn't decode to a call |
| wrong function | the called function names differ from the reference's as a set (one missing, or one the reference doesn't call) |
| wrong number of calls | the same set of names, but too many or too few calls (one `spotify.play` where the reference has two) |
| wrong arguments | the right calls; the checker rejects the arguments |
| right | the checker accepts it |

Tasks where no tool fits (BFCL irrelevance; our "no tool fits"):

| outcome | meaning |
|---|---|
| declined | no `<tool_call>` tag (right) |
| called anyway | any `<tool_call>` tag, whether or not it decodes |

The reports print the two groups as separate tables. "right" equals BFCL's
accuracy on the AST categories. On irrelevance BFCL also passes a reply whose
`<tool_call>` doesn't decode (here it called anyway); the report prints BFCL's
count beside ours. A control prints with every run: BFCL's own reference answer
written in Qwen's format (shows the harness can give full marks). Under each outcome
table, the model's replies that hit the token cap (max new tokens, in Setup) are
counted per category with the outcome they were given: a reply cut off before any
`<tool_call>` tag counts as declined on irrelevance, so the report prints "declined,
of which hit token cap".

**Splits.** `--split eval` (the default) is every item minus a seeded dev slice
(`bfcl.dev_slice(category, seed=0)`: round(10%) of each category, drawn with
`Random("0:bfcl-dev:<category>")`), the set published numbers come from; `--split
dev` is the slice alone (Milestone 0); `--split all` is both, and its Setup prints a
warning. `--limit N` takes the first N of the split in bfcl's order (default: all).
The capture's name carries the split (`bfcl-<model>-eval-greedy-limitall.txt`),
except `all`, which keeps the name captures had before splits existed.

**Setup** prints the dtype (`device.preferred_dtype`: float32 on mps and cpu,
bfloat16 on a bf16-capable GPU, so a Mac capture and a Lambda capture can differ)
and the determinism flags (`device.set_determinism`, the one place they are set:
on CUDA `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; a no-op on mps and cpu). Every reply row records
`dtype` and `device`, so `rescore` and `bfcl-rescore` print them too (for files
written before that, from the device line of the capture they came with).
`--samples 1` with `--temperature > 0` is refused (it would sample but be labelled
greedy). Shares print rounded half up (1 of 400 is 0.3%); "per 1k" prints one
decimal.

**`bfcl-rescore --replies <file>`** re-judges a replies file with bfcl's checker
(no model) and rewrites the capture `bfcl-score` would have written; its split,
mode and limit come from the file name. Timings come from the replies, or are
carried over from the earlier capture. `--split eval` on an all-split file keeps
the eval items only and writes `bfcl-<model>-eval-<mode>-limit<N>.txt` and its
`-replies.jsonl` (the names `bfcl-score --split eval` writes), leaving the all-split
file as it was; the figures read that eval capture by default.

Decoding is set explicitly: greedy is argmax (Qwen's `generation_config.json` would
otherwise add repetition penalty 1.1, top-p 0.8, top-k 20), and `--samples N
--temperature T` samples the full softmax.

**Generation: one batched engine** (`engine.py`; `score`, `bfcl-score` and
`milestone0` all use it, and the training evals will). Items are sorted by prompt
length (longest first) and run `--batch-size` sequences at a time (default by device:
cuda 64, mps 8, cpu 4; an item's samples stay in one batch), left-padded with the
attention mask, one forward pass per token with the KV cache; a row stops at the
model's eos or `<|im_end|>` and the batch ends when every row has (or at the cap).
Greedy is the argmax of the float32 logits. Sampling is our own loop: each reply
(item id, sample j) draws 96 random bits per token from its own `random.Random`
seeded with `"<seed>:<item id>:<j>"`, scaled to an integer below the total of the
probabilities floored to multiples of 2^-40 and summed by an integer cumulative sum
(exact, and allowed in CUDA's deterministic mode), and takes the token whose interval
holds it. So a reply's random numbers don't depend on its batch, its position or
what ran before; the forward pass's rounding can still depend on the batch's shape,
so a capture is reproduced at the same batch size. Setup prints "engine: batched,
sorted by length" and the batch size; replies rows record `engine` and `batch_size`;
seconds per item are the batch's seconds x the item's share of its rows. Captures
made before (Hugging Face `generate`, one item at a time, `bfcl.generate_text`) print
no engine line, and `rescore`/`bfcl-rescore` reproduce them unchanged.
`tinytools engine-check` compares the two engines on the Mac
(`outputs/engine-check.txt`).
`simple` is BFCL V4's `simple_python`; `--limit N` takes the first N items in bfcl's
file order.

## Our tool tasks

Five tools that run (`src/tinytools/tools/`), pure Python over fixed fake data in
`src/tinytools/tools/data/` (made once by `scripts/make_tool_data.py`, then committed;
the JSON is the source of truth):

| tool | data | execution match compares |
|---|---|---|
| calculator | none: Python arithmetic, parsed with `ast` (no `eval`), exponents capped | the number, within 1e-6 relative or 1e-9 absolute |
| search_contacts | 90 contacts; 30 with a tricky spelling (Jon / John Smith, Siobhan O'Neill) | the set of contacts returned |
| get_calendar_event | 78 events, 2026-02-23 to 2026-04-24 | the set of events returned |
| create_calendar_event | the same calendar | the whole calendar after the call |
| search_documents | 80 documents (16 subjects x Q1-Q4, plus 16 others) | the set of documents returned |

**Frozen clock:** 2026-03-11 09:30, a Wednesday (`clock.json`), stated in the system
prompt after Qwen's default sentence as "The current time is 2026-03-11 09:30,
Wednesday." (the shape ToolACE's context lines have). No tool reads the real clock.

**Distractor tools** (schemas only, offered in prompts, never a right answer, and
calling one is an error): `send_email`, `delete_calendar_event`, `get_weather`,
`set_reminder`, `get_stock_price`, `translate_text`. No-tool-fits requests need one
of these, and it is not offered; nor is a tool that could stand in for it (a
reminder request is never offered `create_calendar_event`).

**Tasks** (`tinytools tasks --seed 0`, `tasks.py`): choose+fill (pick the tool, fill
arguments from the request), tricky arguments (dates relative to the clock, 12-hour
times and worded durations, unit conversions and percentages, names with a
confusable neighbour or an unusual spelling), no tool fits (right = no call). Splits,
disjoint by request text and by (template, slot values): test 600/200/200, dev
120/40/40 (Milestone 0 only), train 3000/1000/1000 (post 10; the 60/20/20 mix is
frozen), and `train-sample`, a seeded 300/100/100 sample of train for the
ready-for-RL measurement (named `train-r6a` until review round 1; same tasks, same
order, same bytes). Each task offers 3-5 tools. The splits are written to
`data/tasks/*.jsonl` (gitignored: regenerated byte for byte; `tests/test_tasks.py`
pins their sha256).

**Checks** (`checkers.py`): exact match (one call, the reference's tool, arguments
equal after normalizing key order, number format and whitespace; an expression
compares as a syntax tree) and execution match (one call, the reference's tool,
both run on fresh data, no error, same result as in the table). Every reference
call runs without error and every reference search finds something, so an empty
result can never match. Three details of execution match: a start time with zero
seconds (`"15:00:00"`) runs as `"15:00"` (exact match still compares the strings);
a calculator call whose expression is a bare number (`"9396"`, no operator) fails
when the reference's has an operator, since it skips the tool's work and post
10's reward must not pay for that (the expressions must have the same normalized
tree, with one narrow percent rule: `3438 * 0.48` or `3438 * (48 / 100)` matches
`3438 * 48 / 100`, but `2.5` never matches `250 / 100`); and the filler words dropped from title and
document searches include their plurals ("meetings", "events", "files").
Outcomes come from `classify.py`, with execution match as the verdict.

**Scoring** (`tinytools score`): the generation path `bfcl-score` uses (the
batched engine). `--limit N` takes the first N tasks of each kind by id.
Prints the outcomes per kind (tasks that need a call; no tool fits, separately),
right by execution and by exact match,
two controls (reference answers; never calling), every reply that passes
execution but not exact match, and the reverse count (should be 0). Greedy runs
also print one test task whole (the templated prompt, its token count and the
tool list's, the reply (the greedy pass's own), parsed call and tool result as fed back; the prompt is
also written unbroken to `outputs/r1-prompt.txt`). With `--samples`: right per
task out of G, all-wrong / mixed / all-right, mixed groups per 1,000 answers
(post 8's unit), and greedy right vs pass@G. Replies go to
`outputs/score-<model>-<split>-<mode>-limit<N>-replies.jsonl`.


## The SFT training sets

`tinytools sft-data --seeds 1,2,3` (`sft.py`) reads `data/toolace-en-qwen.jsonl`
(run `toolace-convert` first; its sha256 is checked) and writes `data/sft/real.jsonl`
(no randomness) and one control set per training seed,
`data/sft/control-seed<N>.jsonl` (seed N's SFT run trains control N). All are
gitignored, regenerated byte for byte, sha256 pinned in `tests/test_sft.py`.

* **Rows (both arms).** Every row with 2+ tools and a call, and every row with no
  call, whatever its tool list (one tool or none): the declining and asking
  examples. No-call rows are identical in both arms. Rows with one tool and a call
  are left out (the control would always draw the right tool). Then some rows are
  dropped from both arms, counted by reason in the capture: an assistant turn with
  no call and no text, or a text turn holding ToolACE's `[Name(...)]` syntax
  for a listed tool, which would train a call as plain text
  (`sft.drop_reason`, checked after truncation).
* **Truncation (both arms).** Each conversation is cut right after its first
  assistant turn that makes a call; not-calling turns before it stay; a row with no
  call is kept whole. No row keeps a tool result.
* **real.jsonl:** the truncated rows.
* **control-seed<N>.jsonl:** the same rows, each call in the call turn replaced one
  for one (parallel turns keep their number of calls) by a call to a tool drawn
  uniformly from the row's whole list, the right one included. No argument comes
  from the row itself. Arguments come from, in order: a call to that tool in
  another row that fits this row's schema; one that doesn't; else the call is
  built from the schema (even when the drawn tool is the right one): each
  required parameter takes an enum value, a same-named argument from another row,
  the schema default, or a value of the declared type copied from any argument in
  another row (seeded choice). Empty values (`""`, `[]`, `{}`, `null`) are never
  used. Not-calling turns are unchanged. Draws come from `Random("seed:row id")`.
  Each control record carries `random_calls`: per call, the tool replaced, the
  tool drawn, the argument source, the donor row, and each built parameter's
  source and donor row (empty for no-call rows).

Records have the `toolace-convert` shape plus `n_tokens` and `n_trained` (the
scored tokens: each assistant turn from after its `<|im_start|>assistant` header
through its `<|im_end|>`; the reports call them "scored"). The report
(`outputs/sft-data.txt`) prints which rows train by type, the truncation, the turns
in each arm beside the full set, tokens and the control/real token ratio per seed,
each seed's hit rate against chance by k, the argument sources and how built
parameters were filled, how many control calls equal the real call (coincidence
only: mostly tools that take no arguments), and one row (`toolace-4231`) with its
seed-1 control twin.


## Post 9's main run: SFT, scoring, the report

`experiments/main.toml` (plan 3e, "T8 spec"; A100 40 GB, ~5.9 h expected, 7.5 h and $16 at
most) runs, in order: the data; `sft-check`; `engine-check --quick`; `sft-score --untrained`;
`sft-score --untrained --noise-floor` (r6a's baseline); then
per seed 1, 2, 3 `sft-pair` (train real, score it, train control, score it); seed 1 real
trained again; `sft-report`; and the untrained replies rescored into the r1-r3b captures.
`scripts/lambda.sh estimate main` prints the live price against the caps.

**Training** (`train.py`, settings fixed before any result and printed in r4c): Qwen2.5-1.5B-
Instruct with frozen bf16 weights; LoRA rank 16, alpha 32, dropout 0 on q, k, v, o, gate, up,
down in every layer, with A, B, their gradients and AdamW's state in float32 (the matmuls run
in the activations' dtype, as autocast does); 2 epochs of the 8,504 rows; 16 rows per step,
split into micro-batches by a token budget of 8,192 (rows x longest row), gradients
accumulated; loss the mean over the step's scored tokens; AdamW (0.9, 0.999), weight decay 0,
no clipping; lr 2e-4, Hugging Face's cosine schedule with 3% warmup (32 of 1,064 steps); max
length 3,072 with nothing cut (the longest row is 2,975). Seed N sets LoRA A's init and the
shuffle order; real N and control N share both (the pair). Gradient checkpointing is off only if
`sft-check`'s memory step (the two largest micro-batches, forward and backward, then AdamW, on
the run's GPU, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) peaks at no more than
85% of the GPU's memory without it. Attention always gets an explicit mask, so every
micro-batch runs the same op; if an op has no deterministic CUDA kernel, that step is redone
with `warn_only` and the run's record says so.

**Scoring** an adapter merges it: W + (alpha/r) BA in float32, cast to the weights' dtype once,
then the batched engine (batch 64 on CUDA) runs our 1,000 test tasks greedy, BFCL's 1,116 eval
items greedy, and the 500 train-sample tasks greedy and 8 at T=1. The unmerged adapter is
saved beside it (post 10 needs it). `sft-check` merges an untrained adapter (B = 0) and checks
the replies on the 200 dev tasks are the base model's, byte for byte, and merges an adapter
with random B (0.01 x N(0, 1)) and checks the merged model's logits are the LoRA model's on 6
dev prompts (float32; relative gap at most 0.01, where a merge 25% off in scale must fail);
it also checks the four SFT sets' sha256 are the Mac's (pinned in `sft.py`), the 1.5B template
renders every SFT row as sft-data's 0.5B rendering did, and that every assistant turn's
`<|im_end|>` is scored.

**Files**: `outputs/main/<label>/` per model (`untrained`, `real-seed<N>`, `control-seed<N>`,
`real-seed1-repeat`): `adapter.safetensors`, `losses.jsonl`, `train.json`, the three replies
files and `score.json` (written last; a directory without it is incomplete). Every file is
written whole and renamed into place. Adapters are compared by a digest of their tensors,
not the file: safetensors writes its metadata in no fixed order.

**The report** (`sft-report`, `outputs/sft.txt`) re-checks every saved reply with the current
checkers, so a recapture after review is a rescore (`sft-report`, or `sft-score --adapter` for
a saved adapter); retraining is needed only if the trainer changes. It reports only seeds
whose real and control runs are both complete, and prints r4c (settings, dtypes, checks), r4d
(per run: first and last loss, wall time; seed 1 trained twice: losses and adapters compared),
r5a (right per arm and category, BFCL calls pooled, gains with paired intervals), r5b (six
outcomes per arm; right function vs chance), r5c (declining), r5d (the four effects, each
SFT - untrained, control - untrained, SFT - control, the five primary comparisons marked) and
r6a (groups of 8 on the train-sample tasks; first the noise floor, untrained -> untrained
drawn again with generation seed 1; then the untrained -> arm transition table pooled, per seed
and by kind, share differences, and the four movements read beyond the floor with paired
intervals; tasks that need a call and all kinds). Intervals: paired bootstrap over tasks, 10,000 resamples, numpy
`default_rng(0)` afresh per comparison, 95% percentile; an arm's per-task value is the mean of
its seeds' 0/1 (`effects.py`).

**Deadline and fetching.** `sft-pair` and `sft-train --repeat` take `--deadline-epoch`: from
what has been measured (seconds per token slot of training, seconds per scoring; Milestone 0's
probe before any) x 1.5, with 15 minutes kept for the report, work that can't finish is not
started (exit 0, so the report still runs on the whole pairs). After each pair the job prints
`TINYTOOLS_FETCH_NOW` and the runner fetches what exists (a checkpoint). Each fetch first
touches `.run/fetch-N` on the instance; a later fetch skips a file the laptop already has only
if it has not changed since the last completed fetch's marker (`find -cnewer`), so a file
rewritten at the same size is fetched again.

**Quick mode** (`--quick`, the Mac): Qwen2.5-0.5B-Instruct, the first 32 SFT rows (4 steps),
2 tasks of each kind and category, into `outputs/main-quick/` and `outputs/sft*-quick.txt`. It
shows the pipeline works; its numbers are not results.

**The run (2026-11-23).** `runs/2026-11-23-main/` (commit 29aeae0): ok in 3.86 h, $7.69,
terminated and confirmed. A first attempt (`runs/2026-11-22-main/`, $0.87) stopped at untrained
scoring on a report line wider than 74 columns, printed only in full mode; since fa690fc a wide
line is folded outside tests rather than raised. Every `sft-check` gate passed (memory 28.6 of
36.0 GB without checkpointing; both merges), and seed 1 trained twice gave identical losses
and a bitwise-identical adapter. Its captures are copied into `outputs/` (`sft.txt` is the
report; `sft-check.txt`, `sft-score-untrained*.txt`, `sft-pair-seed<N>.txt`,
`sft-train-real-seed1-repeat.txt`, and the 1.5B-Instruct `score-*` / `bfcl-*` rescores), which
is where the post's receipts are checked. The seven adapters (about 70 MB each) are in the run
directory, not in git. What the report says, in short (seed means; read the numbers in
`outputs/sft.txt`): on our tasks that need a call, right goes from 60.0% to 90.0%, mostly by
filling arguments (wrong arguments 25.5% -> 8.4%; the control gets worse); calling at all rises
in both arms alike (the format, not the content); declining where no tool fits gets worse; on
BFCL's call items right falls from 85.1% to 51.3%, because both arms often reply with a request
for parameters the question already gives (ToolACE's no-call rows). The untrained Instruct model
already calls (BFCL calls 85.1% right), which is why post 9 adds the base arm below.


## Post 9's base arm: the same SFT on a model with no tool calling

`--profile base` (plan 3f; `experiments/base.toml`, A100 40 GB, ~4.9 h expected, 6 h and $12 at
most) runs every `sft-*` command, and `rescore` / `bfcl-rescore` of the untrained captures, on
Qwen2.5-1.5B (base) instead of the Instruct model: the same SFT sets (sha256-pinned), settings,
seeds, scoring, noise floor, repeat check and report. The default profile, `main`, is the
Instruct arm exactly as it ran (its report, regenerated from the main run's replies, is byte
for byte the fetched `sft.txt` but for the machine header). What differs, fixed before any
result:

* **Prompts.** Every training row and every scored prompt (ours, BFCL, train-sample) is
  rendered with the Instruct tokenizer's chat template, so it is byte for byte the Instruct
  arm's. The base template differs only in its default system line (the base model's own
  template would change every BFCL prompt and about half the training rows, those without
  a date line; ours carry the system line explicitly; `sft-check` counts them). The two tokenizers split text identically (vocabulary,
  merges, normalizer, added tokens; `train.load_tokenizer` asserts it); only the eos name
  and the template differ. `bfcl.prompt_text` builds BFCL prompts with that tokenizer; bfcl-eval's
  `QwenFCHandler` only decodes and judges replies.
* **Where replies end.** The base model's generation_config stops only at `<|endoftext|>`; the
  profile sets `<|im_end|>` and `<|endoftext|>` on every model it loads, and `sft-check` prints
  what the engine stops at.
* **Four embedding rows.** In the base model `<|im_start|>`, `<|im_end|>`, `<tool_call>` and
  `</tool_call>` (ids 151644, 151645, 151657, 151658: a token's id is its row) have the same
  embedding (pairwise cosine 1.0000, norm 0.414 against a median 1.032; the Instruct model's are
  distinct, cosine 0.92-0.95). They are not alone: on 1.5B base 2,234 rows share
  `<|im_start|>`'s direction (21 added tokens, 1,942 rare byte tokens such as 'À', and the 271
  padding rows past the tokenizer's 151,665 ids). These tokens never appeared in pretraining,
  and a row that is never a target gets the same softmax push down as its twins, so rows that
  start equal stay equal. The embedding is tied to the output head, so "end the turn" and
  "start a call" score the same from any hidden state, and LoRA, which adapts only attention and
  MLP, can't separate them. The base arm trains
  a float32 delta on those rows (4 x 1,536 = 6,144 parameters, zero at the start) beside LoRA,
  in the same AdamW group (lr, schedule, betas, weight decay 0). One parameter feeds both the
  input embedding and the tied head (`lora.inject_rows`: one-hot products, no scatter, so CUDA's
  deterministic mode holds). It is saved in the adapter file (`embed_rows.delta`,
  `embed_rows.ids`; the digest covers both; metadata names the tokens and ids), and merging
  writes E + delta into the tied embedding (float32, cast once). r4c prints the tokens, ids and
  both parameter counts; each `sft-pair` capture prints the delta's size per row and the four
  rows' cosines before and after.
* **Checks** (`sft-check --profile base`, all gating): main's, with the delta in the memory
  step (which also probes the micro-batch with the most scored tokens: the delta adds a
  full-vocabulary term per scored token) and in both merges (B = 0 and delta = 0 merged gives the base model's replies byte for
  byte; random B and a random delta merged gives the LoRA model's float32 logits, and the merge
  without the rows must fail); plus the prompts against the Instruct arm's, the stop tokens,
  and the four rows' identity in the base model beside the Instruct model's (the reason for
  the delta, as a receipt: the four rows' cosines, and the count of every row in the embedding
  that shares them).
* **Files.** `outputs/base/` (`outputs/base-quick/` with `--quick`, on Qwen2.5-0.5B, whose rows
  have the same defect), captures `outputs/sft*-base.txt` (`-base-quick`), the untrained copies
  `outputs/score-Qwen2.5-1.5B-*` and `outputs/bfcl-Qwen2.5-1.5B-*`, r1's prompt in
  `outputs/base/r1-prompt.txt`. Nothing of the main run's is overwritten.
* **Deadline.** The untrained base model runs nearly every reply to the 512-token cap (~0.8 h to
  score), so the base arm estimates a pair from the trained models' scoring once one is
  measured (before that, the plan's prior), not from the untrained one's; main's rule is
  unchanged. Known case (review T9e S3): if the stages before the pairs took about twice their
  expected time, seed 1's pair could be started late enough to reach the hard limit; the
  runner's limit and caps still hold.
* **Two readings, and the rule.** The untrained base model can't write the tags but often
  writes the call itself as bare JSON (`{"name": ..., "arguments": ...}`, then text to the cap;
  Milestone 0: 32 of 160 dev call tasks). So every arm, base and Instruct, is reported on two
  readings: strict (calls only inside `<tool_call>` tags, as everything above) and tag-free (the
  call objects at the start of the reply, tags or not). The base arm's rule, fixed before the
  run (plan 3f): it is featured in the post if SFT - untrained on our tasks that need a call,
  right, on the tag-free reading, has a 95% interval excluding 0; otherwise it is reported in a
  paragraph. Both arms are shown side by side either way. The tag-free rescore reads saved
  replies (no model).
* **The run.** `runs/2026-11-23-base/` (commit 44b5cb6), launched 2026-11-23 16:37.

