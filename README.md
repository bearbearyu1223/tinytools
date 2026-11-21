# tinytools

Companion project for posts 9-12 of the LLM Architecture Refresh series: teaching a
small model to call tools (SFT on ToolACE, scored on BFCL), then RL with execution
rewards. Local only. This first piece is data and evaluation; no training yet.

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
uv run tinytools milestone0              # six models on the dev split, the rule's verdict (H100)
scripts/lambda.sh doctor                 # Lambda settings, SSH key, API; launches nothing
scripts/lambda.sh estimate milestone0    # price x hours against the experiment's cap
scripts/lambda.sh run milestone0         # up -> setup -> run -> fetch -> down (docs/lambda-playbook.md)
uv run pytest -q && uvx ruff check --select F,E9,B
```

Every command prints a 74-column report on stdout (progress on stderr only) and
writes the same report to `outputs/`. `bfcl-score` also writes every reply, its
outcome and BFCL's error type to `outputs/bfcl-<model>-<mode>-limit<N>-replies.jsonl`.
`data/` (converted ToolACE) and `bfcl_project/` (bfcl-eval's scratch directories)
are gitignored.

GPU runs go through `scripts/lambda.sh` and `experiments/<name>.toml`, as
`docs/lambda-playbook.md` describes: an approved estimate, a hard time limit, termination
through Lambda's API whatever happens, and a record in `runs/<date>-<exp>/run.json` and
`runs/ledger.csv`. Settings and keys live in `lambda.env` (gitignored; see
`lambda.env.example`).

Environment: `TINYTOOLS_DEVICE=cpu|mps|cuda` overrides the device (default cuda >
mps > cpu); `TINYTOOLS_QUIET=1` silences progress.

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
--temperature T` samples the full softmax. Each item's generator is seeded with
`seed + crc32(item id)`, so an item's replies don't depend on what ran before it.
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

**Scoring** (`tinytools score`): the generation path `bfcl-score` uses
(`bfcl.generate_text`). `--limit N` takes the first N tasks of each kind by id.
Prints the outcomes per kind (tasks that need a call; no tool fits, separately),
right by execution and by exact match,
two controls (reference answers; never calling), every reply that passes
execution but not exact match, and the reverse count (should be 0). Greedy runs
also print one test task whole (the templated prompt, its token count and the
tool list's, the reply, parsed call and tool result as fed back; the prompt is
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
