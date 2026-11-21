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
uv run tinytools bfcl-score --limit 20 --samples 8 --temperature 1.0
uv run tinytools tasks              # our tool tasks -> data/tasks/, counts (r2a)
uv run tinytools check-demo         # three calls checked both ways (r2b)
uv run tinytools score --split test --limit 30              # greedy, 30 per kind
uv run tinytools score --split test --limit 10 --samples 8 --temperature 1.0
uv run pytest -q && uvx ruff check --select F,E9,B
```

Every command prints a 74-column report on stdout (progress on stderr only) and
writes the same report to `outputs/`. `bfcl-score` also writes every reply, its
outcome and BFCL's error type to `outputs/bfcl-<model>-<mode>-limit<N>-replies.jsonl`.
`data/` (converted ToolACE) and `bfcl_project/` (bfcl-eval's scratch directories)
are gitignored.

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
  whose tools are written another way (Markdown, XML, YAML, HTML or LaTeX tables:
  ToolACE's format-variation rows, which also use other call syntaxes) are counted and
  dropped.
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
the five-way split (`classify.py`):

| outcome | meaning |
|---|---|
| no call | no `<tool_call>` tag |
| unparseable | a tag, but some tagged block doesn't decode to a call |
| wrong function | called names differ from the reference (as multisets: includes the wrong number of calls); on irrelevance, any decodable call |
| wrong arguments | right names, BFCL's checker rejects the arguments |
| right | BFCL's checker accepts it; on irrelevance, no tag |

"right" equals BFCL's accuracy on the AST categories. On irrelevance BFCL also
passes a reply whose `<tool_call>` doesn't decode; the report prints BFCL's count
beside ours. Two controls print with every run: BFCL's own reference answer written in
Qwen's format (shows the harness can give full marks), and how many replies hit the
512-token cap.

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
frozen), and `train-r6a`, a seeded 300/100/100 sample of train for the
ready-for-RL measurement. Each task offers 3-5 tools. The splits are written to
`data/tasks/*.jsonl` (gitignored: regenerated byte for byte; `tests/test_tasks.py`
pins their sha256).

**Checks** (`checkers.py`): exact match (one call, the reference's tool, arguments
equal after normalizing key order, number format and whitespace; an expression
compares as a syntax tree) and execution match (one call, the reference's tool,
both run on fresh data, no error, same result as in the table). Every reference
call runs without error and every reference search finds something, so an empty
result can never match. Outcomes come from `classify.py`, with execution match as
the verdict.

**Scoring** (`tinytools score`): the generation path `bfcl-score` uses
(`bfcl.generate_text`). `--limit N` takes the first N tasks of each kind by id.
Prints the five-way split per kind with right by execution and by exact match,
two controls (reference answers; never calling), every reply that passes
execution but not exact match, and the reverse count (should be 0). Greedy runs
also print one test task whole (the templated prompt, its token count and the
tool list's, the reply, parsed call and tool result as fed back; the prompt is
also written unbroken to `outputs/r1-prompt.txt`). With `--samples`: right per
task out of G, all-wrong / mixed / all-right, mixed groups per 1,000 answers
(post 8's unit), and greedy right vs pass@G. Replies go to
`outputs/score-<model>-<split>-<mode>-limit<N>-replies.jsonl`.

