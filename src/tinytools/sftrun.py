"""Post 9's main run: the SFT checks, training, merged-adapter scoring and the report.

Commands (``cli.py`` wires them; ``experiments/main.toml`` runs them in order):

    sft-check   the template byte-identity check, the four sets' sha256 against the
                pinned ones, row lengths, <|im_end|> scored, one training step's memory
                (decides gradient checkpointing), the system-prompt shapes, and two merge
                checks (an untrained adapter, B = 0, merged, gives the base model's
                replies byte for byte on the dev tasks; one with random B, merged, gives
                the LoRA model's logits)
    sft-score   score the untrained model, or a saved adapter merged into it: our test
                tasks greedy, BFCL eval greedy, and the train-sample tasks greedy + 8 at T=1
                (``--noise-floor``: the untrained model's train-sample 8 at T=1 again,
                generation seed 1, r6a's baseline)
    sft-train   train one adapter (``--repeat``: seed 1 real again, compared with the first)
    sft-pair    one seed's pair: train real, score, train control, score
    sft-report  r4c, r4d, r5a-d and r6a, from the saved runs and replies (no model)

Everything lands in ``outputs/main/<label>/`` (``outputs/main-quick/`` with ``--quick``):
``untrained/``, ``untrained-gen-seed1/`` (the noise floor), ``real-seed<N>/``,
``control-seed<N>/``, ``real-seed1-repeat/``. A label's
``score.json`` is written last, so a directory without one is incomplete; ``sft-report``
uses only seeds whose real and control runs are both complete (pairs stay whole). Every
file is written whole and renamed into place, so a fetch mid-run never sees half a file.

**Recapture after review = rescore.** ``sft-report`` re-checks every saved reply with
the current checkers; ``sft-score --adapter`` re-scores a saved adapter. Retraining is
needed only if the trainer changes.

**Profiles** (``--profile``, ``train.PROFILES``; plan 3f). ``main`` (the default) is the
Instruct arm, unchanged. ``base`` runs every command above on Qwen2.5-1.5B (base) into
``outputs/base/`` (``outputs/base-quick/``, on Qwen2.5-0.5B, with ``--quick``), with
captures named ``sft*-base.txt``: prompts and training rows in the Instruct tokenizer's
chat template, generation stopping at ``<|im_end|>`` or ``<|endoftext|>`` (set on the
loaded model, whatever its generation_config says), and four embedding rows trained
beside LoRA. ``sft-check`` adds the checks the base arm needs: its prompts are the
Instruct arm's byte for byte, the stop tokens, and the four rows' identity in the base
model (the reason for the delta). ``onetool`` (plan 3g) is main's settings and model,
the real arm only, on ``data/sft/real-onetool.jsonl`` (``sft-data --variant onetool``)
into ``outputs/onetool/``, captures ``sft*-onetool.txt``: ``sft-check`` pins and checks
that one set, ``sft-pair`` trains and scores the real arm alone (the pair's code over
the profile's arms), there is no untrained scoring (the main run's is read), and
``sft-report`` is the comparison with the main run's real arm (:mod:`tinytools.onetool`).
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import train as T

SAMPLES = 8
TEMPERATURE = 1.0
SEED = 0  # generation seed (sampling draws per reply: seed, task id, sample number)
OURS_FILE = "ours-test-greedy-replies.jsonl"
BFCL_FILE = "bfcl-eval-greedy-replies.jsonl"
SAMPLE_MODE = f"samples{SAMPLES}-t{TEMPERATURE:g}"
SAMPLE_FILE = f"train-sample-{SAMPLE_MODE}-replies.jsonl"
UNTRAINED = "untrained"
# r6a's noise floor (T8 review B2, plan 3d): the untrained model's train-sample 8 at T=1
# drawn a second time with this generation seed, into outputs/main/<FLOOR>/.
FLOOR_SEED = 1
FLOOR = f"untrained-gen-seed{FLOOR_SEED}"


@dataclass(frozen=True)
class Mode:
    quick: bool
    settings: T.Settings
    ours_limit: int | None  # test tasks per kind
    bfcl_limit: int | None  # BFCL eval items per category
    sample_limit: int | None  # train-sample tasks per kind
    dev_limit: int | None  # dev tasks per kind (the merge check)
    out: Path
    suffix: str  # in capture names
    profile: str = "main"
    tokenizer: str | None = None  # whose tokenizer and chat template (None: the model's own)
    embed_rows: tuple[str, ...] = ()  # tokens whose embedding rows train (the base arm)
    stop_tokens: tuple[str, ...] = ()  # set on the loaded model (the base arm)
    real_set: str = "real.jsonl"  # the real arm's file in data/sft/ (onetool: its variant)
    arms: tuple[str, ...] = T.ARMS  # the arms a seed's runs train (onetool: real only)

    @property
    def title(self) -> str:
        return T.PROFILES[self.profile].title()


def mode(quick: bool, root: Path, settings: T.Settings | None = None,
         profile: str = "main") -> Mode:
    p = T.PROFILES[profile]
    s = settings or p.settings(quick)
    tok = p.tokenizer_for(s.model) if p.tokenizer is not None else None
    lim = 2 if quick else None
    return Mode(quick, s, lim, lim, lim, lim, root / "outputs" / p.out_name(quick),
                p.suffix(quick), p.name, tok, p.embed_rows, p.stop_tokens, p.real_set, p.arms)


def run_label(arm: str, seed: int, repeat: bool = False) -> str:
    return f"{arm}-seed{seed}" + ("-repeat" if repeat else "")


def set_path(root: Path, arm: str, seed: int, real_set: str = "real.jsonl") -> Path:
    from .cli import SFT_DIR

    return SFT_DIR / (real_set if arm == "real" else f"control-seed{seed}.jsonl")


def _free(device) -> None:
    from .milestone0 import _free as free

    gc.collect()
    free(device)


def _setup(device, batch: int) -> dict:
    from .device import dtype_name, preferred_dtype
    from .engine import ENGINE

    return {"dtype": dtype_name(preferred_dtype(device)), "device": device.type,
            "engine": ENGINE, "batch_size": batch}


def _replace(tmp: Path, path: Path) -> None:
    tmp.replace(path)


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


# -- scoring one model --------------------------------------------------------------------


def score_model(tok, lm, label: str, m: Mode, batch: int, setup: dict,
                extra: dict | None = None) -> dict:
    """The three passes every model gets, written to ``<out>/<label>/``; ``score.json`` last."""
    from . import bfcl, ours, progress
    from .cli import _load_or_make_tasks, _write_jsonl, _write_replies

    d = m.out / label
    d.mkdir(parents=True, exist_ok=True)
    splits = _load_or_make_tasks(0)
    t0 = time.perf_counter()
    passes = {}

    test = ours.limit_per_kind(splits["test"], m.ours_limit)
    t = time.perf_counter()
    greedy = ours.score(tok, lm, test, samples=1, temperature=0.0, seed=SEED, batch_size=batch,
                        on_item=progress.items(f"{label}: our test tasks, greedy", len(test)))
    _write_replies(_tmp(d / OURS_FILE), greedy, None, "greedy", setup)
    _replace(_tmp(d / OURS_FILE), d / OURS_FILE)
    passes["ours test, greedy"] = {"items": len(test), "seconds": time.perf_counter() - t}

    records: list[dict] = []
    t = time.perf_counter()
    n_bfcl = 0
    for cat in bfcl.CATEGORIES:
        items = bfcl.load_items(cat, m.bfcl_limit, "eval")
        n_bfcl += len(items)
        bfcl.score(tok, lm, items, samples=1, temperature=0.0, seed=SEED, batch_size=batch,
                   on_item=progress.items(f"{label}: BFCL eval {cat}, greedy", len(items)),
                   records=records, setup=setup)
    _write_jsonl(_tmp(d / BFCL_FILE), records)
    _replace(_tmp(d / BFCL_FILE), d / BFCL_FILE)
    passes["BFCL eval, greedy"] = {"items": n_bfcl, "seconds": time.perf_counter() - t}

    sample = ours.limit_per_kind(splits["train-sample"], m.sample_limit)
    t = time.perf_counter()
    g2 = ours.score(tok, lm, sample, samples=1, temperature=0.0, seed=SEED, batch_size=batch,
                    on_item=progress.items(f"{label}: train-sample, greedy", len(sample)))
    s2 = ours.score(tok, lm, sample, samples=SAMPLES, temperature=TEMPERATURE, seed=SEED,
                    batch_size=batch, on_item=progress.items(
                        f"{label}: train-sample, {SAMPLES} at T={TEMPERATURE:g}", len(sample)))
    _write_replies(_tmp(d / SAMPLE_FILE), g2, s2, SAMPLE_MODE, setup)
    _replace(_tmp(d / SAMPLE_FILE), d / SAMPLE_FILE)
    passes[f"train-sample, greedy + {SAMPLES} at T={TEMPERATURE:g}"] = {
        "items": len(sample), "seconds": time.perf_counter() - t}

    info = {"label": label, "model": m.settings.model, **setup, "seed": SEED,
            "samples": SAMPLES, "temperature": TEMPERATURE,
            "max_new_tokens": bfcl.MAX_NEW_TOKENS,
            "limits": {"ours": m.ours_limit, "bfcl": m.bfcl_limit, "sample": m.sample_limit},
            "passes": passes, "wall_s": time.perf_counter() - t0, **(extra or {}),
            **_profile_fields(m)}
    T.write_json(d / "score.json", info)
    return info


def score_floor(tok, lm, m: Mode, batch: int, setup: dict) -> dict:
    """The noise floor: the train-sample tasks' 8 at T=1 again, generation seed
    :data:`FLOOR_SEED` (no greedy pass: greedy is the same reply), into ``<out>/FLOOR/``;
    ``score.json`` last."""
    from . import ours, progress
    from .cli import _load_or_make_tasks, _write_replies

    d = m.out / FLOOR
    d.mkdir(parents=True, exist_ok=True)
    sample = ours.limit_per_kind(_load_or_make_tasks(0)["train-sample"], m.sample_limit)
    t0 = time.perf_counter()
    s2 = ours.score(tok, lm, sample, samples=SAMPLES, temperature=TEMPERATURE, seed=FLOOR_SEED,
                    batch_size=batch, on_item=progress.items(
                        f"{FLOOR}: train-sample, {SAMPLES} at T={TEMPERATURE:g}", len(sample)))
    _write_replies(_tmp(d / SAMPLE_FILE), [], s2, SAMPLE_MODE,
                   {**setup, "generation_seed": FLOOR_SEED})
    _replace(_tmp(d / SAMPLE_FILE), d / SAMPLE_FILE)
    wall = time.perf_counter() - t0
    info = {"label": FLOOR, "model": m.settings.model, **setup, "seed": FLOOR_SEED,
            "samples": SAMPLES, "temperature": TEMPERATURE, "adapter": None,
            "limits": {"sample": m.sample_limit},
            "passes": {f"train-sample, {SAMPLES} at T={TEMPERATURE:g}": {
                "items": len(sample), "seconds": wall}}, "wall_s": wall, **_profile_fields(m)}
    T.write_json(d / "score.json", info)
    return info


def _profile_fields(m: Mode) -> dict:
    """What a record adds for a profile other than main (main's records are unchanged)."""
    if m.profile == "main":
        return {}
    return {"profile": m.profile, "tokenizer": m.tokenizer, "stop_tokens": list(m.stop_tokens),
            **({"real_set": m.real_set} if m.real_set != "real.jsonl" else {})}


def standard_copies(m: Mode, root: Path) -> list[Path]:
    """The untrained replies under the names ``rescore`` and ``bfcl-rescore`` read, so the
    r1-r3b captures can be made from them without a model (full runs only)."""
    if m.quick:
        return []
    name = m.settings.model.split("/")[-1]
    src = m.out / UNTRAINED
    out = root / "outputs"
    pairs = [(OURS_FILE, f"score-{name}-test-greedy-limitall-replies.jsonl"),
             (BFCL_FILE, f"bfcl-{name}-eval-greedy-limitall-replies.jsonl"),
             (SAMPLE_FILE, f"score-{name}-train-sample-{SAMPLE_MODE}-limitall-replies.jsonl")]
    done = []
    for a, b in pairs:
        shutil.copyfile(src / a, out / b)
        done.append(out / b)
    return done


def print_copies(rep, paths: list[Path]) -> None:
    """The copies ``standard_copies`` made, one file name per line (full runs only: the
    names are long, so they go on lines of their own, not after a ``kv`` label)."""
    if not paths:
        return
    rep.note("copied for rescore, in outputs/:")
    for p in paths:
        rep.note(f"  {p.name}", wrap=True)


def load_merged(model_id: str, adapter: Path, device, tokenizer: str | None = None):
    """The base model with a saved adapter merged in (fp32 sum, cast once; its embedding
    rows too, if it has any)."""
    from . import bfcl
    from .lora import load_adapter, merge_into

    state, meta = load_adapter(adapter)
    if meta.get("base_model") != model_id:
        raise ValueError(f"{adapter} was trained on {meta.get('base_model')}, not {model_id}")
    tok, lm = bfcl.load_model(model_id, device, tokenizer)
    n = merge_into(lm, state, float(meta["alpha"]) / int(meta["rank"]))
    return tok, lm, meta, n


def set_stops(lm, tok, tokens: tuple[str, ...]) -> list[int]:
    """Generation stops at ``tokens`` whatever the model's generation_config said (the
    engine reads ``generation_config.eos_token_id`` and adds ``<|im_end|>``); returns the
    ids. ``()`` leaves the model as loaded (the main arm, as it ran)."""
    if not tokens:
        return []
    ids = T.token_ids(tok, tokens)
    lm.generation_config.eos_token_id = list(ids)
    return ids


def load_for(m: Mode, device, adapter: Path | None = None):
    """(tokenizer, model[, meta, modules merged]) as this profile scores: the profile's
    tokenizer, its stop tokens, the adapter merged if given."""
    from . import bfcl

    if adapter is None:
        tok, lm = bfcl.load_model(m.settings.model, device, m.tokenizer)
        set_stops(lm, tok, m.stop_tokens)
        return tok, lm
    tok, lm, meta, n = load_merged(m.settings.model, adapter, device, m.tokenizer)
    set_stops(lm, tok, m.stop_tokens)
    return tok, lm, meta, n


# -- timings for the deadline ----------------------------------------------------------------


def measured(m: Mode) -> tuple[list[dict], list[float]]:
    """(training records, scoring wall seconds) of every finished run so far."""
    runs, walls = measured_by_label(m)
    return runs, scoring_walls(walls, prefer_trained=m.profile != "main")


def measured_by_label(m: Mode) -> tuple[list[dict], dict[str, float]]:
    """(training records, scoring wall seconds by run label) of every finished run so far."""
    runs, walls = [], {}
    if m.out.exists():
        for d in sorted(m.out.iterdir()):
            if (d / "train.json").exists():
                runs.append(json.loads((d / "train.json").read_text()))
            if (d / "score.json").exists():
                walls[d.name] = json.loads((d / "score.json").read_text())["wall_s"]
    return runs, walls


def scoring_walls(walls: dict[str, float], prefer_trained: bool) -> list[float]:
    """The scoring times a pair's estimate takes the slowest of (an empty list: the prior,
    :data:`SCORE_PRIOR_S`). Main: every scoring so far, untrained included (as main ran).
    The base arm (``prefer_trained``): the trained models' only. The untrained base model
    runs nearly every reply to the 512-token cap (Milestone 0 on the A100: 0.43 s per
    greedy task, against the Instruct model's 0.08), so its scoring (~0.8 h) says nothing
    about a trained model's, and taken as one would leave seeds 2 and 3 unstarted within
    the base run's hours even if training taught the model to end its turns. Until a
    trained model is scored the prior stands, as it does for main before anything is
    measured; a trained model that still runs to the cap is measured by seed 1's pair, so
    seed 2's estimate sees it.

    Known case, accepted (T9e review S3; tests/review/test_review_t9_deadline.py): seed
    1's pair is estimated with the prior, so if its trained models still score at the
    untrained pace (~2,809 s each), a pair admitted late can overrun the deadline. That
    needs the stages before the pairs to take about twice base.toml's estimate (~2 h, so
    the pair starts ~3.85 h in); the runner's hard limit still holds (the pair is cut off,
    nothing before it is lost). Using the untrained wall instead would skip seed 1 needlessly
    after ~2.26 h, so the estimate stays as it is. Pure."""
    trained = [w for k, w in walls.items() if k.startswith(tuple(f"{a}-seed" for a in T.ARMS))]
    return trained if prefer_trained else list(walls.values())


SCORE_PRIOR_S = 0.25 * 3600  # plan 3e: one model's scoring, before any is measured


def planned_padded_tokens(root: Path, s: T.Settings, seed: int,
                          real_set: str = "real.jsonl") -> int:
    """Token slots one run computes (from the records' token counts, no tokenizer)."""
    path = set_path(root, "real", seed, real_set)
    lengths = [r["n_tokens"] for r in T.load_set(path, s.rows)]
    steps = T.plan_steps(len(lengths), seed, s.epochs, s.batch_rows)
    return sum(T.padded_tokens(T.micro_batches(st.rows, lengths, s.token_budget), lengths)
               for st in steps)


def needed_seconds(m: Mode, root: Path, seed: int, trainings: int, scorings: int) -> float:
    runs, scores = measured(m)
    return T.estimate_seconds([r["seconds"] / r["padded_tokens"] for r in runs
                               if r.get("padded_tokens")], scores,
                              planned_padded_tokens(root, m.settings, seed, m.real_set),
                              trainings, scorings,
                              m.settings.grad_checkpointing, SCORE_PRIOR_S)


# -- sft-check ---------------------------------------------------------------------------------


def template_identity(records: list[dict], tok) -> tuple[int, list[str]]:
    """(rows rendered byte-identically by ``tok``'s chat template, ids that differ): text,
    train spans and both token counts, as ``sft.render`` makes them."""
    from . import sft

    same, differ = 0, []
    for r in records:
        again = sft.render(r["id"], sft.unwrap(r), r["messages"], tok)
        if all(again[k] == r[k] for k in ("text", "train_spans", "n_tokens", "n_trained")):
            same += 1
        else:
            differ.append(r["id"])
    return same, differ


END = "<|im_end|>"


def im_end_scored(text: str, offsets: list[tuple[int, int]], mask: list[bool],
                  spans: list[tuple[int, int]]) -> tuple[int, int]:
    """(assistant turns, of them whose closing <|im_end|> is one token inside the scored
    mask). Pure."""
    turns = ok = 0
    by_span = {tuple(o): i for i, o in enumerate(offsets)}
    for _, e in spans:
        turns += 1
        i = by_span.get((e - len(END), e))
        ok += text[e - len(END):e] == END and i is not None and mask[i]
    return turns, ok


def system_shape(text: str) -> list[str]:
    """The system turn's lines before its tools section (the prompt's shape); the whole
    system turn when it has no tools section (a row with an empty tool list)."""
    head = text[len("<|im_start|>system\n"):]
    head = head[:head.index(END)]
    cut = head.find("\n\n# Tools")
    return (head if cut < 0 else head[:cut]).split("\n")


def profile_pins(real_set: str) -> dict[str, str]:
    """The sets a profile trains on, with their pinned sha256 (sft.py): the four sets for
    main and base, the variant's file alone for a variant (onetool). Pure."""
    from . import sft

    if real_set == "real.jsonl":
        return dict(sft.SET_SHA256)
    return {real_set: sft.VARIANT_SHA256[real_set]}


def set_pins(sft_dir: Path, pins: dict[str, str] | None = None
             ) -> dict[str, tuple[str | None, str]]:
    """Each pinned SFT set (``pins``; default the four): (its sha256 here, ``None`` if
    missing; the pinned sha256)."""
    import hashlib

    from . import sft

    out = {}
    for name, want in (sft.SET_SHA256 if pins is None else pins).items():
        path = sft_dir / name
        out[name] = (hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                     want)
    return out


def lengths_census(lengths: list[int], max_length: int) -> dict:
    return {"rows": len(lengths), "longest": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "over 1024": sum(n > 1024 for n in lengths),
            f"over {max_length}": T.rows_over(lengths, max_length)}


def largest_micro_batches(lengths: list[int], s: T.Settings, seeds=(1, 2, 3),
                          k: int = 2) -> list[list[int]]:
    """The ``k`` micro-batches with the most token slots (rows x longest row) over the
    seeds' plans, the longest row breaking ties, each a different set of rows: the step
    most likely to run out of memory, as :func:`memory_probe` runs it. Pure."""
    seen: dict[tuple[int, ...], tuple[int, int]] = {}
    for seed in seeds:
        for st in T.plan_steps(len(lengths), seed, s.epochs, s.batch_rows):
            for mb in T.micro_batches(st.rows, lengths, s.token_budget):
                seen[tuple(sorted(mb))] = (T.padded_tokens([mb], lengths),
                                           max(lengths[i] for i in mb))
    best = sorted(seen, key=lambda rows: (seen[rows], rows), reverse=True)[:k]
    return [sorted(rows, key=lambda i: (-lengths[i], i)) for rows in best]


def most_scored_micro_batch(lengths: list[int], scored: list[int], s: T.Settings,
                            seeds=(1, 2, 3)) -> list[int]:
    """The micro-batch with the most scored tokens over the seeds' plans (more token slots,
    then the rows, breaking ties), longest row first. The base arm's delta adds a
    full-vocabulary term per scored token (T9e review M1), so this one can peak higher than
    the largest by slots. Pure."""
    seen: dict[tuple[int, ...], tuple[int, int]] = {}
    for seed in seeds:
        for st in T.plan_steps(len(lengths), seed, s.epochs, s.batch_rows):
            for mb in T.micro_batches(st.rows, lengths, s.token_budget):
                seen[tuple(sorted(mb))] = (sum(scored[i] for i in mb),
                                           T.padded_tokens([mb], lengths))
    best = max(seen, key=lambda rows: (seen[rows], rows))
    return sorted(best, key=lambda i: (-lengths[i], i))


HEADROOM = 0.85  # checkpointing stays off only if a step's peak is at most this share
ALLOC_CONF = "expandable_segments:True"  # PYTORCH_CUDA_ALLOC_CONF for every sft-* command


def cuda_allocator(device) -> str | None:
    """Sets ``PYTORCH_CUDA_ALLOC_CONF`` to :data:`ALLOC_CONF` on CUDA unless it is set
    (before the first CUDA allocation: call it before a model loads); returns what is in
    effect, ``None`` off CUDA. main.toml also exports it for the whole run."""
    import os

    if device.type != "cuda":
        return None
    return os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF)


def memory_probe(s: T.Settings, records: list[dict], device, tokenizer: str | None = None,
                 embed_rows: tuple[str, ...] = ()) -> dict:
    """One training step as training runs it (forward and backward of the two largest
    micro-batches, gradients accumulated, then the AdamW step) without gradient
    checkpointing; checkpointing stays off for every run only if that step's peak is at
    most :data:`HEADROOM` of the device's memory (CUDA: a true peak; elsewhere there is
    no peak counter, so "ran without running out of memory"). Otherwise the same step
    with checkpointing, which must meet the same bar. A nondeterminism error is handled
    as training handles it (:func:`train.accumulate_or_relax`), recorded, and the strict
    setting restored after. ``tokenizer`` and ``embed_rows``: as training gets them (the
    base arm's delta is in the step). With ``embed_rows`` the step also runs the
    micro-batch with the most scored tokens (:func:`most_scored_micro_batch`) if it is not
    one of the two: the step's peak, which the gate reads, is then the largest of the three
    micro-batches' peaks. Without them (main) the step is main's, unchanged."""
    import torch
    from transformers import AutoModelForCausalLM

    from .device import peak_memory_bytes, preferred_dtype, reset_peak_memory, total_memory_bytes
    from .lora import inject_lora, inject_rows, trainable_parameters

    lengths = [r["n_tokens"] for r in records]
    mbs = largest_micro_batches(lengths, s)
    scored = [r["n_trained"] for r in records]
    by_scored = None
    if embed_rows:
        top = most_scored_micro_batch(lengths, scored, s)
        by_scored = next((j for j, mb in enumerate(mbs) if sorted(mb) == sorted(top)), None)
        if by_scored is None:
            mbs.append(top)
            by_scored = len(mbs) - 1
    rows = [i for mb in mbs for i in mb]
    tok = T.load_tokenizer(s.model, tokenizer)
    row_ids = T.token_ids(tok, embed_rows) if embed_rows else []
    enc = T.encode(tok, [records[i] for i in rows])
    local, k = [], 0
    for mb in mbs:  # positions in enc
        local.append(list(range(k, k + len(mb))))
        k += len(mb)
    n_scored = sum(enc.scored)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    total = total_memory_bytes(device)
    out = {"micro_batches": [{"rows": len(mb), "longest": max(lengths[i] for i in mb),
                              "token_slots": T.padded_tokens([mb], lengths),
                              **({"scored": sum(scored[i] for i in mb)}
                                 if by_scored is not None else {})} for mb in mbs],
           **({"most_scored": by_scored} if by_scored is not None else {}),
           "budget": s.token_budget, "device_bytes": total, "headroom": HEADROOM,
           "peak_rule": device.type == "cuda", "allocator": cuda_allocator(device),
           "attempts": [], "nondeterminism": ""}
    restore = (torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        for ckpt in (False, True):
            model = opt = None
            attempt = {"checkpointing": ckpt, "ran": False, "peak_bytes": None, "within": False}
            out["attempts"].append(attempt)
            try:
                model = AutoModelForCausalLM.from_pretrained(s.model,
                                                             dtype=preferred_dtype(device))
                model.to(device)
                inject_lora(model, s.targets, rank=s.rank, alpha=s.alpha,
                            generator=torch.Generator().manual_seed(1))
                if row_ids:
                    inject_rows(model, row_ids)
                if ckpt:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                model.train()
                opt = torch.optim.AdamW(trainable_parameters(model), lr=s.lr, betas=s.betas,
                                        eps=s.eps, weight_decay=s.weight_decay)
                reset_peak_memory(device)
                _, why = T.accumulate_or_relax(
                    model, opt, enc, local, n_scored, pad, device,
                    "the memory probe" + (" with checkpointing" if ckpt else ""))
                out["nondeterminism"] = out["nondeterminism"] or why
                opt.step()
                peak = peak_memory_bytes(device)
                opt.zero_grad(set_to_none=True)
                attempt.update({"ran": True, "peak_bytes": peak})
                attempt["within"] = (not out["peak_rule"]) or (
                    peak is not None and total is not None and peak <= HEADROOM * total)
            except torch.OutOfMemoryError:
                attempt["out_of_memory"] = True
            finally:
                del model, opt
                _free(device)
            if attempt["within"]:
                break
    finally:
        torch.use_deterministic_algorithms(restore[0], warn_only=restore[1])
    last = out["attempts"][-1]
    out["needs_checkpointing"] = last["checkpointing"]
    out["fits"] = last["within"]
    return out


def _print_memory(rep, mem: dict, model: str, device) -> None:
    """sft-check's memory section, from :func:`memory_probe`'s record."""
    gb = lambda b: "-" if b is None else f"{b / 1e9:.1f} GB"  # noqa: E731
    rep.kv("model", model)
    by_scored = mem.get("most_scored")
    for j, mb in enumerate(mem["micro_batches"], 1):
        rep.kv(f"micro-batch {j}", f"{mb['rows']} rows x {mb['longest']:,} tokens "
               f"= {mb['token_slots']:,} slots")
        if by_scored is not None:
            rep.kv("  scored tokens", f"{mb['scored']:,}"
                   + (" (the most of any micro-batch)" if j - 1 == by_scored else ""))
    rep.kv("token budget", f"{mem['budget']:,}")
    n = len(mem["micro_batches"])
    rep.kv("the step", ("both" if n == 2 else f"the {n}") + " micro-batches, then AdamW")
    if by_scored is not None:
        rep.note("Probed: the two with the most token slots and the one with the")
        rep.note("most scored tokens (the delta adds a full-vocabulary term per")
        rep.note("scored token). The step's peak is the largest of theirs.")
    rep.kv("device memory", gb(mem["device_bytes"]))
    if mem["peak_rule"]:
        rep.kv(f"limit ({mem['headroom']:.0%} of it)", gb(mem["headroom"] * mem["device_bytes"]))
    if mem["allocator"] is not None:
        rep.kv("PYTORCH_CUDA_ALLOC_CONF", mem["allocator"])
    peak_kind = {"cuda": "peak", "mps": "after the step"}.get(device.type, "not measured")
    rows = []
    for a in mem["attempts"]:
        result = ("out of memory" if a.get("out_of_memory") else
                  "yes" if a["within"] else "no")
        rows.append(["on" if a["checkpointing"] else "off",
                     gb(a["peak_bytes"]) if a["ran"] else "-", result])
    rep.table(["checkpointing", f"memory ({peak_kind})",
               "within the limit" if mem["peak_rule"] else "ran"], rows)
    rep.kv("gradient checkpointing", "needed: on" if mem["needs_checkpointing"] else
           "not needed: off")
    rep.kv("fits", "yes" if mem["fits"] else "no")
    if mem["nondeterminism"]:
        rep.note(mem["nondeterminism"], wrap=True)
    if mem["peak_rule"]:
        rep.note("Training reads this: checkpointing is off for every run only if")
        rep.note(f"this step's peak without it is at most {mem['headroom']:.0%} of the device.")
    else:
        rep.note(f"No peak counter on {device.type}: checkpointing is off if the step")
        rep.note(f"ran (the {mem['headroom']:.0%} rule applies on CUDA). Training reads this.")


# The merge check with a nonzero B (T8 review S3). B = 0 merges to W whatever the scaling,
# orientation or module, so it can't fail for the reasons a trained adapter's merge would.
MERGE_B_SCALE = 1e-2  # B = this x N(0, 1): an effect on the logits well above rounding
MERGE_B_SEED = 2  # A is drawn as training draws it (seed 1); B from its own generator
MERGE_PER_KIND = 2  # dev tasks of each kind; every position's logits is compared
MERGE_TOLERANCE = 1e-2  # relative gap, merged vs LoRA model, in float32
MERGE_CONTRAST = 1.25  # a merge with this x the scaling: what a wrong merge looks like


class LogitGap:
    """Merged vs LoRA-wrapped logits, pooled over prompts and positions. The relative gap
    is |merged - LoRA| / |LoRA - base| (Frobenius norms over every logit): the error as
    a share of the adapter's own effect, 0 for a right merge up to rounding, 1 for a
    merge that did nothing. Pure (tensors in, numbers out)."""

    def __init__(self) -> None:
        self.gap_sq = self.effect_sq = 0.0
        self.max_abs = 0.0
        self.argmax_same = self.positions = 0

    def add(self, merged, lora, base) -> None:
        d = (merged.float() - lora.float())
        self.gap_sq += float((d * d).sum())
        e = (lora.float() - base.float())
        self.effect_sq += float((e * e).sum())
        self.max_abs = max(self.max_abs, float(d.abs().max()))
        self.argmax_same += int((merged.argmax(-1) == lora.argmax(-1)).sum())
        self.positions += int(merged.shape[0])

    @property
    def relative(self) -> float:
        return (self.gap_sq / self.effect_sq) ** 0.5 if self.effect_sq else float("inf")


MERGE_ROWS_SEED = 3  # the embedding rows' delta (the base arm): MERGE_B_SCALE x N(0, 1)
ROWS_LEFT_OUT = "merged, rows left out"


def merge_check_random_b(s: T.Settings, model_id: str, texts: list[str], device,
                         tokenizer: str | None = None,
                         embed_rows: tuple[str, ...] = ()) -> dict:
    """An adapter with A drawn as training draws it and B = :data:`MERGE_B_SCALE` x N(0, 1),
    saved and merged with :func:`lora.merge_into` (the scoring path's merge), against the
    LoRA-wrapped model: logits at every position of ``texts``. In float32: bf16 rounding
    of W + (alpha/r) BA moves the logits by as much as a merge a fraction off in scale
    would, so no bf16 tolerance could separate them; the cast to the weights' dtype, the
    one step this leaves out, is what the B = 0 check runs. Contrasts: the merge at
    :data:`MERGE_CONTRAST` x the scaling, and the base model (a merge that did nothing).
    With ``embed_rows`` (the base arm) the LoRA model also has a nonzero row delta
    (:data:`MERGE_B_SCALE` x N(0, 1), seed :data:`MERGE_ROWS_SEED`), the merge writes it
    into the tied embedding, and one more contrast must fail: the merge without the rows."""
    import torch
    from transformers import AutoModelForCausalLM

    from . import lora as L
    from .lora import adapter_state, inject_lora, randomize_b

    def load():
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        m.to(device)
        m.eval()
        return m

    tok = T.load_tokenizer(model_id, tokenizer)
    lora = load()
    wrapped = inject_lora(lora, s.targets, rank=s.rank, alpha=s.alpha,
                          generator=torch.Generator().manual_seed(1))
    randomize_b(wrapped, MERGE_B_SCALE, torch.Generator().manual_seed(MERGE_B_SEED))
    rows = None
    if embed_rows:
        rows = L.inject_rows(lora, T.token_ids(tok, embed_rows))
        L.randomize_rows(rows, MERGE_B_SCALE, torch.Generator().manual_seed(MERGE_ROWS_SEED))
    state = adapter_state(wrapped, rows)
    merged, off = load(), load()
    n = L.merge_into(merged, state, s.scaling)
    L.merge_into(off, state, s.scaling * MERGE_CONTRAST)
    no_rows = None
    if rows is not None:
        no_rows = load()
        L.merge_into(no_rows, L.split_rows(state)[0], s.scaling)
    right, wrong, none, left = LogitGap(), LogitGap(), LogitGap(), LogitGap()
    try:
        with torch.no_grad():
            for text in texts:
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
                ids = ids.to(device)
                lo = lora(input_ids=ids).logits[0].float()
                for a in wrapped.values():
                    a.scaling = 0.0  # the LoRA path x 0: exactly the base model
                if rows is not None:
                    rows.enabled = False  # and the rows' delta off
                base = lora(input_ids=ids).logits[0].float()
                for a in wrapped.values():
                    a.scaling = s.scaling
                if rows is not None:
                    rows.enabled = True
                right.add(merged(input_ids=ids).logits[0], lo, base)
                wrong.add(off(input_ids=ids).logits[0], lo, base)
                if no_rows is not None:
                    left.add(no_rows(input_ids=ids).logits[0], lo, base)
                none.add(base, lo, base)
                del lo, base
    finally:
        del lora, wrapped, merged, off, no_rows
        _free(device)
    gaps = {"merged": right, f"merged, {MERGE_CONTRAST:g} x the scaling": wrong}
    if rows is not None:
        gaps[ROWS_LEFT_OUT] = left
    gaps["base (nothing merged)"] = none
    out = {"model": model_id, "prompts": len(texts), "modules": n, "dtype": "float32",
           "b_scale": MERGE_B_SCALE, "tolerance": MERGE_TOLERANCE,
           "rows": {k: {"relative": g.relative, "max_abs": g.max_abs,
                        "argmax_same": g.argmax_same, "positions": g.positions}
                    for k, g in gaps.items()},
           "passes": right.relative <= MERGE_TOLERANCE < wrong.relative and (
               rows is None or MERGE_TOLERANCE < left.relative)}
    if rows is not None:
        out["embed_rows"] = {"tokens": list(embed_rows), "ids": rows.ids.tolist(),
                             "seed": MERGE_ROWS_SEED}
    return out


# -- sft-check, the base arm (plan 3f) ----------------------------------------------------------


def rendered_text(record: dict, tok) -> str:
    """A training row's text as ``tok``'s chat template renders its messages (no tokens)."""
    from . import sft, toolace

    return toolace.assistant_spans(record["messages"],
                                   toolace.make_renderer(tok, sft.unwrap(record)))[0]


def same_prompts(a: list[str], b: list[str]) -> int:
    """How many of two prompt lists are byte-identical, position by position. Pure."""
    if len(a) != len(b):
        raise ValueError("two prompt lists of different lengths")
    return sum(x == y for x, y in zip(a, b, strict=True))


def base_prompts_check(m: Mode, real: list[dict]) -> dict:
    """The base arm's prompts, rendered as it renders them (the profile's tokenizer, loaded
    the way scoring and training load it), against the Instruct arm's (the main profile's
    tokenizer for the same mode, loaded by name) and, as the contrast, against the base
    model's own chat template. Every training row (against sft-data's text, which the
    Instruct arm trained on), our test, dev and train-sample tasks, BFCL's eval items."""
    from transformers import AutoTokenizer

    from . import bfcl, ours, progress, tasks
    from .cli import _load_or_make_tasks

    model = m.settings.model
    arm = T.load_tokenizer(model, m.tokenizer)  # raises unless it splits text as the model's
    main = T.PROFILES["main"]
    instruct_id = main.tokenizer_for(main.settings(m.quick).model)
    instruct = AutoTokenizer.from_pretrained(instruct_id)
    own = AutoTokenizer.from_pretrained(model)
    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    sets = []
    progress.status("base prompts: training rows")
    sets.append(("training rows (real.jsonl)", [r["text"] for r in real],
                 [rendered_text(r, arm) for r in real], [rendered_text(r, own) for r in real]))
    for split in ("test", "dev", "train-sample"):
        ts = sorted(splits[split], key=lambda t: t["id"])
        sets.append((f"our {split} tasks", [ours.prompt(instruct, t) for t in ts],
                     [ours.prompt(arm, t) for t in ts], [ours.prompt(own, t) for t in ts]))
    items = [it for c in bfcl.CATEGORIES for it in bfcl.load_items(c, None, "eval")]
    progress.status("base prompts: BFCL eval items")
    sets.append(("BFCL eval items", [bfcl.prompt_text(instruct, it) for it in items],
                 [bfcl.prompt_text(arm, it) for it in items],
                 [bfcl.prompt_text(own, it) for it in items]))
    rows = [{"set": name, "n": len(want), "as_instruct": same_prompts(want, got),
             "with_own_template": same_prompts(want, base)} for name, want, got, base in sets]
    return {"model": model, "tokenizer": m.tokenizer, "instruct_tokenizer": instruct_id,
            "splits_as_model": T.tokenizers_match(arm, own), "vocabulary": len(arm),
            "eos": {"model's own": own.eos_token, "this arm's": arm.eos_token},
            "sets": rows,
            "passes": T.tokenizers_match(arm, own) and all(r["as_instruct"] == r["n"]
                                                           for r in rows)}


def _print_base_prompts(rep, pr: dict) -> None:
    rep.section("The base arm's prompts: the Instruct chat template")
    rep.kv("model", pr["model"])
    rep.kv("tokenizer and chat template", pr["tokenizer"])
    rep.kv("  splits text as the model's own", pr["splits_as_model"])
    rep.note("  (vocabulary, merges, normalizer, added tokens: the same ids for")
    own, arm = pr["eos"]["model's own"], pr["eos"]["this arm's"]
    rep.note(f"  every text; {pr['vocabulary']:,} tokens; the eos name differs:")
    rep.note(f"  {own} vs {arm}; where replies end is set below)")
    rep.note("Each prompt rendered as this arm renders it, against the Instruct")
    rep.note("arm's (training rows: sft-data's text); beside it, the base model's")
    rep.note("own chat template, whose default system line differs:")
    rep.table(["prompts", "n", "as the Instruct arm's", "own template"],
              [[r["set"], r["n"], r["as_instruct"], r["with_own_template"]]
               for r in pr["sets"]])
    rep.kv("all as the Instruct arm's", pr["passes"])


def stops_check(tok, lm, tokens: tuple[str, ...]) -> dict:
    """What generation stops at for this arm: the model's own generation_config, then the
    engine's stop set after :func:`set_stops` (``engine.decoding``, what scoring reads)."""
    from transformers import GenerationConfig

    from . import bfcl
    from .engine import decoding

    own = GenerationConfig.from_pretrained(lm.name_or_path).eos_token_id
    own = own if isinstance(own, list) else [own]
    want = T.token_ids(tok, tokens)
    set_stops(lm, tok, tokens)
    stop = list(decoding(tok, lm, 1, 0.0, bfcl.MAX_NEW_TOKENS).stop)
    # keyed by str: checks.json round-trips it
    names = {str(i): tok.convert_ids_to_tokens(i) for i in [*own, *want, *stop] if i is not None}
    return {"model_eos": own, "wanted": want, "engine_stop": stop, "names": names,
            "passes": set(stop) == set(want)}


def _print_stops(rep, st: dict) -> None:
    rep.section("Where the base arm's replies end")
    cols = st["wanted"]
    rep.table(["a reply ends at", *[st["names"][str(i)] for i in cols]],
              [[label, *[i in ids for i in cols]]
               for label, ids in (("the model's own generation_config", st["model_eos"]),
                                  ("as loaded for this arm", st["wanted"]),
                                  ("the engine (what scoring reads)", st["engine_stop"]))])
    rep.kv("token ids", " and ".join(str(i) for i in cols))
    rep.kv("the engine stops at both", st["passes"])
    rep.note("The base model's config stops only at <|endoftext|>; the engine adds")
    rep.note("<|im_end|>, and the profile sets both on every model it loads, so a")
    rep.note("reply ends at the end of its turn whatever the config says.")


ROW_IDENTICAL = 0.9999  # pairwise cosine at or above this: the rows are one direction


def hub_rows(model_id: str, ids: list[int]) -> "torch.Tensor":  # noqa: F821
    """Rows of a model's input embedding read from its safetensors file (no model loaded;
    downloaded if not cached), float32."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    key = "model.embed_tokens.weight"
    try:
        index = json.loads(Path(hf_hub_download(model_id, "model.safetensors.index.json"))
                           .read_text())
        name = index["weight_map"][key]
    except Exception:  # noqa: BLE001 - no index: one file
        name = "model.safetensors"
    with safe_open(hf_hub_download(model_id, name), framework="pt") as f:
        sl = f.get_slice(key)
        return torch.stack([sl[i:i + 1][0] for i in ids]).float()


def rows_like(w, i: int, threshold: float = ROW_IDENTICAL, chunk: int = 16384) -> list[int]:
    """Ids of every row of ``w`` whose cosine with row ``i`` is at least ``threshold`` (row
    ``i`` included), in float32, a chunk of rows at a time. Pure."""
    import torch

    v = w[i].detach().float()
    v = v / torch.linalg.vector_norm(v)
    out = []
    for a in range(0, len(w), chunk):
        r = w[a:a + chunk].detach().float()
        cos = (r @ v) / torch.linalg.vector_norm(r, dim=1).clamp_min(1e-30)
        out += [a + int(j) for j in torch.nonzero(cos >= threshold).flatten().cpu()]
    return out


def id_runs(ids: list[int]) -> str:
    """Sorted ids as runs: ``[1, 2, 3, 7]`` -> ``"1-3, 7"``. Pure."""
    runs: list[list[int]] = []
    for i in sorted(ids):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def id_summary(ids: list[int], max_runs: int = 3) -> str:
    """:func:`id_runs` if there are at most ``max_runs`` runs; otherwise the longest run
    and how many ids are outside it, and their range. Pure."""
    full = id_runs(ids)
    runs = full.split(", ")
    if len(runs) <= max_runs:
        return full
    size = lambda r: int(r.split("-")[1]) - int(r.split("-")[0]) + 1 if "-" in r else 1  # noqa: E731
    big = max(runs, key=lambda r: (size(r), -runs.index(r)))
    a, b = (int(x) for x in (big.split("-") if "-" in big else (big, big)))
    rest = [i for i in ids if not a <= i <= b]
    return f"{big} and {len(rest):,} more from {min(rest)} to {max(rest)}"


def id_kinds(ids: list[int], vocab_size: int, added: set[int]) -> dict[str, int]:
    """How many of ``ids`` are added tokens, ordinary vocabulary tokens, and padding rows
    (an embedding row with no token: id at or past the tokenizer's ``len``). Pure."""
    pad = sum(i >= vocab_size for i in ids)
    add = sum(i < vocab_size and i in added for i in ids)
    return {"added tokens": add, "vocabulary tokens": len(ids) - pad - add, "padding rows": pad}


def rows_check(lm, model: str, tokens: tuple[str, ...], ids: list[int],
               instruct: str, tok=None) -> dict:
    """The four rows in the loaded base model (as it scores: the device's dtype) beside the
    Instruct model's: norms, cosines, largest difference; and the median row norm. With
    ``tok``: every row of the embedding identical to the first token's (T9e review S2), and
    what those ids are."""
    import torch

    from .device import dtype_name
    from .lora import row_similarity, tied

    w = lm.get_input_embeddings().weight
    base = torch.stack([w[i].detach().float().cpu() for i in ids])
    median = float(torch.linalg.vector_norm(w.detach(), dim=1, dtype=torch.float32)
                   .median().cpu())
    b, ins = row_similarity(base), row_similarity(hub_rows(instruct, ids))
    same = {}
    if tok is not None:
        like = rows_like(w, ids[0])
        same = {"same_as_first": {"n": len(like), "ids": id_summary(like),
                                  "kinds": id_kinds(like, len(tok),
                                                    set(tok.added_tokens_decoder)),
                                  "trained": sum(i in like for i in ids)}}
    return {**same, "model": model, "instruct": instruct, "tokens": list(tokens), "ids": ids,
            "tied": tied(lm), "rows": len(w), "hidden": int(w.shape[1]),
            "dtype": dtype_name(w.dtype),
            "median_norm": median, "base": b, "instruct_rows": ins, "threshold": ROW_IDENTICAL,
            "identical": b["cos_min"] >= ROW_IDENTICAL,
            "passes": tied(lm) and b["cos_min"] >= ROW_IDENTICAL}


def _print_rows(rep, rw: dict) -> None:
    b, ins = rw["base"], rw["instruct_rows"]
    short = rw["instruct"].split("/")[-1]
    rep.section("The four embedding rows: why the base arm trains them")
    rep.kv("model", rw["model"])
    rep.kv("embeddings tied (input = output)", rw["tied"])
    rep.note(f"Each row's norm and its cosine with {rw['tokens'][0]}, in the base")
    rep.note(f"model ({rw['dtype']}, as loaded) and in {short}:")
    rep.table(["token", "id", "norm", "cosine", "Instruct norm", "cosine"],
              [[t, i, f"{b['norms'][j]:.3f}", f"{b['cos_first'][j]:.4f}",
                f"{ins['norms'][j]:.3f}", f"{ins['cos_first'][j]:.4f}"]
               for j, (t, i) in enumerate(zip(rw["tokens"], rw["ids"], strict=True))])
    rep.kv("median row norm, base", f"{rw['median_norm']:.3f} (all {rw['rows']:,} rows)")
    rep.kv("pairwise cosine, base", f"{b['cos_min']:.4f} to {b['cos_max']:.4f}")
    rep.kv("  largest |difference|, base", f"{b['max_abs_diff']:.1e}")
    rep.kv("pairwise cosine, Instruct", f"{ins['cos_min']:.4f} to {ins['cos_max']:.4f}")
    rep.kv("identical in the base model", rw["identical"])
    rep.note(f"(identical: every pairwise cosine at least {rw['threshold']:g})")
    sf = rw.get("same_as_first")
    if sf:
        kinds = " + ".join(f"{v:,} {k}" for k, v in sf["kinds"].items() if v)
        rep.note(f"rows sharing {rw['tokens'][0]}'s embedding: {sf['n']:,}")
        rep.note(f"  ({kinds}; ids {sf['ids']})", wrap=True)
        rep.note(f"(every row of the embedding, cosine at least {rw['threshold']:g};"
                 f" the {sf['trained']} above among them)", wrap=True)
        rep.note(f"Only the {len(rw['ids'])} the format uses are trained; the others stay "
                 "frozen.", wrap=True)
    rep.note("Tied and identical: from any hidden state, ending the turn and")
    rep.note("opening a call score the same, and LoRA can't separate them. The")
    rep.note(f"base arm trains a delta on these rows: {len(rw['ids'])} x {rw['hidden']:,} = "
             f"{len(rw['ids']) * rw['hidden']:,} float32")
    rep.note("parameters, beside LoRA, in the same optimizer.")



def run_check(root: Path, quick: bool, model: str | None, dev_limit: int | None,
              batch_size: int | None, profile: str = "main") -> int:
    from transformers import AutoTokenizer

    from . import bfcl, ours, progress, tasks
    from .cli import SFT_DIR, _block, _load_or_make_tasks, _quiet_hf
    from .device import get_device, set_determinism
    from .engine import default_batch_size
    from .lora import ATTENTION_MLP, adapter_state, inject_lora, inject_rows, save_adapter, tied
    from .milestone0 import token_mask
    from .ours import pct
    from .report import Report

    _quiet_hf()
    m = mode(quick, root, profile=profile)
    prof = T.PROFILES[m.profile]
    model = model or m.settings.model
    model_tok = prof.tokenizer_for(model) if prof.tokenizer is not None else None
    dev_limit = dev_limit if dev_limit is not None else m.dev_limit
    device = get_device()
    determinism = set_determinism(device)
    cuda_allocator(device)  # before any model loads
    batch = batch_size or default_batch_size(device.type)
    m.out.mkdir(parents=True, exist_ok=True)
    rep = Report("tinytools sft-check" + m.title
                 + (" --quick (not for publication)" if quick else ""),
                 out=root / "outputs" / f"sft-check{m.suffix}.txt")
    rep.header()
    checks: dict = {}

    # 1. the template: the training model's tokenizer against sft-data's rendering (the base
    # arm: the Instruct tokenizer it renders with)
    main_model = prof.tokenizer_for(prof.full.model)
    tok = AutoTokenizer.from_pretrained(main_model)
    capture = root / "outputs" / prof.data_capture
    made_with = "?"
    if capture.exists():
        line = next((x for x in capture.read_text().splitlines()
                     if x.strip().startswith("tokenizer (chat template)")), "")
        made_with = line.split()[-1] if line else "?"
    rep.section(f"The chat template: {main_model} vs sft-data's rendering")
    rep.kv("sft-data rendered with", made_with)
    rep.note("Each training row re-rendered with this model's tokenizer and chat")
    rep.note("template: text, scored spans, tokens and scored tokens compared.")
    want_pins = profile_pins(m.real_set)  # the profile's sets only (not every file here)
    files = sorted((SFT_DIR / n for n in want_pins if (SFT_DIR / n).exists()),
                   key=lambda p: (p.name != m.real_set, p.name))
    rows_t = []
    template = {}
    real: list[dict] = []
    for f in files:
        recs = T.load_set(f, None)
        if f.name == m.real_set:
            real = recs
        same, differ = template_identity(recs, tok)
        template[f.name] = {"rows": len(recs), "identical": same, "differ": differ[:20]}
        rows_t.append([str(f.relative_to(root)), len(recs), same, len(differ)])
        progress.status(f"template check: {f.name}: {same} of {len(recs)} identical")
    rep.table(["file", "rows", "identical", "differ"], rows_t)
    ok = bool(files) and all(v["identical"] == v["rows"] for v in template.values())
    rep.kv("byte-identical", ok)
    checks["template"] = {"model": main_model, "rendered_with": made_with, "files": template,
                          "identical": ok}

    # 1b. the sets are the Mac's bytes (T8 review S4)
    rep.section("The SFT sets: sha256 against the Mac's")
    pins = set_pins(SFT_DIR, want_pins)
    rep.table(["file", "sha256 (first 16)", "as pinned"],
              [[f"data/sft/{name}", (h or "missing")[:16], h == want]
               for name, (h, want) in pins.items()])
    pinned = all(h == want for h, want in pins.values())
    rep.kv("all four as pinned (sft.py)" if len(pins) == 4 else "as pinned (sft.py)", pinned)
    checks["sha256"] = {name: {"sha256": h, "pinned": want} for name, (h, want) in pins.items()}

    # 1c. the base arm: every prompt and training row as the Instruct arm's, byte for byte
    prompts_ok = True
    if m.tokenizer is not None:
        pr = base_prompts_check(m, real)
        _print_base_prompts(rep, pr)
        checks["base_prompts"] = pr
        prompts_ok = pr["passes"]

    # 2. lengths, scored tokens, <|im_end|>
    s = m.settings
    rep.section("Row lengths (tokens), the real arm")
    lengths = [r["n_tokens"] for r in real]
    census = lengths_census(lengths, s.max_length)
    rep.kv("rows", census["rows"])
    rep.kv("longest", census["longest"])
    rep.kv("mean", f"{census['mean']:.0f}")
    rep.kv("longer than 1,024", census["over 1024"])
    rep.kv(f"longer than {s.max_length:,} (cut: none)", census[f"over {s.max_length}"])
    scored = sum(r["n_trained"] for r in real)
    rep.kv("tokens", sum(lengths))
    rep.kv("  scored", f"{scored} ({100 * scored / sum(lengths):.2f}%)")
    rep.kv("  masked", sum(lengths) - scored)
    turns = closed = 0
    for r in real:
        enc = tok(r["text"], return_offsets_mapping=True, add_special_tokens=False)
        offs = [tuple(o) for o in enc["offset_mapping"]]
        spans = [tuple(x) for x in r["train_spans"]]
        t, k = im_end_scored(r["text"], offs, token_mask(offs, spans), spans)
        turns += t
        closed += k
    rep.kv("assistant turns", turns)
    rep.kv("  ending in a scored <|im_end|>", closed)
    checks["lengths"] = census | {"tokens": sum(lengths), "scored": scored,
                                  "assistant_turns": turns, "im_end_scored": closed}

    # 2b. memory: one training step on the two largest micro-batches (T8 review B1)
    rep.section("Memory: one training step, the two largest micro-batches" if not m.embed_rows
                else "Memory: one training step, the largest micro-batches")
    mem = memory_probe(s, real if s.rows is None else real[:s.rows], device, m.tokenizer,
                       m.embed_rows)
    _print_memory(rep, mem, s.model, device)
    checks["memory"] = mem

    # 3. the system prompt's shape: training rows, our tasks, BFCL
    rep.section("The system prompt's shape: training rows vs our tasks and BFCL")
    rep.note("The system turn's lines before its tools section, as the model")
    rep.note("reads them.")
    with_line = [r for r in real if len(system_shape(r["text"])) > 1]
    shapes = {}
    examples = [("training row with a date line", with_line[0]["text"] if with_line else None),
                ("training row without one", next((r["text"] for r in real
                                                   if len(system_shape(r["text"])) == 1), None))]
    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    examples.append(("our test task", ours.prompt(tok, sorted(splits["test"],
                                                              key=lambda t: t["id"])[0])))
    item = bfcl.load_items("simple", 1, "eval")[0]
    examples.append(("BFCL eval item", bfcl.prompt_text(tok, item)))
    for label, text in examples:
        if text is None:
            continue
        lines = system_shape(text)
        shapes[label] = lines
        rep.note(f"{label}:")
        for line in lines:
            _block(rep, line, indent=6)
    rep.kv("training rows with a date line", pct(len(with_line), len(real)))
    rep.note("Our tasks state the frozen clock as ToolACE's rows state theirs;")
    rep.note("BFCL's prompts have Qwen's default sentence alone.")
    checks["system_shapes"] = shapes
    checks["rows_with_date_line"] = [len(with_line), len(real)]

    # 3b. the base arm: where generation stops, and the four rows (the reason for the delta)
    tok_m, lm = bfcl.load_model(model, device, model_tok)
    stops_ok = rows_ok = True
    row_ids: list[int] = []
    if m.stop_tokens:
        st = stops_check(tok_m, lm, m.stop_tokens)
        _print_stops(rep, st)
        checks["stops"] = st
        stops_ok = st["passes"]
    if m.embed_rows:
        row_ids = T.token_ids(tok_m, m.embed_rows)
        rw = rows_check(lm, model, m.embed_rows, row_ids, prof.tokenizer_for(model), tok_m)
        _print_rows(rep, rw)
        checks["embed_rows"] = rw
        rows_ok = rw["passes"]

    # 4. the merge check: an untrained adapter (B = 0) merged gives the base model's replies
    rep.section("Merging an untrained adapter (B = 0): replies vs the base model")
    dev = ours.limit_per_kind(splits["dev"], dev_limit)
    rep.kv("model", model)
    rep.kv("tasks", f"{len(dev)} of our dev tasks, greedy")
    if len(dev) < len(splits["dev"]):
        rep.note(f"(the first {dev_limit} of each kind, of {len(splits['dev'])})")
    rep.kv("batch size", batch)
    rep.kv("determinism", determinism, wrap=True)
    base = ours.score(tok_m, lm, dev, samples=1, temperature=0.0, seed=SEED, batch_size=batch,
                      on_item=progress.items("merge check: base model", len(dev)))
    import torch

    wrapped = inject_lora(lm, ATTENTION_MLP, rank=s.rank, alpha=s.alpha,
                          generator=torch.Generator().manual_seed(1))
    rows0 = inject_rows(lm, row_ids) if row_ids else None  # delta = 0
    b0 = m.out / "b0-adapter.safetensors"  # not under a run's directory: not fetched
    save_adapter(b0, adapter_state(wrapped, rows0),
                 {"base_model": model, "arm": "untrained (B = 0)", "seed": "1",
                  "rank": str(s.rank), "alpha": str(s.alpha), "targets": ",".join(ATTENTION_MLP)})
    tok2, merged, _, n_mod = load_merged(model, b0, device, model_tok)
    set_stops(merged, tok2, m.stop_tokens)
    same_w = sum(torch.equal(merged.get_submodule(n).weight, a.base.weight)
                 for n, a in wrapped.items())
    same_rows = tied_after = None
    if rows0 is not None:
        w0, w1 = lm.get_input_embeddings().weight, merged.get_input_embeddings().weight
        same_rows = sum(torch.equal(w0[i], w1[i]) for i in row_ids)
        tied_after = tied(merged)
    del lm, wrapped, rows0
    _free(device)
    again = ours.score(tok2, merged, dev, samples=1, temperature=0.0, seed=SEED,
                       batch_size=batch, on_item=progress.items("merge check: merged B = 0",
                                                                len(dev)))
    del merged
    _free(device)
    identical = sum(a.replies[0] == b.replies[0] for a, b in zip(base, again, strict=True))
    rep.kv("modules merged", n_mod)
    rep.kv("  weights equal to the base's", f"{same_w} of {n_mod}")
    if same_rows is not None:
        rep.kv("embedding rows merged (delta = 0)", len(row_ids))
        rep.kv("  rows equal to the base's", f"{same_rows} of {len(row_ids)}")
        rep.kv("  output head still the embedding", tied_after)
    rep.kv("replies byte-identical", f"{identical} of {len(dev)}")
    rep.note("The adapter: A drawn as training draws it (seed 1), B = 0. Merging")
    rep.note("computes W + (alpha/r) B A in float32 and casts it to the weight's")
    rep.note("dtype once, the path every trained adapter is scored through.")
    if same_rows is not None:
        rep.note("The rows: E + delta, the same way, into the tied embedding.")
    checks["merge_b0"] = {"model": model, "tasks": len(dev), "identical": identical,
                          "modules": n_mod, "weights_equal": same_w, "batch_size": batch,
                          "device": device.type}
    if same_rows is not None:
        checks["merge_b0"] |= {"rows": len(row_ids), "rows_equal": same_rows,
                               "tied": tied_after}

    # 4b. the merge check with a nonzero B: logits, merged vs the LoRA model (T8 review S3)
    rep.section("Merging an adapter with random B: logits vs the LoRA model")
    few = ours.limit_per_kind(splits["dev"], MERGE_PER_KIND)
    mg = merge_check_random_b(s, model, [ours.prompt(tok_m, t) for t in few], device,
                              model_tok, m.embed_rows)
    rep.kv("model", model)
    rep.kv("prompts", f"{len(few)} dev tasks, {MERGE_PER_KIND} per kind")
    rep.kv("compared", "the logits at every position")
    rep.kv("adapter", f"A: seed 1; B = {MERGE_B_SCALE:g} x N(0, 1), seed {MERGE_B_SEED}")
    if "embed_rows" in mg:
        rep.kv("  embedding rows", f"delta = {MERGE_B_SCALE:g} x N(0, 1), seed {MERGE_ROWS_SEED}")
    rep.kv("dtype", "float32, all " + ("four" if "embed_rows" in mg else "three") + " models")
    rep.kv("modules merged", mg["modules"])
    rep.table(["vs the LoRA model", "relative gap", "max |diff|", "argmax same"],
              [[k, f"{v['relative']:.2e}", f"{v['max_abs']:.2e}",
                f"{v['argmax_same']} of {v['positions']}"] for k, v in mg["rows"].items()])
    rep.kv("tolerance (relative gap)", f"{MERGE_TOLERANCE:g}")
    rep.kv("passes", mg["passes"])
    rep.note("relative gap: |merged - LoRA| / |LoRA - base| over every logit: the")
    rep.note("error as a share of the adapter's effect (base: a merge that did")
    rep.note("nothing, 1). Passes: the merge within the tolerance and the")
    rep.note("mis-scaled merge outside it. In float32, so rounding is the only")
    rep.note("gap; the cast to the weights' dtype is what the B = 0 check runs.")
    if "embed_rows" in mg:
        rep.note("With the embedding rows: the merge without them must fail too.")
    checks["merge_random_b"] = mg
    T.write_json(m.out / "checks.json", checks)
    from dataclasses import asdict

    from .device import dtype_name, preferred_dtype

    T.write_json(m.out / "setup.json", {
        "seeds": [1, 2, 3], "determinism": determinism,
        "dtype": dtype_name(preferred_dtype(device)), "model": s.model,
        "training": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(s).items()},
        "gradient_checkpointing": mem["needs_checkpointing"],
        "cuda_alloc_conf": mem["allocator"],
        "generation": {"seed": SEED, "samples": SAMPLES, "temperature": TEMPERATURE,
                       "batch_size": batch},
        **({"profile": m.profile, "tokenizer": m.tokenizer, "embed_rows": list(m.embed_rows),
            "stop_tokens": list(m.stop_tokens)} if m.profile != "main" else {}),
        **({"real_set": m.real_set} if m.real_set != "real.jsonl" else {})})
    passed = ok and pinned and census[f"over {s.max_length}"] == 0 and closed == turns \
        and identical == len(dev) and same_w == n_mod and mem["fits"] and mg["passes"] \
        and prompts_ok and stops_ok and rows_ok and same_rows in (None, len(row_ids)) \
        and tied_after in (None, True)
    rep.blank()
    rep.kv("all checks pass", passed)
    rep.close()
    return 0 if passed else 1


# -- sft-score, sft-train, sft-pair --------------------------------------------------------


def _print_score(rep, info: dict, d: Path) -> None:
    """A short summary of one model's scoring (the numbers the report compares)."""
    from .classify import is_right

    rows = [json.loads(x) for x in (d / OURS_FILE).read_text(encoding="utf-8").split("\n") if x]
    bf = [json.loads(x) for x in (d / BFCL_FILE).read_text(encoding="utf-8").split("\n") if x]
    sm = [json.loads(x) for x in (d / SAMPLE_FILE).read_text(encoding="utf-8").split("\n") if x]
    rep.kv("ours test, greedy, right", f"{sum(is_right(r['outcome']) for r in rows)} of {len(rows)}")
    rep.kv("BFCL eval, greedy, right", f"{sum(is_right(r['outcome']) for r in bf)} of {len(bf)}")
    g = [r for r in sm if r["pass"] == "greedy"]
    rep.kv("train-sample, greedy, right", f"{sum(is_right(r['outcome']) for r in g)} of {len(g)}")
    for name, p in info["passes"].items():
        rep.kv(f"  {name[:31]}", f"{p['items']} items, {p['seconds'] / 60:.1f} min")
    rep.note("(per arm, outcome by outcome and with intervals: sft-report)")


def _score_one(root: Path, m: Mode, label: str, adapter: Path | None, device, batch: int,
               rep) -> dict:
    from . import progress

    setup = _setup(device, batch)
    if adapter is None:
        progress.status(f"loading {m.settings.model} on {device}")
        tok, lm = load_for(m, device)
        extra = {"adapter": None}
    else:
        progress.status(f"loading {m.settings.model} + {adapter.parent.name}, merged, on {device}")
        tok, lm, meta, n = load_for(m, device, adapter)
        from .lora import load_adapter

        extra = {"adapter": str(adapter.relative_to(root)), "modules_merged": n,
                 "adapter_sha256": T.tensors_sha256(load_adapter(adapter)[0]),
                 "adapter_meta": meta}
    try:
        info = score_model(tok, lm, label, m, batch, setup, extra)
    finally:
        del lm
        _free(device)
    rep.section(f"Scored: {label}")
    rep.kv("model", m.settings.model)
    rep.kv("adapter", "none" if adapter is None else "merged into the weights")
    if adapter is not None:
        rep.kv("adapter tensors sha256 (first 16)", extra["adapter_sha256"][:16])
    _print_score(rep, info, m.out / label)
    return info


def _header(rep, m: Mode, device, determinism: str, batch: int, deadline: float | None,
            gen_seed: int = SEED, floor: bool = False) -> None:
    from . import bfcl
    from .engine import print_setup

    rep.header()
    rep.section("Setup")
    rep.kv("model", m.settings.model)
    _profile_lines(rep, m)
    rep.kv("output directory", str(m.out.name))
    rep.kv("decoding", f"{SAMPLES} at T={TEMPERATURE:g} (train-sample only)" if floor else
           f"greedy; {SAMPLES} at T={TEMPERATURE:g} (train-sample)")
    rep.kv("top_p / top_k / repetition penalty", "1.0 / off / 1.0")
    rep.kv("max new tokens", bfcl.MAX_NEW_TOKENS)
    rep.kv("seed (generation)", f"{gen_seed} (+ task id and sample number)")
    print_setup(rep, batch, (1, SAMPLES))
    rep.kv("determinism", determinism, wrap=True)
    if device.type == "cuda":
        import os

        rep.kv("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set"))
    rep.kv("deadline", "none" if deadline is None else
           time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(deadline)))


def _profile_lines(rep, m: Mode) -> None:
    """The Setup lines a profile other than main adds (main's Setup is unchanged)."""
    if m.profile == "main":
        return
    rep.kv("profile", m.profile)
    if m.real_set != "real.jsonl":
        rep.kv("  training set", f"data/sft/{m.real_set}")
        rep.kv("  arms", ", ".join(m.arms))
    if m.tokenizer is not None:
        rep.kv("  chat template and tokenizer", m.tokenizer)
    if m.stop_tokens:
        rep.kv("  replies end at", ", ".join(m.stop_tokens))
    if m.embed_rows:
        rep.kv("  embedding rows trained", len(m.embed_rows))
        rep.note("    " + " ".join(m.embed_rows))


def _context(root: Path, quick: bool, batch_size: int | None, token_budget: int | None = None,
             checkpointing: bool = False, profile: str = "main"):
    from dataclasses import replace

    from .cli import _quiet_hf
    from .device import get_device, set_determinism
    from .engine import default_batch_size

    _quiet_hf()
    base = T.PROFILES[profile].settings(quick)
    m0 = mode(quick, root, base, profile)
    checks = m0.out / "checks.json"
    if not checkpointing and token_budget is None and checks.exists():
        # sft-check measured the largest micro-batch on this machine: its verdict decides
        checkpointing = json.loads(checks.read_text()).get("memory", {}).get(
            "needs_checkpointing", False)
    s = replace(base, token_budget=token_budget or base.token_budget,
                grad_checkpointing=checkpointing)
    m = mode(quick, root, s, profile)
    device = get_device()
    determinism = set_determinism(device)  # before any model loads (cuBLAS workspace)
    cuda_allocator(device)  # likewise (the CUDA caching allocator reads it once)
    return m, device, determinism, batch_size or default_batch_size(device.type)


def run_score(root: Path, quick: bool, adapter: Path | None, batch_size: int | None,
              floor: bool = False, profile: str = "main") -> int:
    """``floor``: the untrained model's train-sample samples again with generation seed
    :data:`FLOOR_SEED` (r6a's noise floor), into ``<out>/FLOOR/``."""
    from . import progress
    from .report import Report

    if floor and adapter is not None:
        raise ValueError("the noise floor is the untrained model's")
    if adapter is None and T.PROFILES[profile].variant is not None:
        sys.exit(f"--profile {profile} scores trained adapters only: its untrained model is "
                 "the main run's, read by sft-report (plan 3g)")
    m, device, determinism, batch = _context(root, quick, batch_size, profile=profile)
    label = FLOOR if floor else UNTRAINED if adapter is None else adapter.parent.name
    rep = Report(f"tinytools sft-score ({label})" + m.title + (" --quick" if quick else ""),
                 out=root / "outputs" / f"sft-score-{label}{m.suffix}.txt")
    _header(rep, m, device, determinism, batch, None,
            gen_seed=FLOOR_SEED if floor else SEED, floor=floor)
    if floor:
        progress.status(f"loading {m.settings.model} on {device}")
        tok, lm = load_for(m, device)
        try:
            info = score_floor(tok, lm, m, batch, _setup(device, batch))
        finally:
            del lm
            _free(device)
        _print_floor(rep, info, m.out / FLOOR)
        rep.close()
        return 0
    _score_one(root, m, label, adapter, device, batch, rep)
    if adapter is None:
        print_copies(rep, standard_copies(m, root))
    rep.close()
    return 0


def _print_floor(rep, info: dict, d: Path) -> None:
    """The noise floor's summary: right of the samples, and the groups of 8."""
    from . import effects as E
    from .classify import is_right

    rows = [json.loads(x) for x in (d / SAMPLE_FILE).read_text(encoding="utf-8").split("\n") if x]
    right: dict = {}
    for r in rows:
        right[r["id"]] = right.get(r["id"], 0) + is_right(r["outcome"])
    rep.section(f"Scored: {FLOOR} (r6a's noise floor)")
    rep.kv("model", f"{info['model']}, untrained")
    rep.kv(f"train-sample, {SAMPLES} at T={TEMPERATURE:g}, right",
           f"{sum(right.values())} of {len(rows)}")
    groups = [E.group(v, SAMPLES) for v in right.values()]
    rep.kv(" / ".join(E.GROUPS), " / ".join(str(groups.count(g)) for g in E.GROUPS)
           + " tasks")
    for name, p in info["passes"].items():
        rep.kv(f"  {name[:31]}", f"{p['items']} items, {p['seconds'] / 60:.1f} min")
    rep.note(f"Generation seed {FLOOR_SEED} where the untrained scoring used {SEED}: the same")
    rep.note("model drawn again. sft-report reads r6a's movements beyond it.")


def _train_one(root: Path, m: Mode, arm: str, seed: int, device, rep,
               repeat: bool = False) -> T.Result:
    label = run_label(arm, seed, repeat)
    res = T.train(m.settings, arm, seed, set_path(root, arm, seed, m.real_set), m.out / label,
                  device, real_path=None if arm == "real" else set_path(root, "real", seed,
                                                                        m.real_set),
                  label=f"{label}, {m.settings.model.split('/')[-1]}",
                  tokenizer=m.tokenizer, embed_rows=m.embed_rows)
    _free(device)
    rep.section(f"Trained: {label}")
    rep.kv("rows / steps", f"{res.rows} / {res.steps} ({res.micro_batches} micro-batches)")
    rep.kv("loss, first / last step", f"{res.losses[0]:.4f} / {res.losses[-1]:.4f}")
    rep.kv("wall time", f"{res.seconds / 60:.1f} min")
    rep.kv("adapter tensors sha256 (first 16)", res.adapter_sha256[:16])
    if res.embed_rows is not None:
        _print_trained_rows(rep, res.embed_rows)
    if res.note:
        rep.note(res.note, wrap=True)
    return res


def _print_trained_rows(rep, r: dict) -> None:
    """The embedding rows after training: the delta's size per token, and whether the four
    rows (base + delta) are still one direction."""
    rep.kv("embedding rows (delta)", f"{r['trainable']:,} parameters")
    rep.table(["token", "id", "|delta|"],
              [[t, i, f"{n:.3f}"] for t, i, n in zip(r["tokens"], r["ids"], r["delta_norms"],
                                                     strict=True)])
    rep.kv("pairwise cosine, before", f"{r['before']['cos_min']:.4f} to "
           f"{r['before']['cos_max']:.4f}")
    rep.kv("pairwise cosine, after", f"{r['after']['cos_min']:.4f} to "
           f"{r['after']['cos_max']:.4f}")


def _not_started(rep, what: str, need: float, deadline: float) -> None:
    from . import progress

    left = deadline - time.time()
    msg = (f"{what} not started: it needs ~{T.SAFETY * need / 60:.0f} min (estimate x "
           f"{T.SAFETY:g}) and {left / 60:.0f} min are left before the deadline")
    progress.status(msg)
    rep.section("Not started")
    import textwrap

    for line in textwrap.wrap(msg, 70):
        rep.note(line)


def run_train(root: Path, quick: bool, arm: str, seed: int, repeat: bool,
              deadline: float | None, token_budget: int | None, checkpointing: bool,
              profile: str = "main") -> int:
    from .report import Report

    if arm not in T.PROFILES[profile].arms:
        sys.exit(f"--profile {profile} trains the {' and '.join(T.PROFILES[profile].arms)} "
                 f"arm only, not {arm}")
    m, device, determinism, batch = _context(root, quick, None, token_budget, checkpointing,
                                             profile)
    label = run_label(arm, seed, repeat)
    rep = Report(f"tinytools sft-train ({label})" + m.title + (" --quick" if quick else ""),
                 out=root / "outputs" / f"sft-train-{label}{m.suffix}.txt")
    _header(rep, m, device, determinism, batch, deadline)
    need = needed_seconds(m, root, seed, 1, 0)
    if not T.fits(time.time(), deadline, need, T.REPORT_ALLOW_S):
        _not_started(rep, label, need, deadline)
        rep.close()
        return 0
    res = _train_one(root, m, arm, seed, device, rep, repeat)
    if repeat:
        first = m.out / run_label(arm, seed)
        if (first / "train.json").exists():
            d = determinism_check(json.loads((first / "train.json").read_text()),
                                  T.result_record(res),
                                  first / "adapter.safetensors",
                                  m.out / label / "adapter.safetensors")
            rep.section(f"Trained twice: {run_label(arm, seed)}")
            for k, v in d.items():
                rep.kv(k, v)
    rep.close()
    return 0


def asdict_(res: T.Result) -> dict:
    from dataclasses import asdict

    return asdict(res)


def run_pair(root: Path, quick: bool, seed: int, deadline: float | None, batch_size: int | None,
             token_budget: int | None, checkpointing: bool, profile: str = "main") -> int:
    """One seed's pair, whole or not at all: train real, score, train control, score. A
    profile with the real arm only (onetool) runs the same code for its one arm: train
    real, score."""
    from .report import Report

    m, device, determinism, batch = _context(root, quick, batch_size, token_budget, checkpointing,
                                             profile)
    rep = Report(f"tinytools sft-pair (seed {seed})" + m.title + (" --quick" if quick else ""),
                 out=root / "outputs" / f"sft-pair-seed{seed}{m.suffix}.txt")
    _header(rep, m, device, determinism, batch, deadline)
    n = len(m.arms)
    need = needed_seconds(m, root, seed, n, n)
    if not T.fits(time.time(), deadline, need, T.REPORT_ALLOW_S):
        _not_started(rep, f"seed {seed}'s pair" if n == 2 else f"seed {seed}'s run", need,
                     deadline)
        rep.close()
        return 0
    results = {}
    for arm in m.arms:
        results[arm] = _train_one(root, m, arm, seed, device, rep)
        _score_one(root, m, run_label(arm, seed), m.out / run_label(arm, seed) /
                   "adapter.safetensors", device, batch, rep)
    if "control" not in results:
        rep.close()
        return 0
    rep.section(f"The pair: real and control, seed {seed}")
    r, c = results["real"], results["control"]
    rep.kv("same LoRA init (A)", r.init_sha256 == c.init_sha256)
    rep.kv("same first-step rows, same order", r.first_step_rows == c.first_step_rows)
    rep.kv("same rows and steps", (r.rows, r.steps) == (c.rows, c.steps))
    rep.close()
    return 0


def determinism_check(first: dict, second: dict, a1: Path, a2: Path) -> dict:
    """The same run trained twice: what matched (losses, adapter bytes, init, first rows)."""
    from .lora import load_adapter

    l1, l2 = first["losses"], second["losses"]
    equal_steps = sum(x == y for x, y in zip(l1, l2, strict=True))
    out = {"same LoRA init (A)": first["init_sha256"] == second["init_sha256"],
           "same first-step rows": first["first_step_rows"] == second["first_step_rows"],
           "losses identical, step by step": f"{equal_steps} of {len(l1)}",
           "largest loss difference": f"{max(abs(x - y) for x, y in zip(l1, l2, strict=True)):.3e}",
           "adapters bitwise identical": first["adapter_sha256"] == second["adapter_sha256"]}
    if not out["adapters bitwise identical"]:
        s1, _ = load_adapter(a1)
        s2, _ = load_adapter(a2)
        out["largest adapter difference"] = (
            f"{max(float((s1[k] - s2[k]).abs().max()) for k in s1):.3e}")
    return out


# -- sft-report ------------------------------------------------------------------------------


@dataclass
class Scores:
    """One model's replies, re-checked with the current checkers."""

    label: str
    ours: dict  # test task id -> outcome
    bfcl: dict  # item id -> outcome
    sample_greedy: dict  # train-sample task id -> greedy right (bool)
    sample_right: dict  # train-sample task id -> right of SAMPLES
    info: dict


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x]


def load_scores(d: Path, by_id: dict, bfcl_items: dict, reading: str = "strict",
                samples: bool = True) -> Scores:
    """``reading``: ``strict`` or ``tagfree`` (:mod:`tinytools.tagfree`); the replies are
    read that way, then checked by the same checkers. ``samples=False``: the greedy test
    passes only (the train-sample dicts stay empty; the onetool report)."""
    from . import bfcl, progress
    from .checkers import check_reply
    from .classify import is_right
    from .tagfree import read

    progress.status(f"re-checking {d.name}" + ("" if reading == "strict" else f" ({reading})"))
    ours = {}
    for r in _rows(d / OURS_FILE):
        ours[r["id"]] = check_reply(read(r["reply"], reading), by_id[r["id"]]["reference"]).outcome
    rows = _rows(d / BFCL_FILE)
    for r in rows:
        r["reply"] = read(r["reply"], reading)
    out = {}
    for cat, items in bfcl_items.items():
        mine = [r for r in rows if r["category"] == cat]
        if [r["id"] for r in mine] != [it.id for it in items]:
            raise ValueError(f"{d.name}: the BFCL {cat} replies are not the report's items")
        bfcl.rejudge(items, [[r] for r in mine])
        out.update({r["id"]: r["outcome"] for r in mine})
    greedy, right = {}, {}
    for r in _rows(d / SAMPLE_FILE) if samples else []:
        ok = is_right(check_reply(read(r["reply"], reading), by_id[r["id"]]["reference"]).outcome)
        if r["pass"] == "greedy":
            greedy[r["id"]] = ok
        else:
            right[r["id"]] = right.get(r["id"], 0) + ok
    return Scores(d.name, ours, out, greedy, right, json.loads((d / "score.json").read_text()))


def load_floor(d: Path, by_id: dict, reading: str = "strict") -> dict:
    """The noise floor's replies, re-checked: train-sample task id -> right of SAMPLES."""
    from . import progress
    from .checkers import check_reply
    from .classify import is_right
    from .tagfree import read

    progress.status(f"re-checking {d.name}" + ("" if reading == "strict" else f" ({reading})"))
    right: dict = {}
    for r in _rows(d / SAMPLE_FILE):
        if r["pass"] != SAMPLE_MODE:
            raise ValueError(f"{d.name}: a {r['pass']!r} reply in the noise floor")
        ok = is_right(check_reply(read(r["reply"], reading), by_id[r["id"]]["reference"]).outcome)
        right[r["id"]] = right.get(r["id"], 0) + ok
    return right


def complete_seeds(out: Path) -> tuple[list[int], list[str]]:
    """Seeds whose real and control runs are both trained and scored; and the labels of
    runs left out because their pair is not whole."""
    seeds, partial = [], []
    found = {}
    for d in out.iterdir() if out.exists() else []:
        for arm in T.ARMS:
            if d.name.startswith(f"{arm}-seed") and not d.name.endswith("-repeat"):
                found.setdefault(int(d.name.split("seed")[1]), {})[arm] = (
                    (d / "train.json").exists() and (d / "score.json").exists())
    for seed in sorted(found):
        if all(found[seed].get(a) for a in T.ARMS):
            seeds.append(seed)
        else:
            partial += [run_label(a, seed) for a in T.ARMS if a in found[seed]]
    return seeds, partial


def _shown(path: Path, root: Path) -> str:
    """``path`` relative to the repo when it is inside it, else as given."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def run_report(root: Path, quick: bool, profile: str = "main", reading: str = "strict",
               source: Path | None = None, main_from: Path | None = None) -> int:
    """sft-report. ``reading``: ``strict`` (the report as it always was, ``sft*.txt``) or
    ``tagfree`` (r5a-d and r6a on the tag-free reading, beside a strict vs tag-free table,
    and for the base profile plan 3f's rule; ``sft-tagfree*.txt``). ``source``: the arm
    tree to read instead of the profile's ``outputs/<profile>/`` (read-only). A variant
    profile (onetool) gets its comparison with the main run's arms instead
    (:mod:`tinytools.onetool`; ``main_from``: the main run's arm tree)."""
    from dataclasses import replace

    if T.PROFILES[profile].variant is not None:
        from . import onetool

        return onetool.run_report(root, quick, reading, source, main_from, profile)

    from . import bfcl, ours, tasks
    from .cli import _load_or_make_tasks, _quiet_hf
    from .report import WIDTH, Report
    from .tagfree import READINGS, STRICT, TAGFREE

    if reading not in READINGS:
        raise ValueError(f"unknown reading {reading!r}; one of {READINGS}")
    _quiet_hf()
    m = mode(quick, root, profile=profile)
    if source is not None:
        m = replace(m, out=source)
    if not (m.out / UNTRAINED / "score.json").exists():
        sys.exit(f"{_shown(m.out, root)}/{UNTRAINED} is not scored; run sft-score first")
    seeds, partial = complete_seeds(m.out)
    splits = _load_or_make_tasks(tasks.DEFAULT_SEED)
    test = ours.limit_per_kind(splits["test"], m.ours_limit)
    sample = ours.limit_per_kind(splits["train-sample"], m.sample_limit)
    by_id = {t["id"]: t for t in splits["test"] + splits["train-sample"]}
    items = {cat: bfcl.load_items(cat, m.bfcl_limit, "eval") for cat in bfcl.CATEGORIES}

    def load(rd: str) -> Arms:
        un = load_scores(m.out / UNTRAINED, by_id, items, rd)
        sft = [load_scores(m.out / run_label("real", s), by_id, items, rd) for s in seeds]
        ctl = [load_scores(m.out / run_label("control", s), by_id, items, rd) for s in seeds]
        for sc in [un, *sft, *ctl]:
            if list(sc.ours) != [t["id"] for t in test] or list(sc.sample_greedy) != [
                    t["id"] for t in sample]:
                raise ValueError(f"{sc.label}: its tasks are not the report's")
        floor = (load_floor(m.out / FLOOR, by_id, rd)
                 if (m.out / FLOOR / "score.json").exists() else None)
        if floor is not None and list(floor) != [t["id"] for t in sample]:
            raise ValueError(f"{FLOOR}: its tasks are not the report's")
        return Arms(un, sft, ctl, floor)

    a = load(reading)
    un, sft, ctl, floor = a.un, a.sft, a.ctl, a.floor
    trains = {d.name: json.loads((d / "train.json").read_text())
              for d in sorted(m.out.iterdir()) if (d / "train.json").exists()}
    checks = json.loads((m.out / "checks.json").read_text()) if (
        m.out / "checks.json").exists() else {}

    if reading == STRICT:
        rep = Report("tinytools sft-report" + m.title
                     + (" --quick (not for publication)" if quick else ""),
                     out=root / "outputs" / f"sft{m.suffix}.txt")
    else:
        title = f"tinytools sft-report{m.title} --reading tagfree" + (" --quick" if quick else "")
        if quick:
            title += ("\n" if len(title) + 22 > WIDTH else " ") + "(not for publication)"
        rep = Report(title, out=root / "outputs" / f"sft-tagfree{m.suffix}.txt")
    rep.header()
    _report_setup(rep, m, un, seeds, partial, test, sample, items)
    if reading == TAGFREE:
        strict = load(STRICT)
        _tagfree_setup(rep, _shown(m.out, root))
        _readings(rep, strict, a, seeds, test, items, sample)
    elif seeds:
        _r4c(rep, m, trains, seeds, checks)
        _r4d(rep, trains, seeds, m.out)
    _r5a(rep, un, sft, ctl, seeds, test, items)
    if seeds:
        _r5b(rep, un, sft, ctl, seeds, test, items)
        _r5c(rep, un, sft, ctl, seeds, test, items, reading)
        _r5d(rep, un, sft, ctl, test, items)
        if reading == TAGFREE and m.profile == "base":
            _base_rule(rep, a, strict, test, items)
        _r6a(rep, un, sft, ctl, seeds, sample, floor)
    rep.close()
    return 0


@dataclass(frozen=True)
class Arms:
    """Every arm's scores on one reading."""

    un: Scores
    sft: list[Scores]
    ctl: list[Scores]
    floor: dict | None


def _tagfree_setup(rep, source: str) -> None:
    from .tagfree import RULE

    rep.kv("replies read from", source, wrap=True)
    rep.kv("reading", "tag-free; strict beside it")
    for line in RULE:
        rep.note(line)
    rep.note("Training (r4c, r4d) is the same on both readings: see sft-report.")


def _arm_rows(a: Arms, seeds: list[int]) -> list[tuple[str, Scores]]:
    return ([("untrained", a.un)] + [(f"SFT, seed {s}", x) for s, x in zip(seeds, a.sft, strict=True)]
            + [(f"control, seed {s}", x) for s, x in zip(seeds, a.ctl, strict=True)])


def _readings(rep, strict: Arms, tagfree: Arms, seeds, test, items, sample) -> None:
    """Strict vs tag-free, right, per arm: where the two readings differ."""
    from fractions import Fraction

    from . import effects as E
    from . import tasks
    from .ours import share

    rep.section("Strict vs tag-free, right, per arm")
    rep.note("Right replies, count (share), on each reading. points: tag-free -")
    rep.note("strict. changed: replies whose outcome differs between the readings")
    rep.note("(any outcome, not only right).")
    sets = _sets(test, items)
    pairs = list(zip(_arm_rows(strict, seeds), _arm_rows(tagfree, seeds), strict=True))
    for key in (E.OURS_CALL, E.OURS_NOFIT, E.BFCL_CALL, E.BFCL_IRR):
        src, ids = sets[key]
        if not ids:
            continue
        rows = []
        for (label, st), (_, tf) in pairs:
            a, b = int(sum(_right(st, src, ids))), int(sum(_right(tf, src, ids)))
            changed = sum(x != y for x, y in zip(_outcomes(st, src, ids), _outcomes(tf, src, ids),
                                                 strict=True))
            rows.append([label, f"{a} ({share(a, len(ids))})", f"{b} ({share(b, len(ids))})",
                         E.points(Fraction(b - a, len(ids))), changed])
        rep.blank()
        rep.note(f"{E.SETS[key]} ({len(ids)}), greedy:")
        rep.table(["arm", "strict", "tag-free", "points", "changed"], rows)
    sample_pairs = [(label, st.sample_right, tf.sample_right) for (label, st), (_, tf) in pairs]
    if strict.floor is not None:
        sample_pairs.insert(1, ("untrained again", strict.floor, tagfree.floor))
    scopes = [("that need a call", [t["id"] for t in sample if t["kind"] != tasks.NO_FIT]),
              ("of all kinds", [t["id"] for t in sample])]
    for label_, ids in scopes:
        if not ids:
            continue
        n = SAMPLES * len(ids)
        rows = []
        for label, st, tf in sample_pairs:
            a, b = sum(st[i] for i in ids), sum(tf[i] for i in ids)
            rows.append([label, f"{a} ({share(a, n)})", f"{b} ({share(b, n)})",
                         E.points(Fraction(b - a, n))])
        rep.blank()
        rep.note(f"r6a's answers: train-sample tasks {label_} ({len(ids)}), {SAMPLES}")
        rep.note(f"each at T={TEMPERATURE:g} ({n:,} answers):")
        rep.table(["arm", "strict", "tag-free", "points"], rows)
    if strict.floor is not None:
        rep.note(f"untrained again: {FLOOR} (r6a's noise floor).")


def base_rule(un: Scores, sft: list[Scores], ids: list[str]):
    """Plan 3f's rule: SFT - untrained, ours, tasks that need a call, right (paired)."""
    from . import effects as E

    return E.paired("SFT", E.seed_mean([_right(x, "ours", ids) for x in sft]), "untrained",
                    _right(un, "ours", ids), len(sft), 1)


def _base_rule(rep, tagfree: Arms, strict: Arms, test, items) -> None:
    from . import effects as E

    rep.section("The base arm's rule (fixed before the run)")
    _, ids = _sets(test, items)[E.OURS_CALL]
    p, q = base_rule(tagfree.un, tagfree.sft, ids), base_rule(strict.un, strict.sft, ids)
    rep.note(f"base-arm rule (fixed before the run): SFT - untrained, ours, tasks that need a call, "
             f"right, tag-free: {E.points(p.diff)} points {E.interval(p.lo, p.hi)}, excludes 0: "
             f"{'yes' if p.excludes_zero else 'no'} -> "
             f"{'featured' if p.excludes_zero else 'a paragraph'}", wrap=True)
    rep.note(f"(untrained {E.percent(p.share_b)}, SFT {E.percent(p.share_a)}, seed mean; "
             f"{RESAMPLES_TEXT}, over {p.n} tasks.)", wrap=True)
    rep.note(f"Strict, for contrast (not the rule's reading): {E.points(q.diff)} points "
             f"{E.interval(q.lo, q.hi)} (untrained {E.percent(q.share_b)}, SFT "
             f"{E.percent(q.share_a)}).", wrap=True)


def _report_setup(rep, m: Mode, un: Scores, seeds, partial, test, sample, items) -> None:
    from . import tasks

    rep.section("Setup")
    rep.kv("model", m.settings.model)
    _profile_lines(rep, m)
    rep.kv("arms", "untrained, SFT (real), control")
    rep.kv("seeds with a whole pair", ", ".join(map(str, seeds)) or "none")
    if partial:
        rep.kv("left out (pair not whole)", ", ".join(partial))
    info = un.info
    rep.kv("scoring", f"{info['engine']} engine, batch {info['batch_size']}")
    rep.kv("  dtype, device", f"{info['dtype']}, {info['device']}")
    rep.kv("decoding", f"greedy; {SAMPLES} at T={TEMPERATURE:g} on train-sample")
    rep.kv("max new tokens", info["max_new_tokens"])
    n = {k: sum(t["kind"] == k for t in test) for k in tasks.KINDS}
    rep.kv("our test tasks", f"{len(test)} ({' / '.join(str(n[k]) for k in tasks.KINDS)})")
    rep.kv("BFCL eval items", f"{sum(len(v) for v in items.values())} ("
           + " / ".join(str(len(v)) for v in items.values()) + ")")
    rep.kv("train-sample tasks (r6a)", len(sample))
    rep.kv("adapters scored", "merged into the base weights")
    rep.note("W + (alpha/r) BA computed in float32, cast to the weights' dtype once.")
    rep.kv("checked by", "sft-report, from the saved replies")
    rep.note("Every reply is re-checked here with the current checkers (ours:")
    rep.note("execution match; BFCL: its AST checker), so a recapture after")
    rep.note("review is this command on the saved replies.")
    rep.kv("intervals", "paired bootstrap over tasks")
    rep.note(f"({RESAMPLES_TEXT}.)")
    rep.note("An arm's per-task value is the mean of its seeds' 0/1 values.")
    rep.note("Claims come from the five primary comparisons (r5d, marked);")
    rep.note("every other interval is descriptive.")


RESAMPLES_TEXT = "bootstrap, 10,000 resamples, seed 0, 95%"


def _r4c(rep, m: Mode, trains: dict, seeds: list[int], checks: dict) -> None:
    used = {k: v for k, v in trains.items() if not k.endswith("-repeat")}
    first = used[run_label("real", seeds[0])]
    s = first["settings"]
    for k, v in trains.items():
        if any(v["settings"][f] != s[f] for f in T.PAIRED):
            raise ValueError(f"{k} was trained with other settings than {run_label('real', seeds[0])}")
    pair = tuple(m.arms) == T.ARMS
    rep.section("Training: settings, the same for both arms (r4c)" if pair else
                "Training: settings (r4c)")
    rep.kv("model (base weights frozen)", s["model"])
    rep.kv("rows per arm" if pair else "rows", first["rows"])
    rep.kv("epochs", s["epochs"])
    rep.kv("optimizer steps", first["steps"])
    rep.note(f"{s['batch_rows']} rows each; the last step of an epoch takes the rest.")
    rep.kv("effective batch", f"{s['batch_rows']} rows")
    rep.kv("  micro-batches", f"token budget {s['token_budget']:,}")
    rep.note("  (rows x the longest row's tokens; gradients accumulated)")
    rep.kv("  gradient checkpointing", "on" if s["grad_checkpointing"] else "off")
    rep.kv("loss", "mean over the step's scored tokens")
    rep.kv("optimizer", f"AdamW, betas ({s['betas'][0]:g}, {s['betas'][1]:g}), eps {s['eps']:g}")
    rep.kv("  weight decay", f"{s['weight_decay']:g}")
    rep.kv("  gradient clipping", "none")
    rep.kv("learning rate", f"{s['lr']:g}, cosine to 0")
    rep.kv("  warmup", f"{s['warmup_ratio']:.0%} of the steps: {first['warmup']}")
    rep.kv("LoRA", f"rank {s['rank']}, alpha {s['alpha']:g}, dropout {s['dropout']:g}")
    rep.kv("  on", ",".join(x.removesuffix("_proj") for x in s["targets"]) + "; all layers")
    rep.kv("  trainable parameters", f"{first['trainable']:,}")
    er = first.get("embed_rows")
    if er is not None:
        rep.kv("embedding rows (delta)", f"{len(er['ids'])} rows, zero at the start")
        for t, i in zip(er["tokens"], er["ids"], strict=True):
            rep.kv(f"  {t}", f"id {i}")
        rep.kv("  trainable parameters", f"{er['trainable']:,}")
        rep.note("  In LoRA's AdamW group: its lr, schedule and betas;")
        rep.note(f"  weight decay {er['weight_decay']:g}, as LoRA's.")
        rep.note("  Embeddings are tied: the input embedding and the output head")
        rep.note("  both see E + delta.")
        rep.kv("trainable parameters, all", f"{first['trainable'] + er['trainable']:,}")
    for k, v in first["dtypes"].items():
        rep.kv(f"dtype: {k}", v)
    rep.kv("max length", f"{s['max_length']:,} tokens, no truncation")
    rep.kv("  rows cut", first["rows_cut"])
    rep.kv("  longest row", first["longest"])
    rep.kv("seed N sets", "LoRA A's init and the shuffle order")
    if pair:
        rep.note("Real N and control N share both: the pair. The real set has no")
        rep.note("randomness; control set N was drawn by sft-data with seed N.")
    else:
        rep.note("The real set has no randomness.")
    rep.blank()
    rows = []
    for arm in m.arms:
        t = used[run_label(arm, seeds[0])]
        rows.append([f"{arm}, seed {seeds[0]}", t["tokens"], t["scored"], t["tokens"] - t["scored"],
                     t["padded_tokens"]])
    rep.table(["tokens, one epoch", "all", "scored", "masked", "computed x2"], rows)
    rep.note("scored: the loss is computed on them (assistant turns through")
    rep.note("<|im_end|>); computed x2: token slots of both epochs with padding.")
    if checks:
        tpl, ln, mg = checks.get("template", {}), checks.get("lengths", {}), checks.get("merge_b0", {})
        rep.blank()
        rep.note("Checks (sft-check, before training):")
        if tpl:
            files = tpl["files"]
            rep.kv("  template: rows identical", f"{sum(v['identical'] for v in files.values())} "
                   f"of {sum(v['rows'] for v in files.values())} ({len(files)} "
                   f"file{'s' if len(files) != 1 else ''})")
            rep.note(f"    {tpl['model'].split('/')[-1]} vs sft-data's rendering with")
            rep.note(f"    {tpl['rendered_with'].split('/')[-1]}")
        if ln:
            rep.kv("  <|im_end|> scored", f"{ln['im_end_scored']} of {ln['assistant_turns']} "
                   "assistant turns")
        shapes, dated = checks.get("system_shapes", {}), checks.get("rows_with_date_line")
        if shapes:
            from .cli import _block

            rep.note("  system prompt before the tools section:")
            for label, lines in shapes.items():
                rep.note(f"    {label}:")
                for line in lines:
                    _block(rep, line, indent=8)
            if dated:
                rep.kv("  training rows with a date line", f"{dated[0]} of {dated[1]}")
        if mg:
            rep.kv("  merged B = 0 vs base", f"{mg['identical']} of {mg['tasks']} replies "
                   "identical")
            rep.note(f"    ({mg['model'].split('/')[-1]}, dev tasks, batch {mg['batch_size']}, "
                     f"{mg['device']})")
        rb, pins = checks.get("merge_random_b", {}), checks.get("sha256", {})
        if rb:
            gap = rb["rows"]["merged"]["relative"]
            rep.kv("  merged random B vs LoRA", f"gap {gap:.2e} (tolerance {rb['tolerance']:g})")
            rep.note(f"    ({rb['model'].split('/')[-1]}, {rb['prompts']} dev prompts, logits, "
                     f"{rb['dtype']})")
            if ROWS_LEFT_OUT in rb["rows"]:
                rep.kv("    merged without the rows", f"gap {rb['rows'][ROWS_LEFT_OUT]['relative']:.2e}"
                       " (must fail)")
        if pins:
            rep.kv("  SFT sets, sha256 as pinned", f"{sum(v['sha256'] == v['pinned'] for v in pins.values())} "
                   f"of {len(pins)}")
        _r4c_base_checks(rep, checks)


def _r4c_base_checks(rep, checks: dict) -> None:
    """r4c's lines for the base arm's own checks (absent from main's checks.json)."""
    pr, st, rw = checks.get("base_prompts"), checks.get("stops"), checks.get("embed_rows")
    if pr:
        n = sum(r["n"] for r in pr["sets"])
        rep.kv("  prompts as the Instruct arm's", f"{sum(r['as_instruct'] for r in pr['sets']):,} "
               f"of {n:,}")
        rep.note(f"    (training rows, our tasks, BFCL; {pr['tokenizer'].split('/')[-1]}'s")
        rep.note(f"    template; the base's own: {sum(r['with_own_template'] for r in pr['sets']):,}"
                 " identical)")
    if st:
        rep.kv("  replies end at", " or ".join(st["names"][str(i)] for i in st["engine_stop"]))
    if rw:
        rep.kv("  the four rows, base model", f"cosine {rw['base']['cos_min']:.4f} to "
               f"{rw['base']['cos_max']:.4f}")
        rep.note(f"    (norm {min(rw['base']['norms']):.3f}; median row "
                 f"{rw['median_norm']:.3f}; Instruct cosine "
                 f"{rw['instruct_rows']['cos_min']:.2f} to {rw['instruct_rows']['cos_max']:.2f})")
    mg = checks.get("merge_b0", {})
    if "rows" in mg:
        rep.kv("  merged delta = 0: rows as base", f"{mg['rows_equal']} of {mg['rows']}")


def _r4d(rep, trains: dict, seeds: list[int], out: Path) -> None:
    rep.section("Training runs (r4d)")
    rows = []
    for s in seeds:
        for arm in T.ARMS:
            t = trains[run_label(arm, s)]
            rows.append([arm, s, t["steps"], f"{t['losses'][0]:.4f}", f"{t['losses'][-1]:.4f}",
                         f"{t['seconds'] / 60:.1f}"])
    rep_label = run_label("real", seeds[0], True)
    if rep_label in trains:
        t = trains[rep_label]
        rows.append(["real again", t["seed"], t["steps"], f"{t['losses'][0]:.4f}",
                     f"{t['losses'][-1]:.4f}", f"{t['seconds'] / 60:.1f}"])
    rep.table(["arm", "seed", "steps", "loss, first", "loss, last", "minutes"], rows)
    rep.note("loss: the step's mean over its scored tokens. minutes: wall clock,")
    rep.note("loading included (varies run to run).")
    notes = {t.get("note") for t in trains.values()} - {"", None}
    for n in notes:
        rep.note(n, wrap=True)
    rep.blank()
    if rep_label in trains:
        first = run_label("real", seeds[0])
        rep.note(f"Training determinism: {first} trained twice (two processes)")
        for k, v in determinism_check(trains[first], trains[rep_label],
                                      out / first / "adapter.safetensors",
                                      out / rep_label / "adapter.safetensors").items():
            rep.kv(f"  {k}", v)
    else:
        rep.note("Training determinism: not measured (seed 1 real was trained once).")
    rep.blank()
    rep.note("The pairs: real N and control N")
    rep.table(["seed", "same init (A)", "same first-step rows"],
              [[s, trains[run_label("real", s)]["init_sha256"] ==
                trains[run_label("control", s)]["init_sha256"],
                trains[run_label("real", s)]["first_step_rows"] ==
                trains[run_label("control", s)]["first_step_rows"]] for s in seeds])


BFCL_CALL_CATS = ("simple", "multiple", "parallel", "parallel_multiple")
POOLED = "BFCL calls"


def _short(cat: str) -> str:
    return {"parallel_multiple": "par_multi"}.get(cat, cat)


def _categories(test: list[dict], items: dict) -> list[tuple[str, str, list[str]]]:
    """r5a's rows: (label, source, task ids): our three kinds, BFCL's five, BFCL's calls pooled."""
    from . import tasks

    out = [(tasks.SHORT_KIND[k], "ours", [t["id"] for t in test if t["kind"] == k])
           for k in tasks.KINDS]
    out += [(_short(c), "bfcl", [i.id for i in its]) for c, its in items.items()]
    out.append((POOLED, "bfcl", [i.id for c in BFCL_CALL_CATS for i in items[c]]))
    return [c for c in out if c[2]]


def _outcomes(sc: Scores, src: str, ids: list[str]) -> list[str]:
    d = sc.ours if src == "ours" else sc.bfcl
    return [d[i] for i in ids]


def _right(sc: Scores, src: str, ids: list[str]) -> list[float]:
    from .classify import is_right

    return [1.0 if is_right(o) else 0.0 for o in _outcomes(sc, src, ids)]


def _sets(test: list[dict], items: dict) -> dict:
    from . import effects as E
    from . import tasks

    return {E.OURS_CALL: ("ours", [t["id"] for t in test if t["kind"] != tasks.NO_FIT]),
            E.OURS_NOFIT: ("ours", [t["id"] for t in test if t["kind"] == tasks.NO_FIT]),
            E.BFCL_CALL: ("bfcl", [i.id for c in BFCL_CALL_CATS for i in items[c]]),
            E.BFCL_IRR: ("bfcl", [i.id for i in items["irrelevance"]])}


def _r5a(rep, un: Scores, sft: list[Scores], ctl: list[Scores], seeds: list[int], test,
         items) -> None:
    from . import effects as E
    from .ours import share

    rep.section("Right, per arm and category (r5a)")
    cats = _categories(test, items)
    rep.note("Right replies (count; greedy), per seed:")
    heads = ["category", "n", "untr"] + [f"SFT {s}" for s in seeds] + [f"ctl {s}" for s in seeds]
    rep.table(heads, [[label, len(ids), int(sum(_right(un, src, ids))),
                       *[int(sum(_right(x, src, ids))) for x in sft],
                       *[int(sum(_right(x, src, ids))) for x in ctl]]
                      for label, src, ids in cats])
    rep.note("no tool fits / irrelevance: right = declined. BFCL calls: simple,")
    rep.note("multiple, parallel and par_multi pooled.")
    for arm, runs in (("SFT", sft), ("control", ctl)):
        if not runs:
            continue
        rep.blank()
        rep.note(f"{arm} - untrained, paired by task ({arm}: mean of {len(runs)} seed"
                 f"{'s' if len(runs) > 1 else ''}):")
        rows = []
        for label, src, ids in cats:
            u = _right(un, src, ids)
            per = [_right(x, src, ids) for x in runs]
            p = E.paired(arm, E.seed_mean(per), "untrained", u, len(runs), 1)
            counts = [int(sum(v)) for v in per]
            rng = (f"{share(min(counts), len(ids))[:-1]}-{share(max(counts), len(ids))}"
                   if len(runs) > 1 else "-")
            rows.append([label, E.percent(p.share_b), E.percent(p.share_a), rng, E.points(p.diff),
                         E.interval(p.lo, p.hi)])
        rep.table(["category", "untrained", arm, "seed range", "gain", "95% interval"], rows)
    rep.note("gain: percentage points; interval: paired bootstrap over tasks.")


def _mean(counts: list[int]) -> str:
    return f"{sum(counts) / len(counts):.1f}"


def _r5b(rep, un, sft, ctl, seeds, test, items) -> None:
    from . import effects as E
    from .classify import CALL_OUTCOMES, short_label
    from .ours import pct

    rep.section("Outcomes per arm, tasks that need a call (r5b)")
    sets = _sets(test, items)
    for key in (E.OURS_CALL, E.BFCL_CALL):
        src, ids = sets[key]
        rep.note(f"{E.SETS[key]} ({len(ids)}), greedy:")

        def counts(sc, src=src, ids=ids):
            o = _outcomes(sc, src, ids)
            return [sum(x == c for x in o) for c in CALL_OUTCOMES]

        rows = [["untrained", *counts(un)]]
        for arm, runs in (("SFT", sft), ("control", ctl)):
            per = [counts(x) for x in runs]
            rows.append([f"{arm}, mean", *[_mean([p[i] for p in per])
                                           for i in range(len(CALL_OUTCOMES))]])
            rows += [[f"  seed {s}", *p] for s, p in zip(seeds, per, strict=True)]
        rep.table(["arm", *[short_label(c) for c in CALL_OUTCOMES]], rows)
        rep.blank()
    rep.note("Right function among replies that decoded to calls, beside chance")
    rep.note("(1/k for k tools offered): the control's diagnostic, not an effect.")
    by_id = {t["id"]: t for t in test}
    k_items = {i.id: len(i.prompt_functions) for c in ("multiple", "parallel_multiple")
               for i in items[c]}
    groups = [("ours", sets[E.OURS_CALL][1], [len(by_id[i]["tools"]) for i in sets[E.OURS_CALL][1]]),
              ("bfcl", list(k_items), list(k_items.values()))]
    for (src, ids, ks), title in zip(groups, ("ours, tasks that need a call (k = 3-5)",
                                              "BFCL multiple and par_multi"), strict=True):
        if not ids:
            continue
        rows = []
        for arm, runs in (("untrained", [un]), ("SFT", sft), ("control", ctl)):
            hits = n = 0
            chance = 0.0
            for sc in runs:
                h, k, c = E.right_function(_outcomes(sc, src, ids), ks)
                hits, n, chance = hits + h, n + k, chance + c
            rows.append([arm, pct(hits, n), f"{100 * chance / n:.1f}%" if n else "-"])
        rep.blank()
        rep.note(f"{title}, all seeds pooled:")
        rep.table(["arm", "right function", "chance"], rows)


def _r5c(rep, un, sft, ctl, seeds, test, items, reading: str = "strict") -> None:
    from . import effects as E
    from .ours import pct

    rep.section("Declining where no tool fits, per arm (r5c)")
    sets = _sets(test, items)
    cols = [(E.SETS[k], *sets[k]) for k in (E.OURS_NOFIT, E.BFCL_IRR)]

    def cell(sc, src, ids):
        return pct(int(sum(_right(sc, src, ids))), len(ids))

    rows = [["untrained", *[cell(un, src, ids) for _, src, ids in cols]]]
    for arm, runs in (("SFT", sft), ("control", ctl)):
        for s, sc in zip(seeds, runs, strict=True):
            rows.append([f"{arm}, seed {s}", *[cell(sc, src, ids) for _, src, ids in cols]])
    rep.table(["arm", "ours: declined", "BFCL irrelevance: declined"], rows)
    if reading == "strict":
        rep.note("declined: no <tool_call> tag (right). The effect, with intervals:")
        rep.note("r5d, declining.")
    else:
        rep.note("declined: no <tool_call> tag and no call object at the start")
        rep.note("(right). The effect, with intervals: r5d, declining.")


def _r5d(rep, un, sft, ctl, test, items) -> None:
    from . import effects as E

    rep.section("The effects: three paired differences each (r5d)")
    rep.note("Each effect is a per-task share: on every task a reply has one")
    rep.note("outcome; the variable is 1 when it is the effect's. Lower is better")
    rep.note("for every effect below except \"right\". \"SFT taught X\" is read only")
    rep.note("from SFT - untrained; SFT - control is the gap between the arms.")
    rep.note("[primary N]: one of the five comparisons claims come from (fixed")
    rep.note("before the run); every other row is descriptive.")
    sets = _sets(test, items)
    blocks = [(eff, key) for eff in E.EFFECTS for key in E.EFFECT_SETS[eff]]
    blocks.insert(4, (E.RIGHT_SHARE, E.BFCL_CALL))
    for eff, key in blocks:
        src, ids = sets[key]
        if not ids:
            continue
        r = E.effect_result(eff, key, _outcomes(un, src, ids),
                            [_outcomes(x, src, ids) for x in sft],
                            [_outcomes(x, src, ids) for x in ctl])
        title = eff.name + (" (harm check only)" if eff.harm_check else "")
        rep.blank()
        rep.print(f"  {title}" + (f"  [primary {r.primary}]" if r.primary else ""))
        rep.note(f"  1 = {eff.variable}")
        rep.note(f"  on {E.SETS[key]}, {r.n} tasks")
        rep.note("  " + "   ".join(f"{a} {E.percent(v)}" for a, v in r.shares.items())
                 + "   (seed means)")
        rep.table(["difference", "points", "95% interval", "excludes 0"],
                  [[f"{p.a} - {p.b}", E.points(p.diff), E.interval(p.lo, p.hi),
                    p.excludes_zero] for p in r.diffs])
    rep.blank()
    rep.note(f"Intervals: {RESAMPLES_TEXT}, over tasks.")


def _r6a_head(rep, sample) -> tuple[list[str], list[str]]:
    """r6a's first lines (what it counts); returns (call-needing ids, all ids)."""
    from . import tasks

    g = SAMPLES
    call_ids = [t["id"] for t in sample if t["kind"] != tasks.NO_FIT]
    all_ids = [t["id"] for t in sample]
    n = {k: sum(t["kind"] == k for t in sample) for k in tasks.KINDS}
    rep.kv("train-sample tasks", f"{len(sample)} ("
           + " / ".join(str(n[k]) for k in tasks.KINDS) + ")")
    rep.kv("samples", f"{g} per task at T={TEMPERATURE:g}, seed {SEED}")
    rep.kv("reward", "execution match (declining is right")
    rep.note("                                     on no-tool-fits tasks)")
    rep.note(f"all wrong: 0 of {g} right; all right: {g} of {g}; mixed: the rest (a")
    rep.note("reward's signal). Two denominators: the tasks that need a call")
    rep.note(f"({len(call_ids)}, as Milestone 0's rule (c)) and all kinds ({len(all_ids)}).")
    return call_ids, all_ids


def _transition_rows(un_rights, scope_rows, g: int) -> list[list]:
    """The transition table's rows: per scope (label, ids, per-seed rights by id), the
    untrained group (rows) by the other's group (columns), summed over its seeds."""
    from . import effects as E

    table = []
    for label, ids, xs in scope_rows:
        if not ids:
            continue
        t = E.add_tables([E.transitions([un_rights[i] for i in ids], [x[i] for i in ids], g)
                          for x in xs])
        for j, frm in enumerate(E.GROUPS):
            table.append([label if j == 0 else "", frm, *[t[(frm, to)] for to in E.GROUPS],
                          sum(t[(frm, to)] for to in E.GROUPS)])
    return table


def _r6a(rep, un, sft, ctl, seeds, sample, floor: dict | None = None) -> None:
    from . import effects as E
    from . import tasks
    from .ours import MIXED_HEADERS, histogram, mixed_rows, pass_at_k, pct

    g = SAMPLES
    rep.section(f"Groups of {g} on post 10's training tasks (r6a)")
    call_ids, all_ids = _r6a_head(rep, sample)
    scopes = [("need a call", call_ids), ("all kinds", all_ids)]
    kinds = [(tasks.SHORT_KIND[k], [t["id"] for t in sample if t["kind"] == k])
             for k in tasks.KINDS]

    arms = [("untrained", un)] + [(f"SFT {s}", x) for s, x in zip(seeds, sft, strict=True)] \
        + [(f"control {s}", x) for s, x in zip(seeds, ctl, strict=True)]

    def rights(sc, ids):
        return [sc.sample_right[i] for i in ids]

    rep.blank()
    rep.note(f"Tasks by right answers of {g}, all kinds:")
    rep.table(["arm", *[str(k) for k in range(g + 1)]],
              [[a, *histogram(rights(sc, all_ids), g)] for a, sc in arms])
    for label, ids in scopes:
        rep.blank()
        rep.note(f"Groups, tasks that {label}:" if label == "need a call" else
                 "Groups, all kinds:")
        rep.table(["arm", *MIXED_HEADERS], mixed_rows([(a, rights(sc, ids)) for a, sc in arms], g))
    rep.note("share: mixed / tasks; per 1k: mixed groups per 1,000 answers")
    rep.note(f"(post 8's unit) = (1000 / {g}) x share. Rounded half up.")
    for label, ids in scopes:
        rep.blank()
        rep.note(f"Greedy right vs pass@{g}, tasks that {label}:" if label == "need a call"
                 else f"Greedy right vs pass@{g}, all kinds:")
        rep.table(["arm", "greedy right", f"pass@{g}"],
                  [[a, pct(sum(sc.sample_greedy[i] for i in ids), len(ids)),
                    pct(pass_at_k(rights(sc, ids)), len(ids))] for a, sc in arms])

    # the noise floor first: the untrained model drawn again, no training (T8 review B2)
    rep.section("r6a: untrained -> untrained again, the noise floor")
    if floor is None:
        rep.note(f"Not measured: {FLOOR} was not scored (sft-score --untrained")
        rep.note("--noise-floor). The movements below are read without a baseline.")
    else:
        rep.note(f"Rows: the task's group untrained (generation seed {SEED}); columns: the")
        rep.note(f"same model's {g} drawn again (generation seed {FLOOR_SEED}). No training:")
        rep.note("every movement here is the redraw alone.")
        rep.table(["scope", "untrained", *E.GROUPS, "total"],
                  _transition_rows(un.sample_right, [(f"{label}", ids, [floor])
                                                     for label, ids in scopes]
                                   + [(k, ids, [floor]) for k, ids in kinds], g))
        rep.blank()
        rep.note("The movements, share of tasks with a 95% interval over tasks:")
        for label, ids in scopes:
            rows = []
            for (frm, to), _ in E.MOVEMENTS:
                v = E.movement_values(rights(un, ids), [[floor[i] for i in ids]], g, (frm, to))
                lo, hi = E.bootstrap_interval(v)
                rows.append([f"{frm} -> {to}", E.percent(E.exact_mean(v, 1)),
                             E.share_interval(lo, hi)])
            rep.blank()
            rep.table([f"movement, {label}", f"share of {len(ids)}", "95% interval, %"], rows)

    for arm, runs in (("SFT", sft), ("control", ctl)):
        rep.section(f"r6a: untrained -> {arm}, task by task")
        rep.note(f"Rows: the task's group untrained; columns: under {arm}. Pooled:")
        rep.note(f"summed over the {len(runs)} seed{'s' if len(runs) > 1 else ''} (each task once per seed).")
        per_run = [x.sample_right for x in runs]
        scope_rows = [(f"pooled, {label}", ids, per_run) for label, ids in scopes]
        scope_rows += [(f"seed {s}, {label}", ids, [x.sample_right])
                       for s, x in zip(seeds, runs, strict=True) for label, ids in scopes]
        scope_rows += [(k, ids, per_run) for k, ids in kinds]
        rep.table(["scope", "untrained", *E.GROUPS, "total"],
                  _transition_rows(un.sample_right, scope_rows, g))
        rep.note("By kind: pooled over seeds, all kinds' rows above split three ways.")

        rep.blank()
        rep.note(f"Shares, {arm} (seed mean) - untrained, paired by task:")
        rows = []
        for label, ids in scopes:
            for grp in E.GROUPS:
                p = E.paired(arm, E.seed_mean([E.group_values(rights(x, ids), g, grp)
                                               for x in runs]),
                             "untrained", E.group_values(rights(un, ids), g, grp), len(runs), 1)
                rows.append([label if grp == E.GROUPS[0] else "", grp, E.percent(p.share_b),
                             E.percent(p.share_a), E.points(p.diff), E.interval(p.lo, p.hi)])
        rep.table(["tasks", "group", "untrained", arm, "points", "95% interval"], rows)

        rep.blank()
        rep.note("The movements (named before the run), share of tasks (seed mean):")
        for (frm, to), reading in E.MOVEMENTS:
            rep.note(f"  {frm} -> {to}: {reading.format(arm=arm)}")
        if floor is not None:
            rep.note("floor: the same movement with no training (the redraw above);")
            rep.note("beyond: the arm's share - the floor's, points, paired by task,")
            rep.note("with its 95% interval. A movement is read only beyond the floor.")
        else:
            rep.note("(no noise floor measured: shares with a 95% interval over tasks)")
        largest = None
        for label, ids in scopes:
            rows = []
            for (frm, to), _ in E.MOVEMENTS:
                un_r, xs = rights(un, ids), [rights(x, ids) for x in runs]
                v = E.movement_values(un_r, xs, g, (frm, to))
                share_ = E.exact_mean(v, len(runs))
                if floor is None:
                    lo, hi = E.bootstrap_interval(v)
                    rows.append([f"{frm} -> {to}", E.percent(share_), E.share_interval(lo, hi)])
                    key = share_
                else:
                    fl = E.exact_mean(E.movement_values(un_r, [[floor[i] for i in ids]], g,
                                                        (frm, to)), 1)
                    d = E.movement_values(un_r, xs, g, (frm, to), floor=[floor[i] for i in ids])
                    lo, hi = E.bootstrap_interval(d)
                    key = E.exact_mean(d, len(runs))
                    rows.append([f"{frm} -> {to}", E.percent(share_), E.percent(fl),
                                 E.points(key), E.interval(lo, hi)])
                if label == "need a call" and (largest is None or key > largest[0]):
                    largest = (key, f"{frm} -> {to}", lo, hi)
            rep.blank()
            if floor is None:
                rep.table([f"movement, {label}", f"share of {len(ids)}", "95% interval, %"],
                          rows)
            else:
                rep.table([f"movement, {label} ({len(ids)})", arm, "floor", "beyond",
                           "95% interval"], rows)
        if largest and floor is not None:
            rep.kv("largest beyond the floor", f"{largest[1]}: {E.points(largest[0])} points")
            rep.note(f"  tasks that need a call; 95% interval {E.interval(largest[2], largest[3])}")
        elif largest:
            rep.kv("largest, tasks that need a call", f"{largest[1]}: {E.percent(largest[0])}")
            rep.note(f"  95% interval {E.share_interval(largest[2], largest[3])}%")
