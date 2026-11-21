"""Score a Hugging Face model on BFCL with bfcl-eval's own data and AST checker.

What comes from bfcl-eval (imported, not reimplemented):

* the test items and reference answers (``load_dataset_entry``,
  ``load_ground_truth_entry``), with the Python hint added to each function's
  description for the prompt exactly as its generation step does, and the
  un-hinted docs passed to the checker exactly as its evaluation step does;
* the decoder for Qwen's ``<tool_call>`` format (``QwenFCHandler.decode_ast``);
* the verdict (``_evaluate_single_ast_entry`` and
  ``_evaluate_single_relevance_entry``: the functions ``bfcl evaluate`` runs
  per item, which call ``ast_checker``).

What is ours: generation with Hugging Face transformers on the local device
(bfcl-eval drives a vLLM/SGLang server, which doesn't run on a Mac), the prompt
built with the model's own chat template, and the outcome split
(:mod:`tinytools.classify`).
"""

from __future__ import annotations

import os
import random
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# bfcl-eval creates result/, score/ and .file_locks/ under this directory on
# import; keep them out of site-packages and out of git.
os.environ.setdefault("BFCL_PROJECT_ROOT", str(ROOT / "bfcl_project"))

import torch  # noqa: E402

from .classify import OUTCOMES, classify, is_right  # noqa: E402

# Our names -> bfcl-eval's (V4 renamed "simple" to "simple_python").
CATEGORIES = {
    "simple": "simple_python",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "irrelevance": "irrelevance",
}
MAX_NEW_TOKENS = 512
# gorilla.cs.berkeley.edu/leaderboard.html (last updated 2026-04-12): "Models are
# evaluated using commit f7cf735 ... or install the PyPI package
# pip install bfcl-eval==2025.12.17". Read 2026-11-20.
LEADERBOARD_BFCL_EVAL = "2025.12.17"


def bfcl_version() -> str:
    from importlib.metadata import version

    return version("bfcl-eval")


@dataclass(frozen=True)
class Item:
    id: str
    category: str  # our name
    messages: list[dict]  # the first (only) turn's messages
    prompt_functions: list[dict]  # with bfcl's language hint, for the prompt
    checker_entry: dict  # bfcl's un-hinted entry, for the checker
    ground_truth: list | None  # None for irrelevance

    @property
    def irrelevance(self) -> bool:
        return self.ground_truth is None

    def expected_names(self) -> list[str]:
        return [name for call in self.ground_truth or [] for name in call]


# The dev slice: about 10% of each category, drawn with this seed, used only by
# Milestone 0 and held out of every published BFCL number (split "eval").
DEV_SEED = 0
DEV_SHARE = 10  # percent
SPLITS = ("eval", "dev", "all")


def dev_ids(ids: list[str], category: str, seed: int = DEV_SEED) -> list[str]:
    """The dev slice of one category: round(10%) of its ids (at least one), drawn
    with ``Random("<seed>:bfcl-dev:<category>")``, in the order given. Pure."""
    if not ids:
        return []
    n = max(1, (len(ids) * DEV_SHARE + 50) // 100)
    picked = set(random.Random(f"{seed}:bfcl-dev:{category}").sample(sorted(ids), n))
    return [i for i in ids if i in picked]


def in_split(ids: list[str], dev: list[str], split: str) -> list[str]:
    """``ids`` restricted to a split: dev (the slice), eval (all minus it) or all. Pure."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    d = set(dev)
    return [i for i in ids if split == "all" or (i in d) == (split == "dev")]


def category_ids(category: str) -> list[str]:
    """Every item id of a category, in bfcl-eval's file order."""
    from bfcl_eval.utils import load_dataset_entry

    return [e["id"] for e in load_dataset_entry(CATEGORIES[category],
                                                include_language_specific_hint=False)]


def dev_slice(category: str, seed: int = DEV_SEED) -> list[str]:
    """The seeded dev slice of a BFCL category (item ids, in file order)."""
    return dev_ids(category_ids(category), category, seed)


def load_items(category: str, limit: int | None, split: str = "all",
               dev_seed: int = DEV_SEED) -> list[Item]:
    """The first ``limit`` items of a category's split, in bfcl-eval's file order."""
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

    name = CATEGORIES[category]
    hinted = load_dataset_entry(name, include_language_specific_hint=True)
    plain = load_dataset_entry(name, include_prereq=False, include_language_specific_hint=False)
    truth = None if category == "irrelevance" else load_ground_truth_entry(name)
    ids = [h["id"] for h in hinted]
    keep = set(in_split(ids, dev_ids(ids, category, dev_seed), split))
    items = []
    for i, (h, p) in enumerate(zip(hinted, plain, strict=True)):
        if limit is not None and len(items) >= limit:
            break
        if h["id"] not in keep:
            continue
        assert h["id"] == p["id"]
        gt = None
        if truth is not None:
            assert truth[i]["id"] == h["id"]
            gt = truth[i]["ground_truth"]
        assert len(h["question"]) == 1, "single-turn categories only"
        items.append(Item(h["id"], category, h["question"][0], h["function"], p, gt))
    return items


# -- the checker -----------------------------------------------------------------

MODEL_KEY = "tinytools/hf-qwen-fc"


def handler():
    """bfcl-eval's Qwen function-calling handler, registered under our own key.

    The registry entry matters to the checker only through ``underscore_to_dot``
    (whether to rewrite ``.`` in function names); ours is False because the
    names go into the prompt unchanged.
    """
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

    if MODEL_KEY not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[MODEL_KEY] = ModelConfig(
            model_name=MODEL_KEY, display_name=MODEL_KEY, url="", org="", license="",
            model_handler=QwenFCHandler, input_price=None, output_price=None,
            is_fc_model=True, underscore_to_dot=False,
        )
    return QwenFCHandler(MODEL_KEY, temperature=0, registry_name=MODEL_KEY, is_fc_model=True)


@dataclass(frozen=True)
class Verdict:
    outcome: str
    bfcl_valid: bool
    error_type: str


def judge(h, item: Item, raw: str) -> Verdict:
    """BFCL's verdict on one reply, and our outcome label."""
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.eval_runner import (
        _evaluate_single_ast_entry,
        _evaluate_single_relevance_entry,
    )

    try:
        decoded = h.decode_ast(raw, ReturnFormat.PYTHON, False)
    except Exception:  # noqa: BLE001 - any decode failure means "unparseable"
        decoded = None
    if item.irrelevance:
        res = _evaluate_single_relevance_entry(
            h, item.id, raw, item.checker_entry, MODEL_KEY, CATEGORIES[item.category])
    else:
        res = _evaluate_single_ast_entry(
            h, item.id, raw, item.ground_truth, item.checker_entry, MODEL_KEY,
            CATEGORIES[item.category], language=Language.PYTHON,
            return_format=ReturnFormat.PYTHON, has_tool_call_tag=False)
    valid = bool(res["valid"])
    outcome = classify(raw=raw, decoded=decoded, bfcl_valid=valid, irrelevance=item.irrelevance,
                       expected_names=item.expected_names())
    return Verdict(outcome, valid, res.get("error_type", ""))


def _pick(options: list, required: bool):
    """One allowed value from BFCL's list, or ``None`` to omit the parameter.

    ``""`` in the list means "may be omitted". Omit such a parameter unless the
    function doc requires it (BFCL's data disagrees with its docs on this in a
    few items, e.g. simple_python_200), else take the first non-empty value.
    """
    if "" in options and not required:
        return None
    return _resolve(next(v for v in options if v != ""))


def _resolve(value, required: frozenset = frozenset()):
    if isinstance(value, dict):
        picked = {k: _pick(v, k in required) for k, v in value.items()}
        return {k: v for k, v in picked.items() if v is not None}
    if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
        return [_resolve(x) for x in value]
    return value


def oracle_reply(item: Item) -> str:
    """The reference answer written in Qwen's format: the harness's full-marks control.

    Each parameter takes the first non-empty value BFCL allows (recursively
    inside dict values); optional ones are omitted unless the doc requires them.
    Irrelevance items get a plain refusal.
    """
    import json

    if item.irrelevance:
        return "None of the provided functions can answer this."
    docs = {f["name"]: f for f in item.checker_entry["function"]}
    blocks = []
    for call in item.ground_truth:
        (name, params), = call.items()
        required = frozenset(docs[name]["parameters"].get("required", []))
        blocks.append('<tool_call>\n{"name": %s, "arguments": %s}\n</tool_call>'
                      % (json.dumps(name), json.dumps(_resolve(params, required))))
    return "\n".join(blocks)


# -- generation ------------------------------------------------------------------


def load_model(model_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .device import preferred_dtype

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=preferred_dtype(device))
    model.to(device)
    model.eval()
    return tok, model


def prompt_text(tok, item: Item) -> str:
    tools = [{"type": "function", "function": f} for f in item.prompt_functions]
    return tok.apply_chat_template(item.messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True)


def item_seed(seed: int, item_id: str) -> int:
    """A per-item seed, so an item's samples don't depend on what ran before it."""
    return (seed + zlib.crc32(item_id.encode())) % (2**31)


def generation_config(tok, model, samples: int, temperature: float):
    """Decoding exactly as stated: greedy, or plain temperature sampling.

    Qwen2.5-Instruct's own generation_config.json sets repetition_penalty 1.1,
    top_p 0.8 and top_k 20, which ``generate`` would apply silently; this
    config replaces it so greedy is argmax and sampling is the full softmax.
    """
    from transformers import GenerationConfig

    common = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tok.pad_token_id,
        repetition_penalty=1.0,
        top_p=1.0,
        top_k=0,
    )
    if samples == 1 and temperature == 0:
        return GenerationConfig(do_sample=False, **common)
    return GenerationConfig(do_sample=True, temperature=temperature,
                            num_return_sequences=samples, **common)


@dataclass
class Generated:
    texts: list[str]
    new_tokens: list[int]
    capped: list[bool]
    seconds: float


def generate(tok, model, item: Item, cfg, seed: int) -> Generated:
    return generate_text(tok, model, prompt_text(tok, item), item.id, cfg, seed)


def generate_text(tok, model, prompt: str, item_id: str, cfg, seed: int) -> Generated:
    """Generate from an already templated prompt; seeded by ``seed`` and the item's id.

    The one generation path: ``bfcl-score`` and ``score`` (our tasks) both use it.
    """
    from .device import sync

    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ids = ids.to(device)  # int64 stays int64: move only, no dtype change
    torch.manual_seed(item_seed(seed, item_id))
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
    sync(device)
    seconds = time.perf_counter() - t0
    new = out[:, ids.shape[1]:].cpu()
    eos = set(cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id])
    texts, lengths, capped = [], [], []
    for row in new.tolist():
        n = next((i for i, t in enumerate(row) if t in eos), None)
        capped.append(n is None)
        if n is None:
            n = len(row)
        lengths.append(n)
        texts.append(tok.decode(row[:n], skip_special_tokens=False))
    return Generated(texts, lengths, capped, seconds)


# -- a scoring pass --------------------------------------------------------------


@dataclass
class CategoryResult:
    category: str
    n: int = 0
    outcomes: dict = field(default_factory=lambda: {o: 0 for o in OUTCOMES})
    bfcl_right: int = 0
    hit_cap: int = 0
    capped_outcomes: Counter = field(default_factory=Counter)  # outcome of each capped reply
    new_tokens: int | None = 0  # None: not recorded (a rescore of an older replies file)
    seconds: float | None = 0.0
    right_per_task: list[int] = field(default_factory=list)  # sampling mode
    oracle_right: int = 0
    oracle_n: int = 0

    def add_reply(self, v: Verdict, capped: bool) -> bool:
        """Count one reply; return whether it is right."""
        self.n += 1
        self.outcomes[v.outcome] += 1
        self.bfcl_right += v.bfcl_valid
        if capped:
            self.hit_cap += 1
            self.capped_outcomes[v.outcome] += 1
        return is_right(v.outcome)

    def add_oracle(self, h, item: Item) -> None:
        """The reference-answer control for one item."""
        o = judge(h, item, oracle_reply(item))
        self.oracle_n += 1
        self.oracle_right += o.bfcl_valid and is_right(o.outcome)


def score(tok, model, items: list[Item], *, samples: int, temperature: float, seed: int,
          on_item=None, records: list | None = None, setup: dict | None = None) -> CategoryResult:
    """Generate and judge every item. ``records`` collects one dict per reply (with
    ``setup``'s fields, e.g. dtype and device, so ``bfcl-rescore`` can print them)."""
    h = handler()
    cfg = generation_config(tok, model, samples, temperature)
    res = CategoryResult(items[0].category if items else "")
    for k, item in enumerate(items, start=1):
        g = generate(tok, model, item, cfg, seed)
        res.seconds += g.seconds
        res.new_tokens += sum(g.new_tokens)
        right = 0
        for i, (text, capped, n_new) in enumerate(zip(g.texts, g.capped, g.new_tokens,
                                                      strict=True)):
            v = judge(h, item, text)
            right += res.add_reply(v, capped)
            if records is not None:
                records.append({"id": item.id, "category": item.category, "sample": i,
                                "reply": text, "outcome": v.outcome,
                                "bfcl_valid": v.bfcl_valid, "bfcl_error_type": v.error_type,
                                "hit_cap": capped, "new_tokens": n_new,
                                "task_seconds": g.seconds, **(setup or {})})
        res.right_per_task.append(right)
        res.add_oracle(h, item)
        if on_item:
            on_item(k)
    return res


def rejudge(items: list[Item], rows: list[list[dict]], on_item=None) -> CategoryResult:
    """Judge saved replies again (no model): ``rows[k]`` are item ``k``'s replies,
    in sample order. Updates each row's verdict fields in place."""
    h = handler()
    res = CategoryResult(items[0].category if items else "")
    timed = all("new_tokens" in r and "task_seconds" in r for rs in rows for r in rs)
    if not timed:
        res.new_tokens, res.seconds = None, None
    for k, (item, rs) in enumerate(zip(items, rows, strict=True), start=1):
        right = 0
        for r in rs:
            v = judge(h, item, r["reply"])
            right += res.add_reply(v, r["hit_cap"])
            r.update(outcome=v.outcome, bfcl_valid=v.bfcl_valid, bfcl_error_type=v.error_type)
            if timed:
                res.new_tokens += r["new_tokens"]
        if timed:
            res.seconds += rs[0]["task_seconds"]
        res.right_per_task.append(right)
        res.add_oracle(h, item)
        if on_item:
            on_item(k)
    return res
