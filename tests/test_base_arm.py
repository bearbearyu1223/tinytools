"""Post 9's base arm (plan 3f): the profiles, the four embedding rows (trained through both
the input embedding and the tied output head, saved, merged), the base arm's checks, the
deadline estimate, base.toml, and that the main profile and the main run's adapters are as
they were."""

import json
import math
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from test_train import _sets, _tiny_model, _tok

from tinytools import lambda_run as lr
from tinytools import lora as L
from tinytools import sftrun as R
from tinytools import train as T
from tinytools.cli import _r1_path, build_parser

ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN = ROOT / "runs" / "2026-11-23-main" / "outputs" / "main"


# -- the profiles ----------------------------------------------------------------------------


def test_the_main_profile_is_the_main_run():
    p = T.PROFILES["main"]
    assert p.full is T.MAIN and p.quick is T.QUICK
    assert (p.tokenizer, p.quick_tokenizer, p.embed_rows, p.stop_tokens) == (None, None, (), ())
    assert p.tokenizer_for("anything") == "anything" and p.title() == ""
    for quick in (False, True):
        m = R.mode(quick, ROOT)
        assert m.out.name == ("main-quick" if quick else "main")
        assert m.suffix == ("-quick" if quick else "") and m.title == ""
        assert m.tokenizer is None and m.embed_rows == () and m.stop_tokens == ()
        assert m.settings is (T.QUICK if quick else T.MAIN)


def test_the_base_profile():
    p = T.PROFILES["base"]
    assert (p.full.model, p.quick.model) == ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B")
    # every other setting is main's: the pre-registered T8 spec
    for f in T.PAIRED:
        if f != "model":
            assert getattr(p.full, f) == getattr(T.MAIN, f)
            assert getattr(p.quick, f) == getattr(T.QUICK, f)
    assert p.full.token_budget == T.MAIN.token_budget and p.quick.rows == 32
    assert p.tokenizer_for("Qwen/Qwen2.5-1.5B") == "Qwen/Qwen2.5-1.5B-Instruct"
    assert p.tokenizer_for("Qwen/Qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B-Instruct"
    with pytest.raises(ValueError):
        p.tokenizer_for("Qwen/Qwen2.5-3B")
    assert p.embed_rows == ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>")
    assert set(p.stop_tokens) == {"<|im_end|>", "<|endoftext|>"}
    full, quick = R.mode(False, ROOT, profile="base"), R.mode(True, ROOT, profile="base")
    assert (full.out.name, full.suffix) == ("base", "-base")
    assert (quick.out.name, quick.suffix) == ("base-quick", "-base-quick")
    assert full.tokenizer == "Qwen/Qwen2.5-1.5B-Instruct"
    assert quick.tokenizer == "Qwen/Qwen2.5-0.5B-Instruct"
    assert full.title == " --profile base" and full.ours_limit is None and quick.ours_limit == 2


def test_r1_prompts_of_the_base_arm_go_under_its_directory():
    assert _r1_path("main", "Qwen/Qwen2.5-1.5B-Instruct") == ROOT / "outputs" / "r1-prompt.txt"
    assert _r1_path("base", "Qwen/Qwen2.5-1.5B") == ROOT / "outputs" / "base" / "r1-prompt.txt"
    assert _r1_path("base", "Qwen/Qwen2.5-0.5B") == \
        ROOT / "outputs" / "base-quick" / "r1-prompt.txt"


def test_every_sft_command_takes_the_profile():
    parser = build_parser()
    for cmd in (["sft-check"], ["sft-score", "--untrained"], ["sft-train", "--arm", "real",
                                                              "--seed", "1"],
                ["sft-pair", "--seed", "1"], ["sft-report"],
                ["rescore", "--replies", "x"], ["bfcl-rescore", "--replies", "x"]):
        assert parser.parse_args(cmd).profile == "main"
        assert parser.parse_args([*cmd, "--profile", "base"]).profile == "base"
    with pytest.raises(SystemExit):
        parser.parse_args(["sft-check", "--profile", "other"])


# -- the deadline estimate ------------------------------------------------------------------


def test_main_estimates_from_every_scoring_and_the_base_arm_from_trained_ones():
    walls = {"untrained": 2800.0, R.FLOOR: 1450.0, "real-seed1": 430.0,
             "control-seed1": 450.0}
    assert sorted(R.scoring_walls(walls, prefer_trained=False)) == [430.0, 450.0, 1450.0, 2800.0]
    assert sorted(R.scoring_walls(walls, prefer_trained=True)) == [430.0, 450.0]
    # before a trained model is scored, the base arm's estimate is the prior
    untrained_only = {"untrained": 2800.0, R.FLOOR: 1450.0}
    assert R.scoring_walls(untrained_only, prefer_trained=True) == []
    est = T.estimate_seconds([], [], 16_200_000, 2, 2, False, R.SCORE_PRIOR_S)
    assert est == pytest.approx(2 * T.PRIOR_S_PER_TOKEN * 16_200_000 + 2 * R.SCORE_PRIOR_S)
    # a trained model that ran to the cap is what seed 2's estimate takes
    capped = dict(walls, **{"control-seed1": 2900.0})
    assert max(R.scoring_walls(capped, prefer_trained=True)) == 2900.0


def test_measured_reads_score_walls_per_profile(tmp_path):
    for label, wall in (("untrained", 2800.0), ("real-seed1", 430.0)):
        (tmp_path / "outputs" / "base" / label).mkdir(parents=True)
        (tmp_path / "outputs" / "base" / label / "score.json").write_text(
            json.dumps({"wall_s": wall}))
    (tmp_path / "outputs" / "main" / "untrained").mkdir(parents=True)
    (tmp_path / "outputs" / "main" / "untrained" / "score.json").write_text(
        json.dumps({"wall_s": 640.0}))
    assert R.measured(R.mode(False, tmp_path, profile="base"))[1] == [430.0]
    assert R.measured(R.mode(False, tmp_path))[1] == [640.0]


# -- the embedding rows: both sides, one parameter ------------------------------------------


class Tied(nn.Module):
    """A minimal tied language model (embedding -> mixing -> tied head), float32."""

    def __init__(self, vocab=12, hidden=6):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.emb.weight

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, m):
        self.emb = m

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, m):
        self.head = m

    def forward(self, ids):
        return self.head(torch.tanh(self.mix(self.emb(ids))))


IDS = [3, 7, 8, 11]


def test_the_rows_need_a_tied_model_and_start_as_the_base_model():
    m = Tied()
    ids = torch.tensor([[1, 3, 7, 2, 11]])
    before = m(ids).detach()
    rows = L.inject_rows(m, IDS)
    assert rows.delta.dtype == torch.float32 and rows.delta.requires_grad
    assert tuple(rows.delta.shape) == (4, 6) and not rows.delta.any()
    assert torch.equal(m(ids), before)  # delta = 0: exactly the base model
    # one parameter, shared by both sides; the model's parameters list it once
    assert m.get_input_embeddings().rows.delta is m.get_output_embeddings().rows.delta
    assert sum(p is rows.delta for p in m.parameters()) == 1
    untied = Tied()
    untied.head = nn.Linear(6, 12, bias=False)
    with pytest.raises(ValueError):
        L.inject_rows(untied, IDS)
    with pytest.raises(ValueError):
        L.inject_rows(Tied(), [3, 3])


def test_gradients_reach_the_delta_from_the_input_and_the_output_side():
    m = Tied()
    rows = L.inject_rows(m, IDS)
    emb, head = m.get_input_embeddings(), m.get_output_embeddings()
    # input side alone: only the rows whose tokens are in the input move
    emb(torch.tensor([[3, 1, 3]])).sum().backward()
    g = rows.delta.grad
    assert g[0].abs().sum() > 0 and not g[1:].any()
    assert torch.allclose(g[0], torch.full((6,), 2.0))  # token 3 twice, d(sum)/d(row) = 1
    rows.delta.grad = None
    # output side alone: d(logit of ids[2]) / d(delta[2]) is the hidden state
    x = torch.randn(5, 6)
    head(x)[:, IDS[2]].sum().backward()
    g = rows.delta.grad
    assert torch.allclose(g[2], x.sum(0)) and not g[[0, 1, 3]].any()
    rows.delta.grad = None
    # the whole model: the delta's gradient is the tied weight's on those rows (which sums
    # both sides), from a model with E + delta written into its one tied weight
    with torch.no_grad():
        rows.delta.normal_(0, 0.1)
    ids = torch.tensor([[1, 3, 7, 2, 11, 4]])
    target = torch.tensor([7, 2, 11, 3, 8, 0])
    nn.functional.cross_entropy(m(ids)[0], target).backward()
    ref = Tied()
    with torch.no_grad():
        for j, i in enumerate(IDS):
            ref.emb.weight[i] += rows.delta[j]
    nn.functional.cross_entropy(ref(ids)[0], target).backward()
    assert torch.allclose(rows.delta.grad, ref.emb.weight.grad[IDS], atol=1e-6)
    assert torch.allclose(m(ids), ref(ids), atol=1e-6)


def test_the_rows_are_saved_digested_and_merged_into_the_tied_weight(tmp_path):
    m = Tied()
    rows = L.inject_rows(m, IDS)
    L.randomize_rows(rows, 0.1, torch.Generator().manual_seed(3))
    ids = torch.tensor([[1, 3, 7, 8, 11, 4]])
    wrapped_out = m(ids).detach()
    state = L.adapter_state({}, rows)
    assert state[L.ROWS_IDS].tolist() == IDS and state[L.ROWS_DELTA].dtype == torch.float32
    L.save_adapter(tmp_path / "a.safetensors", state, {"embed_rows_ids": "3,7,8,11"})
    back, meta = L.load_adapter(tmp_path / "a.safetensors")
    assert T.tensors_sha256(back) == T.tensors_sha256(state)
    moved = dict(state, **{L.ROWS_DELTA: state[L.ROWS_DELTA] + 1e-6})
    assert T.tensors_sha256(moved) != T.tensors_sha256(state)  # the digest covers the delta
    assert L.module_names(back) == []
    fresh = Tied()
    assert L.merge_into(fresh, back, 2.0) == 0
    assert L.tied(fresh) and torch.allclose(fresh(ids), wrapped_out, atol=1e-6)
    # delta = 0 merged: the weight bit for bit, in bf16 too
    z = Tied().to(torch.bfloat16)
    w = z.emb.weight.detach().clone()
    L.merge_rows(z, torch.tensor(IDS), torch.zeros(4, 6))
    assert torch.equal(z.emb.weight, w)
    with pytest.raises(ValueError):
        L.split_rows({L.ROWS_DELTA: torch.zeros(4, 6)})  # ids missing


def test_row_similarity():
    r = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    s = L.row_similarity(r)
    assert s["norms"] == [1.0, 2.0, 1.0] and s["cos_max"] == pytest.approx(1.0)
    assert s["cos_min"] == pytest.approx(0.0) and s["max_abs_diff"] == pytest.approx(2.0)
    assert s["cos_first"] == pytest.approx([1.0, 1.0, 0.0])


# -- training and the merge check on a tiny Qwen2 ---------------------------------------------


def test_a_tiny_run_trains_the_rows_beside_lora(tmp_path):
    tok = _tok()
    real, _ = _sets(tmp_path, tok)
    s = T.Settings(model=_tiny_model(tmp_path, tok), batch_rows=4, token_budget=4096)
    dev = torch.device("cpu")
    quiet = lambda k, loss: None  # noqa: E731
    plain = T.train(s, "real", 1, real, tmp_path / "plain", dev, on_step=quiet)
    rows = T.train(s, "real", 1, real, tmp_path / "rows", dev, on_step=quiet,
                   embed_rows=T.EMBED_ROWS)
    ids = T.token_ids(tok, T.EMBED_ROWS)
    assert ids == [151644, 151645, 151657, 151658]
    # LoRA's init and count are untouched by the rows; the rows add 4 x hidden
    assert rows.init_sha256 == plain.init_sha256 and rows.trainable == plain.trainable
    er = rows.embed_rows
    assert er["ids"] == ids and er["tokens"] == list(T.EMBED_ROWS) and er["trainable"] == 4 * 32
    assert er["weight_decay"] == s.weight_decay == 0.0
    assert all(n > 0 for n in er["delta_norms"])  # every row moved: each is in the targets
    assert rows.dtypes["embedding rows delta"] == rows.dtypes["AdamW state (delta)"] == "float32"
    assert rows.losses != plain.losses
    state, meta = L.load_adapter(tmp_path / "rows" / "adapter.safetensors")
    assert state[L.ROWS_IDS].tolist() == ids and state[L.ROWS_DELTA].norm() > 0
    assert meta["embed_rows_ids"] == ",".join(map(str, ids))
    assert meta["embed_rows_tokens"].split() == list(T.EMBED_ROWS)
    assert T.tensors_sha256(state) == rows.adapter_sha256
    # main's records carry no base-arm fields; the base arm's do
    assert "embed_rows" not in json.loads((tmp_path / "plain" / "train.json").read_text())
    assert "tokenizer" not in L.load_adapter(tmp_path / "plain" / "adapter.safetensors")[1]
    assert json.loads((tmp_path / "rows" / "train.json").read_text())["embed_rows"]["ids"] == ids
    again = T.train(s, "real", 1, real, tmp_path / "again", dev, on_step=quiet,
                    embed_rows=T.EMBED_ROWS)
    assert again.adapter_sha256 == rows.adapter_sha256  # deterministic with the rows too


def test_the_random_merge_check_with_rows_fails_when_the_rows_are_not_merged(tmp_path,
                                                                             monkeypatch):
    tok = _tok()
    model = _tiny_model(tmp_path, tok)
    s = T.Settings(model=model)
    texts = ["<|im_start|>user\nWhat is 2 plus 3?<|im_end|>\n<|im_start|>assistant\n",
             "<tool_call>\n{}\n</tool_call><|im_end|>"]
    ok = R.merge_check_random_b(s, model, texts, torch.device("cpu"), None, T.EMBED_ROWS)
    rows = ok["rows"]
    assert ok["passes"] and list(rows) == ["merged", f"merged, {R.MERGE_CONTRAST:g} x the scaling",
                                           R.ROWS_LEFT_OUT, "base (nothing merged)"]
    assert rows["merged"]["relative"] < 1e-4 and rows[R.ROWS_LEFT_OUT]["relative"] > 1e-2
    monkeypatch.setattr(L, "merge_rows", lambda *a, **k: 0)  # rows silently not written
    assert not R.merge_check_random_b(s, model, texts, torch.device("cpu"), None,
                                      T.EMBED_ROWS)["passes"]


def test_the_memory_step_runs_with_the_rows(tmp_path):
    from tinytools import sft

    tok = _tok()
    model = _tiny_model(tmp_path, tok)
    tools = [{"name": "add", "description": "Add.", "parameters": {
        "type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}]
    recs = [sft.render(f"r{i}", tools, [{"role": "user", "content": "add " * (5 * i + 1)},
                                        {"role": "assistant", "content": "ok"}], tok)
            for i in range(8)]
    s = T.Settings(model=model, batch_rows=4, epochs=1,
                   token_budget=2 * max(r["n_tokens"] for r in recs))
    mem = R.memory_probe(s, recs, torch.device("cpu"), None, T.EMBED_ROWS)
    assert mem["fits"] and [a["ran"] for a in mem["attempts"]] == [True]


# -- the base arm's prompts and stops -------------------------------------------------------


def _cached(model_id):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} tokenizer not in the local cache")


def test_the_base_tokenizer_splits_text_as_the_instruct_one():
    base, inst = _cached("Qwen/Qwen2.5-0.5B"), _cached("Qwen/Qwen2.5-0.5B-Instruct")
    assert T.tokenizers_match(base, inst)
    assert base.chat_template != inst.chat_template  # only the default system line differs
    assert base.eos_token == "<|endoftext|>" and inst.eos_token == "<|im_end|>"
    tiny = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: '{"model": 1}'))
    assert not T.tokenizers_match(base, tiny)
    with pytest.raises(ValueError):
        T.token_ids(inst, ["<|im_end|>", "<not a token>"])


def test_replies_end_at_both_tokens_whatever_the_base_config_says():
    from tinytools.engine import decoding

    tok = _cached("Qwen/Qwen2.5-0.5B-Instruct")
    for own in (151643, [151643], None):
        lm = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=own))
        ids = R.set_stops(lm, tok, T.STOP_TOKENS)
        assert set(ids) == {151643, 151645}
        assert set(decoding(tok, lm, 1, 0.0, 512).stop) == {151643, 151645}
    # main sets nothing: the model as loaded (the Instruct config lists both already)
    lm = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
    assert R.set_stops(lm, tok, ()) == [] and lm.generation_config.eos_token_id == [151645, 151643]


def test_same_prompts():
    assert R.same_prompts(["a", "b", "c"], ["a", "x", "c"]) == 2
    with pytest.raises(ValueError):
        R.same_prompts(["a"], [])


def test_the_base_arms_check_sections_fit_74_columns(tmp_path):
    """Fabricated records at full size: every line within 74 columns (tests raise)."""
    from tinytools.report import Report

    pr = {"model": "Qwen/Qwen2.5-1.5B", "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
          "splits_as_model": True, "vocabulary": 151665,
          "eos": {"model's own": "<|endoftext|>", "this arm's": "<|im_end|>"},
          "sets": [{"set": "training rows (real.jsonl)", "n": 8504, "as_instruct": 8504,
                    "with_own_template": 4198},
                   {"set": "BFCL eval items", "n": 1116, "as_instruct": 1116,
                    "with_own_template": 0}], "passes": True}
    st = {"model_eos": [151643], "wanted": [151645, 151643], "engine_stop": [151643, 151645],
          "names": {"151643": "<|endoftext|>", "151645": "<|im_end|>"}, "passes": True}
    sim = {"norms": [0.414] * 4, "cos_min": 1.0, "cos_max": 1.0, "max_abs_diff": 1.2e-4,
           "cos_first": [1.0] * 4}
    rw = {"model": "Qwen/Qwen2.5-1.5B", "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
          "tokens": list(T.EMBED_ROWS), "ids": [151644, 151645, 151657, 151658], "tied": True,
          "rows": 151936, "hidden": 1536, "dtype": "bfloat16", "median_norm": 1.032,
          "base": sim, "instruct_rows": dict(sim, cos_min=0.92, cos_max=0.95),
          "threshold": R.ROW_IDENTICAL, "identical": True, "passes": True}
    rep = Report("t", out=tmp_path / "t.txt")
    R._profile_lines(rep, R.mode(False, ROOT, profile="base"))
    R._print_base_prompts(rep, pr)
    R._print_stops(rep, st)
    R._print_rows(rep, rw)
    R._r4c_base_checks(rep, {"base_prompts": pr, "stops": st, "embed_rows": rw,
                             "merge_b0": {"rows": 4, "rows_equal": 4}})
    R._print_trained_rows(rep, {"tokens": list(T.EMBED_ROWS), "ids": rw["ids"],
                                "trainable": 6144, "delta_norms": [0.5] * 4,
                                "before": sim, "after": dict(sim, cos_min=0.2)})
    rep.close()
    text = (tmp_path / "t.txt").read_text()
    assert "4 x 1,536 = 6,144" in text and "6,144 parameters" in text
    assert all(len(x) <= 74 for x in text.splitlines())


# -- the main run's adapters load and merge as before ----------------------------------------


def _main_adapters():
    found = sorted(MAIN_RUN.glob("*/adapter.safetensors"))
    if not found:
        pytest.skip("the main run's adapters are not on this machine")
    return found


def test_the_main_runs_adapters_load_as_before():
    for path in _main_adapters():
        state, meta = L.load_adapter(path)
        record = json.loads((path.parent / "train.json").read_text())
        assert L.split_rows(state)[1] is None and "embed_rows_ids" not in meta
        assert len(L.module_names(state)) == 28 * 7  # every layer's seven projections
        assert T.tensors_sha256(state) == record["adapter_sha256"]  # the digest as it was
        assert "embed_rows" not in record and "tokenizer" not in record


def _tree(names_shapes):
    """A module tree holding plain nn.Linear at the given dotted names (bf16 weights)."""
    root = nn.Module()
    for name, (out_f, in_f) in names_shapes.items():
        parent = root
        *path, leaf = name.split(".")
        for part in path:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        lin = nn.Linear(in_f, out_f, bias=False).to(torch.bfloat16)
        with torch.no_grad():
            lin.weight.normal_(0, 0.02, generator=torch.Generator().manual_seed(len(name)))
        parent.add_module(leaf, lin)
    return root


def test_the_main_runs_adapters_merge_as_before():
    """Layer 0's seven modules of a real adapter: merge_into writes exactly W + (alpha/r) BA
    in float32 cast once, the formula it used before the base arm."""
    path = _main_adapters()[0]
    state, meta = L.load_adapter(path)
    names = [n for n in L.module_names(state) if n.startswith("model.layers.0.")]
    assert len(names) == 7
    sub = {k: v for k, v in state.items() if k.rsplit(".", 1)[0] in names}
    shapes = {n: (state[f"{n}.B"].shape[0], state[f"{n}.A"].shape[1]) for n in names}
    model = _tree(shapes)
    before = {n: model.get_submodule(n).weight.detach().clone() for n in names}
    scaling = float(meta["alpha"]) / int(meta["rank"])
    assert L.merge_into(model, sub, scaling) == 7
    for n in names:
        want = (before[n].float() + scaling * (sub[f"{n}.B"].float() @ sub[f"{n}.A"].float())
                ).to(torch.bfloat16)
        assert torch.equal(model.get_submodule(n).weight, want)


def test_a_main_train_record_keeps_its_fields():
    path = MAIN_RUN / "real-seed1" / "train.json"
    if not path.exists():
        pytest.skip("the main run's records are not on this machine")
    keys = list(json.loads(path.read_text()))
    res = T.Result(**{k: v for k, v in json.loads(path.read_text()).items()})
    assert list(T.result_record(res)) == keys


# -- base.toml ----------------------------------------------------------------------------------


def _base():
    return lr.parse_experiment((ROOT / "experiments" / "base.toml").read_text(), "base")


def test_the_base_experiment_is_mains_stages_with_the_profile():
    e, m = _base(), lr.parse_experiment((ROOT / "experiments" / "main.toml").read_text(), "main")
    parser = build_parser()

    def stage(c):
        if c.startswith("echo") or c.startswith("export"):
            return c
        words = shlex.split(c.replace('"$TINYTOOLS_DEADLINE_EPOCH"', "1700000000"))
        args = parser.parse_args(words[3:])
        return (words[3], tuple(w for w in words[4:] if w not in ("--profile", "base")
                                and w not in ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct")
                                and not w.startswith("outputs/")), getattr(args, "profile", None))

    base_stages, main_stages = [stage(c) for c in e.commands], [stage(c) for c in m.commands]
    # the same commands with the same flags, in the same order
    assert [x if isinstance(x, str) else x[:2] for x in base_stages] == \
        [x if isinstance(x, str) else x[:2] for x in main_stages]
    for c, st in zip(e.commands, base_stages, strict=True):
        if isinstance(st, tuple) and st[0] not in ("toolace-convert", "sft-data", "engine-check"):
            assert st[2] == "base", c  # every SFT command and every rescore
            assert "--quick" not in c
        if isinstance(st, tuple) and st[0] in ("rescore", "bfcl-rescore"):
            assert "--model Qwen/Qwen2.5-1.5B " in c and "Qwen2.5-1.5B-Instruct" not in c
    assert (e.post, e.instance_type, e.cap_usd, e.setup_record) == (
        9, "gpu_1x_a100_sxm4", 12.0, "outputs/base/setup.json")


def test_the_base_experiment_fetches_the_base_arm_only():
    e = _base()
    want = ["outputs/base/checks.json", "outputs/base/setup.json",
            "outputs/base/untrained/score.json", "outputs/base/real-seed2/train.json",
            "outputs/base/control-seed3/losses.jsonl",
            "outputs/base/real-seed1/adapter.safetensors",
            "outputs/base/real-seed1-repeat/adapter.safetensors",
            f"outputs/base/untrained/{R.OURS_FILE}", f"outputs/base/{R.FLOOR}/{R.SAMPLE_FILE}",
            "outputs/base/r1-prompt.txt", "outputs/sft-base.txt", "outputs/sft-check-base.txt",
            "outputs/sft-pair-seed2-base.txt", "outputs/sft-score-untrained-base.txt",
            f"outputs/sft-score-{R.FLOOR}-base.txt", "outputs/sft-train-real-seed1-repeat-base.txt",
            "outputs/score-Qwen2.5-1.5B-test-greedy-limitall-replies.jsonl",
            "outputs/score-Qwen2.5-1.5B-test-greedy-limitall.txt",
            "outputs/score-Qwen2.5-1.5B-train-sample-samples8-t1-limitall.txt",
            "outputs/bfcl-Qwen2.5-1.5B-eval-greedy-limitall.txt",
            "outputs/sft-data.txt", "outputs/engine-check-quick.txt"]
    for f in want:
        assert any(Path(f).match(p) for p in e.fetch), f
    for f in ["outputs/base/b0-adapter.safetensors", "outputs/sft.txt", "outputs/r1-prompt.txt",
              "outputs/score-Qwen2.5-1.5B-Instruct-test-greedy-limitall.txt",
              "outputs/bfcl-Qwen2.5-1.5B-Instruct-eval-greedy-limitall.txt",
              "outputs/main/untrained/score.json", "outputs/sft-check.txt"]:
        assert not any(Path(f).match(p) for p in e.fetch), f


def test_the_base_experiment_stays_within_both_caps_with_the_ledger():
    e = _base()
    ledger = lr.read_ledger(ROOT / "runs" / "ledger.csv")
    types = {"gpu_1x_a100_sxm4": {"instance_type": {"price_cents_per_hour": 199},
                                  "regions_with_capacity_available": [{"name": "us-east-1"}]}}
    est = lr.make_estimate(e, types, ledger)
    assert est.worst == math.ceil(1.99 * e.max_hours * 100) / 100 <= 12.0
    assert est.spent == 0.0 and est.post == 9  # nothing charged to "base" yet
    assert est.post_spent == pytest.approx(sum(float(r["cost_usd"]) for r in ledger
                                               if r["post"] == "9"))
    assert est.within_cap and est.expected <= est.worst
    assert e.expected_hours < e.max_hours
