"""The SFT trainer: its plan (shuffle, steps, micro-batches, schedule), the seed pair, the
deadline arithmetic, LoRA's precision and merge, and a tiny end-to-end run twice."""

import json
import math

import pytest
import torch

from tinytools import train as T
from tinytools.lora import (adapter_state, inject_lora, load_adapter, merge_into,
                            merged_weight, module_names, save_adapter, trainable_parameters)


def test_the_pre_registered_settings():
    s = T.MAIN
    assert (s.model, s.rank, s.alpha, s.dropout, s.epochs) == (
        "Qwen/Qwen2.5-1.5B-Instruct", 16, 32.0, 0.0, 2)
    assert s.targets == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                         "down_proj")
    assert (s.lr, s.betas, s.weight_decay, s.warmup_ratio, s.batch_rows) == (
        2e-4, (0.9, 0.999), 0.0, 0.03, 16)
    assert s.max_length == 3072 and s.token_budget >= s.max_length and s.rows is None
    assert not s.grad_checkpointing
    q = T.QUICK  # quick changes only the model, the rows and the budget
    assert all(getattr(q, f) == getattr(s, f) for f in T.PAIRED if f not in ("model", "rows"))


def test_shuffle_is_a_seeded_permutation_per_epoch():
    a = T.shuffle_order(100, 1, 0)
    assert sorted(a) == list(range(100))
    assert a == T.shuffle_order(100, 1, 0)
    assert a != T.shuffle_order(100, 2, 0) and a != T.shuffle_order(100, 1, 1)


def test_steps_cover_every_row_once_per_epoch():
    steps = T.plan_steps(8504, 1, 2, 16)
    assert len(steps) == 2 * 532  # 531 full steps and one of 8 rows, per epoch
    for e in (0, 1):
        rows = [i for s in steps if s.epoch == e for i in s.rows]
        assert sorted(rows) == list(range(8504))
    assert [len(s.rows) for s in steps[:532]] == [16] * 531 + [8]


def test_seed_pair_same_first_step_batches_and_init():
    """Two runs with seed 1 (a real run and its control twin, or a run and its repeat)
    have identical first-step batches and LoRA init; another seed differs."""
    lengths = [((i * 37) % 900) + 100 for i in range(64)]
    one = T.plan_steps(64, 1, 2, 16)
    assert one == T.plan_steps(64, 1, 2, 16)
    assert T.micro_batches(one[0].rows, lengths, 2048) == T.micro_batches(
        T.plan_steps(64, 1, 2, 16)[0].rows, lengths, 2048)
    assert one[0] != T.plan_steps(64, 2, 2, 16)[0]

    def init(seed):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
        model[0].q_proj = torch.nn.Linear(8, 8)  # a target name inside the model
        w = inject_lora(model, ["q_proj"], rank=2, alpha=4,
                        generator=torch.Generator().manual_seed(seed))
        return adapter_state(w)

    a, b, c = init(1), init(1), init(2)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["0.q_proj.A"], c["0.q_proj.A"])
    assert torch.count_nonzero(a["0.q_proj.B"]) == 0


def test_micro_batches_respect_the_budget():
    lengths = [3000, 2900, 500, 400, 1500, 1400, 300, 200]
    mbs = T.micro_batches(tuple(range(8)), lengths, 3072)
    assert sorted(i for b in mbs for i in b) == list(range(8))
    for b in mbs:
        assert len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 3072
        assert [lengths[i] for i in b] == sorted((lengths[i] for i in b), reverse=True)
    assert mbs[0] == [0] and mbs[1] == [1]  # the long rows alone
    assert T.padded_tokens(mbs, lengths) == sum(len(b) * max(lengths[i] for i in b) for b in mbs)
    # a step is the same rows whatever the budget
    assert sorted(i for b in T.micro_batches(tuple(range(8)), lengths, 100_000) for i in b) == \
        list(range(8))


def test_schedule_is_hugging_faces_cosine_with_warmup():
    from transformers import get_cosine_schedule_with_warmup

    total = 1064
    warm = T.warmup_steps(total, 0.03)
    assert warm == 32  # ceil(31.92), as HF's Trainer computes it from warmup_ratio
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    hf = get_cosine_schedule_with_warmup(opt, warm, total)
    for k in range(total):
        assert T.lr_multiplier(k, warm, total) == pytest.approx(opt.param_groups[0]["lr"],
                                                                abs=1e-12)
        opt.step()
        hf.step()
    assert T.lr_multiplier(0, warm, total) == 0.0
    assert T.lr_multiplier(warm, warm, total) == 1.0


def test_labels_and_lengths():
    assert T.labels_for([5, 6, 7], [False, True, True]) == [-100, 6, 7]
    with pytest.raises(ValueError):
        T.labels_for([1, 2], [True])
    assert T.rows_over([3072, 3073, 10], 3072) == 1
    assert T.same_rows([{"id": "a"}, {"id": "b"}], [{"id": "a"}, {"id": "b"}])
    assert not T.same_rows([{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}])


def test_deadline_estimate():
    prior = T.estimate_seconds([], [], 1_000_000, 2, 0, False, 900)
    assert prior == pytest.approx(2 * T.PRIOR_S_PER_TOKEN * 1_000_000)
    assert T.estimate_seconds([], [], 1_000_000, 1, 0, True, 900) == pytest.approx(
        1.4 * T.PRIOR_S_PER_TOKEN * 1_000_000)
    # measured: the slowest rate and the slowest scoring
    assert T.estimate_seconds([1e-4, 2e-4], [600, 800], 1000, 2, 2, False, 900) == \
        pytest.approx(2 * 0.2 + 2 * 800)
    assert T.estimate_seconds([], [], 0, 0, 1, False, 900) == 900
    assert T.fits(0, None, 1e9)
    assert T.fits(0, 1000, 600, 100) and not T.fits(0, 1000, 600, 101)  # 1.5 x 600 + keep


# -- LoRA: precision and merging ----------------------------------------------------------


class Block(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False, dtype=dtype)
        self.down_proj = torch.nn.Linear(8, 4, bias=True, dtype=dtype)

    def forward(self, x):
        return self.down_proj(torch.relu(self.q_proj(x)))


def test_adapters_and_their_optimizer_state_are_fp32_on_bf16_weights():
    torch.manual_seed(0)
    model = Block(torch.bfloat16)
    wrapped = inject_lora(model, ["q_proj", "down_proj"], rank=2, alpha=4,
                          generator=torch.Generator().manual_seed(1))
    params = trainable_parameters(model)
    assert all(p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=1e-3)
    out = model(torch.randn(3, 8, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    opt.step()
    a = wrapped["q_proj"]
    assert a.A.grad.dtype == torch.float32 and opt.state[a.A]["exp_avg"].dtype == torch.float32
    assert a.base.weight.dtype == torch.bfloat16 and not a.base.weight.requires_grad
    # a step far below bf16's spacing at A's scale still moves the fp32 A
    before = a.A.detach().clone()
    with torch.no_grad():
        a.A.add_(1e-6)
    assert not torch.equal(a.A, before)


def test_merge_matches_the_adapter_and_b0_is_the_base(tmp_path):
    torch.manual_seed(0)
    model = Block()
    x = torch.randn(5, 8)
    base_out = model(x).detach()
    wrapped = inject_lora(model, ["q_proj", "down_proj"], rank=2, alpha=4,
                          generator=torch.Generator().manual_seed(3))
    assert torch.equal(model(x), base_out)  # B = 0: the base model
    with torch.no_grad():
        for w in wrapped.values():
            w.B.normal_()
    lora_out = model(x).detach()
    save_adapter(tmp_path / "a.safetensors", adapter_state(wrapped), {"k": "v"})
    state, meta = load_adapter(tmp_path / "a.safetensors")
    assert meta == {"k": "v"} and sorted(module_names(state)) == ["down_proj", "q_proj"]
    fresh = Block()
    fresh.load_state_dict({k.replace(".base", ""): v for k, v in model.state_dict().items()
                           if not k.endswith((".A", ".B"))})
    assert merge_into(fresh, state, 2.0) == 2
    assert torch.allclose(fresh(x), lora_out, atol=1e-5)
    # B = 0 merged: every weight unchanged, bit for bit (also in bf16)
    w = torch.randn(4, 8).to(torch.bfloat16)
    assert torch.equal(merged_weight(w, torch.randn(2, 8), torch.zeros(4, 2), 2.0), w)
    with pytest.raises(ValueError):
        merge_into(Block(), {"q_proj.A": torch.zeros(2, 8)}, 2.0)


def test_adapters_compare_by_their_tensors(tmp_path):
    state = {"m.A": torch.arange(6, dtype=torch.float32).reshape(2, 3),
             "m.B": torch.zeros(4, 2)}
    meta = {f"k{i}": str(i) for i in range(8)}  # safetensors writes these in no fixed order
    save_adapter(tmp_path / "1.safetensors", state, meta)
    save_adapter(tmp_path / "2.safetensors", {k: v.clone() for k, v in state.items()}, meta)
    one, two = load_adapter(tmp_path / "1.safetensors")[0], load_adapter(tmp_path / "2.safetensors")[0]
    assert T.tensors_sha256(one) == T.tensors_sha256(two) == T.tensors_sha256(state)
    moved = dict(state, **{"m.B": torch.zeros(4, 2).index_fill(0, torch.tensor([1]), 1e-30)})
    assert T.tensors_sha256(moved) != T.tensors_sha256(state)
    assert T.tensors_sha256({"m.A": state["m.A"].double(), "m.B": state["m.B"]}) != \
        T.tensors_sha256(state)


# -- a tiny model, trained twice (CPU) ------------------------------------------------------


def _tok():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen2.5-0.5B-Instruct tokenizer not in the local cache")


def _sets(tmp_path, tok):
    from tinytools import sft

    tools = [{"name": "add", "description": "Add two numbers.",
              "parameters": {"type": "object", "properties": {"a": {"type": "number"},
                                                              "b": {"type": "number"}},
                             "required": ["a", "b"]}},
             {"name": "echo", "description": "Echo text.",
              "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                             "required": ["text"]}}]
    real, ctl = [], []
    for i in range(10):
        user = {"role": "user", "content": f"What is {i} plus {i + 1}?"}
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "add", "arguments": {"a": i, "b": i + 1}}}]}
        twin = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "echo", "arguments": {"text": f"x{i}"}}}]}
        real.append(sft.render(f"row-{i}", tools, [user, call], tok))
        ctl.append(sft.render(f"row-{i}", tools, [user, twin], tok))
    (tmp_path / "real.jsonl").write_bytes(sft.jsonl_bytes(real))
    (tmp_path / "control.jsonl").write_bytes(sft.jsonl_bytes(ctl))
    return tmp_path / "real.jsonl", tmp_path / "control.jsonl"


def _tiny_model(tmp_path, tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096, tie_word_embeddings=True)
    path = tmp_path / "tiny"
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_a_tiny_run_trained_twice_matches_and_pairs_with_its_control(tmp_path):
    tok = _tok()
    real, control = _sets(tmp_path, tok)
    s = T.Settings(model=_tiny_model(tmp_path, tok), batch_rows=4, token_budget=4096)
    dev = torch.device("cpu")
    quiet = lambda k, loss: None  # noqa: E731
    a = T.train(s, "real", 1, real, tmp_path / "a", dev, on_step=quiet)
    b = T.train(s, "real", 1, real, tmp_path / "b", dev, on_step=quiet)
    c = T.train(s, "control", 1, control, tmp_path / "c", dev, real_path=real, on_step=quiet)
    assert a.steps == 2 * math.ceil(10 / 4) and a.rows_cut == 0
    assert a.losses == b.losses and a.adapter_sha256 == b.adapter_sha256
    assert T.tensors_sha256(load_adapter(tmp_path / "a" / "adapter.safetensors")[0]) == \
        a.adapter_sha256
    assert a.init_sha256 == c.init_sha256 and a.first_step_rows == c.first_step_rows
    assert a.losses != c.losses  # different targets
    assert a.dtypes["LoRA A, B"] == a.dtypes["AdamW state"] == "float32"
    assert a.losses[-1] < a.losses[0]
    rec = json.loads((tmp_path / "a" / "train.json").read_text())
    assert rec["scored"] == sum(json.loads(x)["n_trained"]
                                for x in real.read_text().splitlines())
    log = [json.loads(x) for x in (tmp_path / "a" / "losses.jsonl").read_text().splitlines()]
    assert [x["step"] for x in log] == list(range(1, a.steps + 1))
    assert log[0]["lr"] == 0.0  # HF's warmup starts at 0
    other = T.train(s, "real", 2, real, tmp_path / "d", dev, on_step=quiet)
    assert other.init_sha256 != a.init_sha256
    shuffled = tmp_path / "shuffled.jsonl"
    shuffled.write_text("".join(reversed(control.read_text().splitlines(keepends=True))))
    with pytest.raises(ValueError):  # a control set must hold the real rows in order
        T.train(s, "control", 1, shuffled, tmp_path / "e", dev, real_path=real, on_step=quiet)
