"""Check the from-scratch implementation against HF transformers, piece by piece.

Ordered from smallest to largest unit, so `uv run pytest -x` stops at the first thing to fix.
Everything runs in float32 on CPU so tolerances can be tight.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama

from tiny_infer import loader, model
from tiny_infer.generate import generate, sample

ATOL = 1e-4
PROMPT = "The quick brown fox jumps over the lazy dog. In a shocking turn of events,"


@pytest.fixture(scope="session")
def ours():
    return loader.load()


@pytest.fixture(scope="session")
def hf():
    m = AutoModelForCausalLM.from_pretrained(
        loader.model_dir(), dtype=torch.float32, attn_implementation="eager"
    )
    return m.eval()


@pytest.fixture(scope="session")
def tokens():
    return torch.tensor(loader.load_tokenizer().encode(PROMPT))


def close(name, got, want, atol=ATOL, rtol=0.0):
    # rtol matters for the residual stream: a few entries grow to ~1e4 ("massive
    # activations"), where fp32 rounding alone is ~1e-3 absolute. Real bugs are O(1).
    assert got.shape == want.shape, f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    diff = (got - want).abs()
    if (diff > atol + rtol * want.abs()).any():
        i = diff.argmax()
        pytest.fail(
            f"{name}: max abs error {diff.max().item():.3g} where expected value is "
            f"{want.flatten()[i].item():.4g} (atol {atol}, rtol {rtol})"
        )


@torch.inference_mode()
def test_rms_norm(ours, hf):
    cfg, params = ours
    x = torch.randn(7, cfg.dim)
    close("rms_norm", model.rms_norm(x, params["norm.weight"], cfg.norm_eps), hf.model.norm(x))


@torch.inference_mode()
def test_rope_cos_sin(ours, hf):
    cfg, _ = ours
    pos = torch.arange(11)
    cos, sin = model.rope_cos_sin(cfg, pos)
    want_cos, want_sin = hf.model.rotary_emb(torch.zeros(1), pos[None])
    close("cos", cos, want_cos[0])
    close("sin", sin, want_sin[0])


@torch.inference_mode()
def test_apply_rope(ours, hf):
    cfg, _ = ours
    pos = torch.arange(11)
    cos, sin = hf.model.rotary_emb(torch.zeros(1), pos[None])
    q = torch.randn(1, cfg.n_heads, 11, cfg.head_dim)
    want, _ = modeling_llama.apply_rotary_pos_emb(q, q, cos, sin)
    close("apply_rope", model.apply_rope(q[0], cos[0], sin[0]), want[0])


@torch.inference_mode()
def test_layers(ours, hf, tokens):
    """Pinpoints the first decoder layer whose output diverges from HF."""
    cfg, params = ours
    _, layer_outputs = model.forward(tokens, params, cfg)
    want = hf(tokens[None], output_hidden_states=True).hidden_states
    assert len(layer_outputs) == cfg.n_layers
    # HF's hidden_states[i + 1] is the output of layer i, except the last entry,
    # which has the final norm applied. That one is covered by test_logits.
    for i in range(cfg.n_layers - 1):
        close(f"layer {i} output", layer_outputs[i], want[i + 1][0], atol=1e-3, rtol=1e-4)


@torch.inference_mode()
def test_logits(ours, hf, tokens):
    cfg, params = ours
    logits, _ = model.forward(tokens, params, cfg)
    want = hf(tokens[None]).logits[0]
    close("logits", logits, want, atol=1e-3)
    assert torch.equal(logits.argmax(-1), want.argmax(-1)), "argmax tokens differ"


def test_sample_greedy():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    assert sample(logits, temperature=0.0) == 1


def test_sample_top_k_and_top_p():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([5.0, 4.0, 3.0, -10.0, -10.0])
    picks = {sample(logits, temperature=1.0, top_k=2, generator=g) for _ in range(200)}
    assert picks == {0, 1}, f"top_k=2 should only ever pick tokens 0 and 1, got {picks}"
    # Token 0 alone carries ~0.67 of the mass, so top_p=0.5 must always pick it.
    picks = {sample(logits, temperature=1.0, top_p=0.5, generator=g) for _ in range(200)}
    assert picks == {0}, f"top_p=0.5 should only pick token 0, got {picks}"


@torch.inference_mode()
def test_greedy_generation_matches_hf(ours, hf, tokens):
    cfg, params = ours
    n = 20
    got = list(generate(tokens.tolist(), params, cfg, max_new_tokens=n))
    want = hf.generate(tokens[None], max_new_tokens=n, do_sample=False, min_new_tokens=n)
    assert got == want[0, len(tokens):].tolist()
