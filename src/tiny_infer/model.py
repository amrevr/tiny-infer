"""A Llama-architecture forward pass, written as plain functions over a dict of weights.

Rules for this file:
  * Only basic tensor ops: matmul, elementwise math, softmax, reshape/transpose, cat, etc.
  * No torch.nn modules, no torch.nn.functional (not even scaled_dot_product_attention),
    and no HF modeling code. Every op should be visible, so the C++ port is mechanical.
  * Single sequence, no batch dimension (batching comes later).

Shape legend:
  T  = sequence length      D  = cfg.dim           V = cfg.vocab_size
  H  = cfg.n_heads          KV = cfg.n_kv_heads    Hd = cfg.head_dim
  F  = cfg.ffn_dim

Run `uv run pytest -x` to check each piece against HF transformers, in order.
"""

import torch

from tiny_infer.loader import Config

Params = dict[str, torch.Tensor]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., D], weight: [D] -> [..., D]

    y = x / sqrt(mean(x^2) + eps) * weight
    """
    raise NotImplementedError


def rope_cos_sin(cfg: Config, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """positions: [T] (int) -> cos, sin: each [T, Hd]

    Frequencies: inv_freq[i] = theta ** (-2i / Hd) for i in [0, Hd/2).
    Angle for (position p, pair i) = p * inv_freq[i].
    Hint: HF Llama weights expect the "half-split" layout, where dimension j is paired
    with dimension j + Hd/2 (not adjacent dims 2i, 2i+1). Lay out cos/sin to match.
    """
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [n_heads, T, Hd], cos/sin: [T, Hd] -> [n_heads, T, Hd]

    Rotate each (x[j], x[j + Hd/2]) pair by its angle.
    """
    raise NotImplementedError


def attention(
    x: torch.Tensor, params: Params, layer: int, cfg: Config, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: [T, D] (already normed) -> [T, D]

    1. Project to q [H, T, Hd], k [KV, T, Hd], v [KV, T, Hd].
    2. Apply RoPE to q and k (not v).
    3. Grouped-query attention: each KV head is shared by H // KV query heads.
    4. Scaled dot-product with a causal mask, softmax, weighted sum of v.
    5. Merge heads back to [T, H*Hd] and apply o_proj.
    """
    raise NotImplementedError


def mlp(x: torch.Tensor, params: Params, layer: int) -> torch.Tensor:
    """x: [T, D] (already normed) -> [T, D]

    SwiGLU: down( silu(gate(x)) * up(x) ), where silu(z) = z * sigmoid(z).
    """
    raise NotImplementedError


def forward(
    tokens: torch.Tensor, params: Params, cfg: Config
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """tokens: [T] (int64) -> (logits [T, V], layer_outputs)

    layer_outputs[i] is the residual stream [T, D] after decoder layer i, before the
    final norm. The tests use it to tell you exactly which layer first diverges.

    Each decoder layer (pre-norm residual):
        h = h + attention(rms_norm(h, input_layernorm), ...)
        h = h + mlp(rms_norm(h, post_attention_layernorm), ...)
    Then: logits = rms_norm(h, norm) @ lm_head.T
    """
    raise NotImplementedError
