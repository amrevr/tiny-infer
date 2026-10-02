"""Plumbing: download a Llama-architecture checkpoint and load it as a flat dict of tensors.

Weight names follow the HF checkpoint with the leading "model." stripped:
    embed_tokens.weight                              [V, D]
    layers.{i}.input_layernorm.weight                [D]
    layers.{i}.self_attn.q_proj.weight               [H*Hd, D]
    layers.{i}.self_attn.k_proj.weight               [KV*Hd, D]
    layers.{i}.self_attn.v_proj.weight               [KV*Hd, D]
    layers.{i}.self_attn.o_proj.weight               [D, H*Hd]
    layers.{i}.post_attention_layernorm.weight       [D]
    layers.{i}.mlp.gate_proj.weight                  [F, D]
    layers.{i}.mlp.up_proj.weight                    [F, D]
    layers.{i}.mlp.down_proj.weight                  [D, F]
    norm.weight                                      [D]
    lm_head.weight                                   [V, D]  (aliases embed_tokens when tied)

Linear weights are stored [out_features, in_features], so a projection is `x @ W.T`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"


@dataclass(frozen=True)
class Config:
    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_dim: int
    rope_theta: float
    norm_eps: float
    max_seq_len: int

    @classmethod
    def from_hf(cls, hf: dict) -> "Config":
        assert hf.get("hidden_act", "silu") == "silu", hf.get("hidden_act")
        # Llama 3.x checkpoints use "llama3" rope scaling; supporting it is a later exercise.
        assert not hf.get("rope_scaling"), f"rope_scaling not supported yet: {hf['rope_scaling']}"
        return cls(
            vocab_size=hf["vocab_size"],
            dim=hf["hidden_size"],
            n_layers=hf["num_hidden_layers"],
            n_heads=hf["num_attention_heads"],
            n_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            head_dim=hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"],
            ffn_dim=hf["intermediate_size"],
            rope_theta=hf.get("rope_theta", 10000.0),
            norm_eps=hf["rms_norm_eps"],
            max_seq_len=hf["max_position_embeddings"],
        )


def model_dir(model_id: str = MODEL_ID) -> Path:
    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def load(
    model_id: str = MODEL_ID, dtype: torch.dtype = torch.float32, device: str = "cpu"
) -> tuple[Config, dict[str, torch.Tensor]]:
    path = model_dir(model_id)
    cfg = Config.from_hf(json.loads((path / "config.json").read_text()))
    raw: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        raw.update(load_file(f))
    params = {k.removeprefix("model."): v.to(device=device, dtype=dtype) for k, v in raw.items()}
    params.setdefault("lm_head.weight", params["embed_tokens.weight"])
    return cfg, params


def load_tokenizer(model_id: str = MODEL_ID):
    # The tokenizer isn't the interesting part of inference, so borrow HF's for now.
    # (Writing a byte-level BPE encoder is a nice side quest for the C++ port.)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir(model_id))
