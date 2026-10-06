"""Sampling and the decode loop.

Usage:
    uv run python -m tiny_infer.generate "The capital of France is" --max-new-tokens 40
"""

import argparse
import sys
import time
from collections.abc import Iterator

import torch

from tiny_infer import loader
from tiny_infer.model import Params, forward


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """logits: [V] -> next token id

    temperature == 0  -> greedy (argmax)
    otherwise         -> scale by 1/temperature, keep only the top_k largest (if top_k > 0),
                         then keep the smallest set whose probability mass >= top_p
                         (if top_p < 1), renormalize, and draw with torch.multinomial.
    """
    raise NotImplementedError


def generate(
    prompt_ids: list[int],
    params: Params,
    cfg: loader.Config,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int | None = None,
    eos_id: int | None = None,
) -> Iterator[int]:
    """Yield new token ids one at a time.

    For now: no KV cache. Run the full forward over the whole sequence every step and
    sample from the last position's logits. Slow on purpose: step 2 of the roadmap is
    adding a KV cache and measuring the difference.
    Stop after max_new_tokens, or right after yielding eos_id.
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=loader.MODEL_ID)
    args = ap.parse_args()

    cfg, params = loader.load(args.model)
    tok = loader.load_tokenizer(args.model)
    prompt_ids = tok.encode(args.prompt)

    print(args.prompt, end="", flush=True)
    out: list[int] = []
    printed = ""
    start = time.perf_counter()
    with torch.inference_mode():
        for t in generate(
            prompt_ids, params, cfg, args.max_new_tokens,
            args.temperature, args.top_k, args.top_p, args.seed, tok.eos_token_id,
        ):
            out.append(t)
            # Decode the whole tail each time: a single token can be a partial UTF-8 char.
            text = tok.decode(out)
            if not text.endswith("�"):
                print(text[len(printed):], end="", flush=True)
                printed = text
    elapsed = time.perf_counter() - start
    print(f"\n\n[{len(out)} tokens in {elapsed:.2f}s, {len(out) / elapsed:.1f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
