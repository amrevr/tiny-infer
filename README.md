# tiny-infer

A from-scratch LLM inference engine, built to learn how inference works.
Python first (plain tensor ops over a dict of weights), then ported to C++.

Model: [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), a standard Llama
architecture (GQA, RoPE, RMSNorm, SwiGLU, tied embeddings). Anything that works here
works on Llama and Qwen with small changes.

Everything runs inside an OrbStack (Docker) container. The venv and the model cache live in
named volumes, so nothing is installed on the Mac itself.

```
docker compose build                                   # once, or after editing the Dockerfile
docker compose run --rm dev uv run pytest -x           # parity vs HF, smallest piece first
docker compose run --rm dev uv run python -m tiny_infer.generate "The capital of France is"
docker compose run --rm dev bash                       # interactive shell
```

The container is Linux, so MPS/Metal isn't available in it. Roadmap steps that need the GPU
(step 3, Metal kernels in step 8) will need a decision about running on the host.

## Layout

| File | What | Who writes it |
|---|---|---|
| `src/tiny_infer/loader.py` | download, config, weights as `dict[str, Tensor]` | provided |
| `src/tiny_infer/model.py` | rms_norm, RoPE, GQA attention, SwiGLU, forward | **you** |
| `src/tiny_infer/generate.py` | sampling + decode loop (CLI provided) | **you** |
| `tests/test_parity.py` | per-op and per-layer checks against HF | provided |

## Roadmap

1. **Forward pass.** Make `pytest` green. Reference: about 23 tok/s on an M5 CPU in fp32 with no cache (measured on the host; the container may differ).
2. **KV cache.** Split prefill from decode. Measure tok/s vs sequence length before and
   after; explain why decode is memory-bandwidth bound (bytes of weights read per token).
3. **Devices and dtypes.** MPS, bf16/fp16. Where does precision loss show up?
   (Hint: some residual-stream entries reach ~1.5e4; see the comment in the tests.)
4. **Quantization.** Weight-only int8, then 4-bit groupwise. Track speed and perplexity.
5. **Batching.** Static batches with padding, then **continuous batching** with a scheduler.
6. **Paged KV cache.** Block tables, an allocator, preemption (the vLLM idea).
7. **Serving.** OpenAI-compatible HTTP endpoint, streaming, TTFT / TPOT metrics.
8. **C++ port.** Safetensors parser, the same forward pass, then Metal or NEON kernels.
   Python tests become the oracle: dump activations and diff them.
9. Extras: speculative decoding, prefix caching, Llama 3 RoPE scaling, a BPE tokenizer.
