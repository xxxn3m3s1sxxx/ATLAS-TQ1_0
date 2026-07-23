# ATLAS — TQ1.0 Ternary Inference Engine

**v2.17.0 — July 2026**

CPU inference engine for 1.58-bit ternary-quantized LLMs.
Born from the question: *"How far can CPU-only + ternary go?"*
Answer: **~3 tok/s for 7B on DDR4. Not enough for interactive use.**

TQ1.0 remains the only production-grade format for ternary 1.58-bit quantization —
5 trits per byte, Base-3 encoded. The format and the packer (which auto-detects
29 architectures) are the real artifact, not the engine itself.

## Features

- **TQ1.0 Packer** (`pack_to_atlas.py`): HuggingFace safetensors → TQ1.0 (auto-detect)
- **5 matmul modes**: int8 (AVX2), int4 (nibble-unpack, ~26% faster on 7B), f32 bypass,
  ternary (vpsignb), TQ1-packed (chunked decode + SIMD)
- **C API**: `atlas_load`, `atlas_generate`, `atlas_free`
- **C++ CLI**: `atlas_cli.cpp`
- **Streaming SSE Server**: `atlas_server.py`
- **MLA + MoE**: DeepSeek-V2 latent attention + sparse mixture-of-experts (v2.17.0)
- **ARM64 NEON**: All 8 hot-path kernels ported (v2.16.1)
- **Binary Tokenizer**: v6 format, no transformers dependency at runtime

## Performance

Measured on Intel Core i7-7700T (Kaby Lake, DDR4-2400, ~20 GB/s).

| Model | Size | tok/s (total) | Path |
|-------|------|:-------------:|------|
| Falcon3-1B | 1.22 GB | 10.1 | f32 bypass |
| Falcon3-3B | 1.96 GB | 7.1 | hybrid |
| Falcon3-7B | 2.75 GB | 3.15 | int4 FFN |
| Falcon3-10B | 3.28 GB | 2.25 | int4 FFN |
| Bonsai-1.7B | 0.86 GB | 13.0 | f32 bypass |
| Bonsai-4B | 1.45 GB | 12.0 | f32 bypass |
| Llama3-8B-1.58 | 3.27 GB | ~4 | hybrid |
| TriLM-1.5B | 0.65 GB | ~15 | f32 bypass |

**Detail**: All 29 HF-validated models are available on the [ATLAS Hub](https://huggingface.co/xxxn3m3s1sxxx)
(Falcon3, Bonsai, BitNet, TriLM, CANN, Llama3-1.58).

## Status

Active development. MLA (Multi-head Latent Attention) and MoE (Mixture-of-Experts)
support for DeepSeek-V2 architecture added in v2.17.0.

## License

Apache 2.0. See `LICENSE`.
