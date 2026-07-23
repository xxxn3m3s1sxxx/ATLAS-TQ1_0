# Roadmap — v2.17+ MLA/MoE

## v2.17.0 — DeepSeek-V2 MLA + MoE (current)

- [x] MLA (Multi-head Latent Attention) with compressed KV cache
- [x] MoE (Mixture-of-Experts) sparse FFN dispatch
- [x] Shared + routed expert forward
- [x] Row-major matmul wrappers for MLA projections
- [x] ARM64 guards for AVX2 intrinsics
- [x] Row-major matmul for router (ttype==3), int4 decode (ttype==8)
- [x] Python inference: `atlas_repack_experts` ctypes binding

## v2.17.x — Bug Fixes & Validation

- [ ] End-to-end test with real DeepSeek-V2-Lite weights (~34 GB)
- [ ] Phase 3 Medium Fixes: M1-M6 (alloca guards, int4 remainder, decompress heap, matmul reorder)
- [ ] Coverage threshold restoration (50% → 60%+) after real model testing
- [ ] MLA attention correctness verification (golden logits vs HF reference)

## v2.18 — Performance

- [ ] Batched MoE dispatch (currently per-expert sequential)
- [ ] MLA attention kernel optimization (score precompute, softmax vectorization)
- [ ] INT4 remainder row handling for compute-bound MoE experts

## v2.20 — Broader Architecture Support

- [ ] DeepSeek-V2-Large / DeepSeek-V3 MLA
- [ ] Mixtral-style MoE (no MLA, standard MHA + sparse FFN)
- [ ] Qwen3 MoE support
