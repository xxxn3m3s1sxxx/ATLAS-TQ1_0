# Known Issues & Limitations

## Hybrid CPU (Intel Alder Lake+) Thread Oversubscription
- **Symptoms**: Lower tok/s than expected on Intel 12th gen+ CPUs. P-cores idle while waiting for E-cores at OpenMP barriers.
- **Workaround**: Set `--threads` to your physical P-core count (not logical threads). Example: i7-12700H (6P+8E) → `--threads 6`.
- **Status**: Architecture limit — no portable P/E-core detection API. Linux `lscpu -e` shows core types, but no runtime OS API.

## Gumbel-max Early EOS (10B/7B Models)
- **Symptoms**: Generation stops prematurely at EOS token, especially with T=0.7 sampling.
- **Cause**: Gumbel noise occasionally boosts EOS logits above natural continuation tokens. Not an engine bug — expected Gumbel-max sampling behavior.
- **Mitigation**: Use `--min-new <N>` to suppress EOS for first N tokens. Use T=0 (deterministic) for 7B+ models for clean argmax output.

## Windows ANSI-Code Page Fallback
- **Symptoms**: On Windows 10 builds before 1903, `SetConsoleCP(CP_UTF8)` may fail silently. Console input in interactive mode falls back to the system ANSI codepage.
- **Workaround**: Use PowerShell 7+ or Windows Terminal. Pipe input via UTF-8 files: `type prompt.txt | atlas.exe model.atlas`.
- **Status**: The CLI argument parser (`CommandLineToArgvW` + `WideCharToMultiByte`) correctly handles all non-ASCII *arguments* on all Windows versions. Only interactive *stdin* input is affected on legacy consoles.

## Pre-Haswell CPUs (Before 2013)
- **Symptoms**: `[ATLAS] Error: AVX2 instruction set required.` on startup.
- **Cause**: ATLAS requires AVX2 (Haswell, ~2013+) for its int8 matmul kernels.
- **Status**: By design. No fallback path for SSE4.1 or AVX1.

## Model Format v5/v6 Compatibility
- **v5 models** (Falcon3 only, pre-v2.0): Load without embedded tokenizer. The CLI falls back to raw token IDs; Python binding uses HuggingFace `AutoTokenizer`.
- **v6+ models** (Falcon3/BitNet/Bonsai): Fully self-contained with binary tokenizer.
- All engines are backward-compatible with v5. New models should be packed as v6+.

## 1B / 3B Models at T=0
- **Symptoms**: Newline collapse or repetition at deterministic sampling.
- **Cause**: Model architecture limit — 1B (18 layers) and 3B (22 layers) lack the depth for stable argmax paths.
- **Recommendation**: Always use `T=0.7, top_k=40` for models below 7B.
