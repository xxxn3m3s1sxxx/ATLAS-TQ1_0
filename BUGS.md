# Known Issues & Limitations

## ✅ Recently Fixed

### v2.10.4 — OOB Logits Read + Debug Print Spam

### Bug #10: Unconditional `logits[96944]` OOB Read in Debug Print (Critical)
- **Fixed**: `atlas_api.cpp` Debug-Print in `atlas_generate` (Prefill top-5) las `logits[V-1]` mit V=73448. Der Print `logits[96944]` lag 586 Bytes über der Allokation → sporadischer Crash bei `0x...EAE0` je nach Heap-Layout.
- **Fix**: Alle unconditional Debug-Prints entfernt (attn_raw/attn_sft, DECLM/PRELM, LMHEAD, Prefill top-5, Decode top-5, before/after barriers).

### Bug #11: `[ACTDBG] max_val=` >1000 Lines per Forward
- **Fixed**: `matmul_tq2_f32` produziert >1000 Zeilen `[ACTDBG] max_val=` pro Forward.
- **Fix**: Entfernt.

### Bug #12: `ATLAS_DLL` Env Var Does Not Override
- **Fixed**: `atlas_infer.py` prüfte `ATLAS_DLL`-Umgebungsvariable nur wenn `atlas.dll` nicht existierte.
- **Fix**: `ATLAS_DLL` überschreibt immer.

### v2.10.3 — Bug Hunt Round 2 (9 bugs)

### Bug #1: Falcon3 BPE Vocab Cutoff (Critical)
- **Fixed**: `pack_to_atlas.py:334` hatte hardcodiertes `v < 128000` als BPE-Vocab-Filter — köpft Falcon3 (V=131072 → 3072 BPE-Tokens verloren).
- **Fix**: `raw_vocab` aus JSON `model.vocab` direkt genutzt. Fallback `tok.get_vocab()` bleibt als Safety-Net.
- **Note**: Wurde nie committed; alle HF-Modelle mit `tok.get_vocab()` gepackt → unbeschädigt.

### Bug #2: Fehlende Llama3-Stops in `generate_c()`
- **Fixed**: `atlas_infer.py:980` fehlten `<|eot_id|>`, `<|start_header_id|>`, `<|end_header_id|>` als Stopp-Token.
- **Fix**: Diese Tokens unter `_is_llama3` Guard hinzugefügt.

### Bug #3: Fehlende Llama3-Special-Suppression in `_cpp_decode()`
- **Fixed**: `atlas_infer.py:1126` fehlten dieselben Tokens in der Decode-Skip-Liste.
- **Fix**: `<|eot_id|>`, `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>` hinzugefügt — nur bei `_is_llama3`.

### Bug #4: Unaligned Pointer Access
- **Fixed**: `atlas_api.cpp:859` hatte `*(const uint32_t*)ap` auf potentiell unaligned Addressen → UB auf ARM/RISC.
- **Fix**: `memcpy(&val, ap, 4)` — portabel, kein Performance-Verlust auf x86.

### Bug #5: `rope_interleaved` Heuristik überschreibt Config
- **Fixed**: `atlas_api.cpp:4874` in `ensure_layer_idx()` setzte `rope_interleaved` unabhängig von config.json.
- **Fix**: `rope_interleaved_set` Flag — Config-JSON setzt es; Heuristik feuert nur wenn Config keinen Wert hatte.

### Bug #6: `xoshiro_state` global → thread_local
- **Identified**: `atlas_api.cpp:133` — `static uint64_t xoshiro_state[4]` global, nicht `thread_local`.
- **Fix**: `static thread_local` — jeder Thread bekommt eigenen Xoshiro-Zustand. Kein Data Race mehr.
- **Note**: Bug #6 in v2.10.3 gefixt (vorher für v2.10.4 geplant).

### Bug #7: `g_has_avx512_vnni` Init-Race
- **Identified**: `atlas_api.cpp:3105` — `static int g_has_avx512_vnni = -1` ohne Thread-Safe-Init. Zwei Threads sehen gleichzeitig `-1` und schreiben beide denselben Wert.
- **Fix**: C++11 function-local `static const int`, initialisiert beim ersten Funktionsaufruf (garantiert Thread-safe via Standard).

### Bug #8: `_eos_id = 0` Sentinel konfligiert mit EOS-Token-ID 0
- **Identified**: `atlas_infer.py:400` — Sentinalwert `0` ist nicht von echtem EOS-Token `0` unterscheidbar.
- **Fix**: Sentinal auf `None` geändert. Fallback-Check `if self._eos_id is None` statt `if self._eos_id == 0`. `token_to_id()` nutzt `if tid is not None`.

### Bug #9: Fehlende BitNet-Stops in `generate_c()` und `_cpp_decode()`
- **Identified**: `atlas_infer.py:980` — `generate_c()` fügt Llama3-Stops hinzu, aber nicht BitNet (`<|eot_id|>`). Gleiches Problem in `_cpp_decode()` (Zeile 1128) und `generate()` (Zeile 888).
- **Fix**: `_is_bitnet`-Guards in allen drei Funktionen.

---

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
