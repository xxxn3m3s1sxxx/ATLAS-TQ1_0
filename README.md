# ATLAS — TQ1.0 Ternary Inference Engine

**v2.16.1 — Juni 2026 — ARCHIVED**

CPU inference engine für 1.58-Bit ternär quantisierte LLMs.
Entstanden aus der Frage: *"Wie weit kommt man mit CPU-only + ternary?"*
Antwort: **~3 tok/s für 7B auf DDR4. Nicht genug für interaktive Nutzung.**

TQ1.0 ist trotzdem das einzige produktionsreife Format für ternäre 1.58-Bit-Quantisierung —
5 Trits pro Byte, Base-3 codiert. Das Format und der Packer (der 29 Architekturen
automatisch erkennt) sind das eigentliche Artefakt, nicht die Engine selbst.

## Funktionsumfang

- **TQ1.0-Packer** (`pack_to_atlas.py`): HuggingFace Safetensors → TQ1.0 (auto-detect)
- **5 Matmul-Modi**: int8 (AVX2), int4 (nibble-unpack, ~26% schneller auf 7B), f32 bypass,
  ternary (vpsignb), TQ1-packed (chunked decode + SIMD)
- **C API**: `atlas_load`, `atlas_generate`, `atlas_free`
- **C++ CLI**: `atlas_cli.cpp`
- **Streaming SSE Server**: `atlas_server.py`
- **ARM64 NEON**: Alle 8 Hot-Path-Kernels portiert (v2.16.1)
- **Binary Tokenizer**: v6-Format, kein transformers-Dependency zur Laufzeit

## Performance

Gemessen auf Intel Core i7-7700T (Kaby Lake, DDR4-2400, ~20 GB/s).

| Modell | Größe | tok/s (total) | Pfad |
|--------|-------|:-------------:|------|
| Falcon3-1B | 1.22 GB | 10.1 | f32 bypass |
| Falcon3-3B | 1.96 GB | 7.1 | hybrid |
| Falcon3-7B | 2.75 GB | 3.15 | int4 FFN |
| Falcon3-10B | 3.28 GB | 2.25 | int4 FFN |
| Bonsai-1.7B | 0.86 GB | 13.0 | f32 bypass |
| Bonsai-4B | 1.45 GB | 12.0 | f32 bypass |
| Llama3-8B-1.58 | 3.27 GB | ~4 | hybrid |
| TriLM-1.5B | 0.65 GB | ~15 | f32 bypass |

**Detail**: Die 29 HF-validierten Modelle sind im [ATLAS Hub](https://huggingface.co/xxxn3m3s1sxxx)
verfügbar (Falcon3, Bonsai, BitNet, TriLM, CANN, Llama3-1.58).

## Warum archiviert?

Drei Gründe:

1. **Physik**: DDR4-Bandbreite (~20 GB/s) limitiert jede CPU-Inference.
   TQ1.0 halbiert die Gewichtsgröße gegenüber GGUF Q4, aber die verbleibenden
   20 GB/s erlauben nur ~3 tok/s für ein 7B — unabhängig vom Kernel.

2. **Proprietäres Format**: GGUF ist der De-facto-Standard. TQ1.0 ist kompakter,
   aber inkompatibel. Der Packer ist das Werkzeug, das fehlt — nicht die Engine.

3. **Kein Ökosystem**: Ein Autor, null User. Ein Projekt, das keiner ausser dir
   bedienen kann, ist ein Tagebuch — kein Produkt.

## Lizenz

MIT. Siehe `LICENSE`.
