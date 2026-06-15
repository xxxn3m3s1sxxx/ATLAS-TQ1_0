# ATLAS 6-Punkte-Upgrade-Plan

Aus sechs Inferenz-Engines destillierte Optimierungen für TQ1.0.

---

## 1. SIMD Bit-Gating *(oxibonsai)*

**Ziel:** Keine bedingten Sprünge (`if/else`) im heißen Inferenzpfad mehr.

**Umsetzung:**
- TQ1.0-Trits (‑1,0,+1) in 2-Bit-Layout codieren: `00`=‑1, `01`=0, `10`=+1
- Null-Zustände per `_mm256_and_si256` logisch maskieren — keine Branch-Prediction-Fehler
- `_mm256_sign_epi8` + Maske ersetzen durch pure `AND`-Operation

**Aufwand:** ~6h (AVX2-Kernel-Rewrite)
**Impact:** Mittel — spart Branch-Miss-Penalty, aber Ternary-Add ist schon schnell

---

## 2. Zero-Copy SSD-to-L3 Streaming *(R3-Engine)*

**Ziel:** RAM-Footprint gegen Null fahren für 4B+ Modelle.

**Umsetzung:**
- `mmap` für `.i8`-Cache-Dateien (Windows: `CreateFileMapping` + `MapViewOfFile`)
- Gewichte direkt von NVMe → L3-Cache via OS-Paging
- `MADV_WILLNEED` (Linux) / `PrefetchVirtualMemory` (Win8+) für Prefetch
- Aktuelles `fread`-Laden ersetzen — lazy Page-Fault-basiertes Streaming

**Aufwand:** ~3h (C++ Ladepfad)
**Impact:** Hoch — 8B-Modell von 6.9GB→~0GB RAM bei 2.2 tok/s

---

## 3. Sparse Segment Reduction / Nuller-Skips *(ternaryLLM)*

**Ziel:** ~60% Null-Gewichte nicht berechnen.

**Umsetzung:**
- TQ1.0-Format um 1-Bit "Null-Block"-Flag pro 32-Trit-Segment erweitern
- Dekompressions-Kernel prüft Flag: wenn Null → ganze Segment-Add überspringen
- Packer (`pack_to_atlas.py`): Null-Block-Detection + Flag-Setzung
- Ca. 2 Bytes Overhead pro 32 Trits → <1% Format-Overhead

**Aufwand:** ~4h (TQ1-Format-Erweiterung + Kernel)
**Impact:** Sehr hoch — 60% weniger Bandbreite bei 99M (Null-Rate am höchsten)

---

## 4. Cache-Line Boundary Alignment *(NexaQuant v1)*

**Ziel:** Keine unalignten 64-Byte-Lesezugriffe mehr.

**Umsetzung:**
- `pack_to_atlas.py`: Jedes TQ1-gepackte Tensor-Row auf 64-Byte-Grenze ausrichten
- Padding-Felder in der Directory-Struktur (Byte 6-7 pro Dir-Entry)
- Kernel: `_mm256_load_si256` (aligned) statt `_mm256_loadu_si256` (unaligned)

**Aufwand:** ~2h (Packer)
**Impact:** Mittel — 5-15% Bandbreitengewinn, aber einfach und risikofrei

---

## 5. Fast Walsh-Hadamard-Transformation *(1.58BitNet)*

**Ziel:** Quantisierungsfehler gleichmäßig über Aktivierungsraum verteilen — stabilisiert kleine Modelle (99M-Loops).

**Umsetzung:**
- Schlanker `HBitLinear`-Preprocessor als optionaler Schritt in `forward_layer_internal`
- FWHT ist multiplications-free (nur ±1-Operationen) — perfekt für Ternary
- Im Packer: Option `--fwht` schaltet es pro Tensor zu
- Als **`set_use_fwht()`** C-API exponiert (default: off)

**Aufwand:** ~4h (C++ FWHT-Kernel + Python-API)
**Impact:** Hoch — direktes Gegenmittel für 99M-Repetition, kein Precision-Verlust

---

## 6. BitMamba-2 SSM-Architektur-Support *(BitMamba-2)*

**Ziel:** Kein KV-Cache — unendlicher Kontext bei konstanter Inferenzgeschwindigkeit.

**Umsetzung:**
- SSM-Kernel (State-Space-Modell) in `atlas_api.cpp`: selektive Scan-Operation
- `pack_to_atlas.py` um Mamba-2-Tensor-Mapping erweitern (A, B, C, D, dt)
- Kein KV-Cache nötig → `ensure_cache()` kann entfallen
- Pipeline: biteniertes Mamba-2-Modell packen + inferieren

**Aufwand:** ~2 Wochen (SSM-Kernel von Grund auf)
**Impact:** Revolutionär — 100K+ Kontext auf Edge-Hardware

---

## Priorisierung (Impact/Aufwand)

| # | Punkt | Aufwand | Impact | Prio |
|---|-------|---------|--------|------|
| 1 | SIMD Bit-Gating | ~6h | Mittel | ⭐⭐ |
| 2 | SSD-to-L3 mmap | ~3h | Hoch | ⭐⭐⭐ |
| 3 | Nuller-Skips | ~4h | Sehr hoch | ⭐⭐⭐⭐ |
| 4 | Cache-Line Align | ~2h | Mittel | ⭐⭐ |
| 5 | FWHT | ~4h | Hoch | ⭐⭐⭐ |
| 6 | BitMamba-2 | ~2 Wo | Revolutionär | ⭐ (langfristig) |

**Nächste Schritte:** 3 → 5 → 2 → 4 → 1 (→ 6 nach TriLM-Serie)
