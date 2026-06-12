# Roadmap → v2.10 Community Release

## Problem

ATLAS läuft technisch, aber eine `.atlas`-Datei zu bekommen erfordert Python, `transformers`, und stundenlange Downloads. Das killt Adoption.

## Blocker (vor v2.10)

### 1. Release-Infrastruktur

| Was | Warum | Dringlichkeit |
|-----|-------|:-------------:|
| **HF Collection** mit vorgefertigten `.atlas`-Dateien | Quickstart muss `curl -O` funktionieren | 🔴 |
| **README Quickstart** auf `.atlas`-URLs zeigen | Aktuell zeigen Links auf rohe HF-Safetensors | 🔴 |
| **Packer-Anleitung** in "Advanced: Custom Models" verschieben | Erstes Erlebnis: Download + ausführen, nicht bauen | 🔴 |
| **CI-Release-Artifacts** (`dist/` mit DLL + CLI) | `release.yml` muss Binaries bereitstellen | 🟡 |
| **Dockerfile** minimal | Für User ohne Build-Toolchain | 🟡 |

**HF Collection Plan:**
- `atlas-community/Falcon3-1B-Instruct-tq1` (~1.2 GB)
- `atlas-community/Falcon3-3B-Instruct-tq1` (~2.0 GB)
- `atlas-community/Ternary-Bonsai-4B-tq1` (~1.5 GB)
- `atlas-community/Ternary-Bonsai-1.7B-tq1` (~0.9 GB)

### 2. TQ2 Performance

| Was | Status | Ziel |
|-----|--------|:----:|
| Bonsai-8B TQ2 (0.58→1.11 tok/s) | Batch-Stores + 2× Unroll done | **≥1.5** tok/s |
| VNNI-Kernel (`c2904a5`) | Ungetestet — kein AVX-512 HW | CI-Validierung |
| TQ2 als Default-Format | Verfrüht — f32-Bypass 2× schneller | Performance-Parität |

### 3. CI/Build

| Was | Status |
|-----|--------|
| **Clang 19+ Matrix** (ubuntu-24.04) | 🔴 Fehlt — `target("avx10.2")` ungetestet |
| **Clang 18 Matrix** (ubuntu-latest) | 🔴 VNNI-Stub muss auf CI grün werden |
| **Windows Coverage** (64.5% → 65%) | 🟡 0.5% Gap |

## Meilensteine

```
M1: HF Collection + README-Refactoring  →  "Download & Run"
M2: CI Matrix grün (Clang 18 + 19)      →  "Build-Sicherheit"
M3: TQ2 Performance ≥1.5 tok/s          →  "Default-würdig"
M4: v2.10 Tag + Release                 →  "Community Launch"
```
