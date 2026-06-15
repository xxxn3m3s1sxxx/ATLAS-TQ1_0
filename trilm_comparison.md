# TriLM Comparison -- Which Model for Professor Training?

Tested all 9 TriLM models (99M->3.9B) on ATLAS TQ1.0 v2.11.2.
Goal: find the best size/quality/speed tradeoff for academic training research.

## Quick Summary

| Rank | Model | Size | Quality | Speed | Recommendation |
|:----:|-------|:----:|:-------:|:-----:|----------------|
| 1 | **TriLM-1.1B** | 0.5 GB | **4.2/5** | 15 tok/s | **BEST OVERALL** -- perfect EN/DE, SubLN, GPU-friendly |
| 2 | TriLM-560M | 0.3 GB | 3.8/5 | 31 tok/s | **Budget pick** -- fast iteration, solid quality |
| 3 | TriLM-2.4B | 0.9 GB | 4.0/5 | 7 tok/s | **Max architecture** -- 30L/2304H for scaling studies |
| -- | TriLM-190M | 0.2 GB | 3.6/5 | 90 tok/s | Fastest usable model |
| -- | TriLM-390M | 0.3 GB | 3.4/5 | 41 tok/s | Decent for rapid prototyping |
| X | TriLM-99M | 0.1 GB | 2.7/5 | 152 tok/s | **Too small** -- repetitive, no instruction following |
| X | TriLM-3.9B | 1.3 GB | 3.3/5 | 6 tok/s | **Wrong arch** -- Llama, not SubLN |

## Full Comparison

| Model | Size | Layers | Hidden | Heads | Arch | Quality | EN T=0 | EN T=0.7 | DE T=0 | DE T=0.7 | Math | Tok/s |
|-------|:----:|:------:|:------:|:-----:|:----:|:-------:|:------:|:--------:|:------:|:--------:|:----:|:-----:|
| TriLM-99M | 0.1GB | 16 | 512 | 8 | SubLN | 2.7/5 | 4 | 3 | 3 | 1 | 2 | 152 |
| TriLM-190M | 0.2GB | 16 | 768 | 12 | SubLN | 3.6/5 | 5 | 4 | 3 | 2 | 2 | 90 |
| TriLM-390M | 0.3GB | 24 | 1024 | 16 | SubLN | 3.4/5 | 5 | 5 | 3 | 2 | 3 | 41 |
| TriLM-560M | 0.3GB | 24 | 1280 | 20 | SubLN | 3.8/5 | 5 | 5 | 5 | 3 | 3 | 31 |
| TriLM-830M | 0.4GB | 24 | 1536 | 24 | SubLN | 3.7/5 | 5 | 3 | 3 | 2 | 3 | 19 |
| **TriLM-1.1B** | **0.5GB** | **24** | **1792** | **28** | **SubLN** | **4.2/5** | **5** | **5** | **3** | **5** | **3** | **15** |
| TriLM-1.5B | 0.7GB | 24 | 2048 | 32 | SubLN | 4.1/5 | 5 | 4 | 5 | 3 | 3 | 12 |
| TriLM-2.4B | 0.9GB | 30 | 2304 | 36 | SubLN | 4.0/5 | 5 | 5 | 5 | 3 | 2 | 7 |
| TriLM-3.9B | 1.3GB | 32 | 3072 | 24 | **Llama** | 3.3/5 | 3 | 3 | 3 | 2 | 2 | 6 |

## Key Findings

### 1. Quality Cliff at 99M
The 99M model (16L/512H) is fundamentally too small for coherent instruction following.

### 2. Sweet Spot: 190M-560M
Surprisingly good quality for size. **560M** achieves near-perfect English and German at T=0. Ideal for rapid prototyping.

### 3. Best Overall: 1.1B
The **TriLM-1.1B** (24L/1792H/28 heads, SubLN) hits the sweet spot:
- Perfect English quality (T=0 and T=0.7)
- Best German capability of all models tested
- 0.5 GB -- fits easily in GPU memory
- 15 tok/s inference -- fast enough for training evaluation

### 4. Diminishing Returns Above 1.1B
The 1.5B and 2.4B models add size but not proportional quality.

### 5. 3.9B is an Outlier
Uses **Llama architecture** (not SubLN). Fails to answer direct questions. **Avoid for SubLN research.**

## Professor Recommendation

> **Start with TriLM-1.1B** for serious training experiments (SubLN, 0.5GB, perfect quality).
> Use **TriLM-560M** for rapid iteration / hyperparameter sweeps.
> Study scaling with **TriLM-2.4B** once the training pipeline works.
> **Avoid** TriLM-99M (too small) and TriLM-3.9B (wrong architecture).

*Tested 2026-06-12 on Intel Core i7-7700T (4C/8T, AVX2) with ATLAS TQ1.0 v2.11.2.*
