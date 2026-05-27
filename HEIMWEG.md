# Heute Abend: v7 Packer + BitNet

## Ready at home
- v7 Packer (integrierter Tokenizer, korrektes 2-Bit-Packing für BitNet)
- BitNet-2B4T model → `.atlas` bauen, testen

## Todo (daheim)
1. v7 Packer pushen ins Repo
2. BitNet-2B4T packen + `.atlas` validieren (negative token ID -151891 fix)
3. `lm_head` quantisiert cachen (384 MB, spart 2–9s Ladezeit)
4. Thread-Tuning: `OMP_NUM_THREADS=4` vs 8 auf Kaby Lake testen
5. TurboQuant Kernel-Gerüst in `atlas_api.cpp`: 2-Bit Matmul als leere Hülle vorbereiten

## Büro-Status (v2.6.5)
- origin/master up-to-date
- 10B `.i8` Cache gebaut (9515 MB, C: 56.8 GB frei)
- Alle 8 Modelle lauffähig
