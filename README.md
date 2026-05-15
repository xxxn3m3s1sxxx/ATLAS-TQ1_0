# Atlas-TQ1-BitNet: SIMD Turbo Edition

Hochoptimierte C++ Inferenz-Engine für 1.58-Bit (Ternary) Quantized Models.

## Performance (i7-13700T @ 13th Gen)

| Modell | Latenz | Speed |
| :--- | :--- | :--- |
| **Falcon3-7B** | ~282 ms/tok | 3.5 tok/s |
| **Falcon3-10B** | ~444 ms/tok | 2.2 tok/s |

Gesamt-Inferenz inkl. Python-Overhead (ask.py): **~319ms/tok**.

## Build

```bash
clang++ -O3 -march=native -fopenmp atlas_falcon3.cpp -o atlas_falcon3_avx2.exe
clang++ -O3 -march=native -fopenmp atlas_falcon3_7b.cpp -o atlas_falcon3_7b_avx2.exe
```

## Usage

```bash
python ask.py "Your prompt" --7b --temp 0.7
python ask.py "Your prompt" --10b --temp 0.3
```

## Files

- `atlas_falcon3.cpp` – 10B inference engine
- `atlas_falcon3_7b.cpp` – 7B inference engine
- `ask.py` – CLI interface
- `falcon3_tq10/` – 10B weights
- `falcon3_7b_tq10/` – 7B weights
