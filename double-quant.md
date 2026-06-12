# Double-Quantization: Warum tritplane3 + TQ1.0 kollabiert

## Problem

Das TritPlane3-Format (`AsadIsmail/Qwen3-1.7B-ternary`) speichert Gewichte als
2-bit ternäre Werte pro Plane, mit per-group fp16 alpha+mu (group_size=32).
Bei 2 Ebenen ergibt das ~6 effective bits, avg_relative_error ~23%.

Die fp32-Rekonstruktion ist: `w_hat = Σ(alpha_i * t_i + mu_i)`.

## Warum per-row int8 funktioniert

Per-row int8 (ttype=11) quantisiert die fp32-Rekonstruktion mit 256 Stufen:

`q = clip(round(w * scale), -127, 127)`, `scale = 127 / max(|w_row|)`

Fehler pro Element: ~0.4% (1/256). Der tritplane3-Vorfehler von 23% wird
dominant — int8 fügt nur ~0.4% Rauschen hinzu. Signal bleibt erhalten.

## Warum TQ1.0 (ternary) scheitert

TQ1.0 (ttype=5) ternarisiert auf 3 Stufen {-1, 0, +1}:

`t = clip(round(w / block_scale), -1, 1)`, `block_scale = max(|w_block|)`

Fehler pro Element: ~33% (1/3). Der tritplane3-Vorfehler von 23% ist bereits
da, TQ1.0 legt 33% drauf. Gesamtfehler: `1 - (1-0.23)*(1-0.33) ≈ 48%`.

Das Signal überlebt zwei Quantisierungen nicht.

## Tabelle

| Pfad | Stufen | Zusatzfehler | Kohärenz |
|------|--------:|:------------:|:--------:|
| tritplane3 → per-row int8 | 256 | ~0.4% | ✅ |
| tritplane3 → TQ1.0 | 3 | ~33% | ❌ garbage |
| Native TQ1.0 (Falcon3) | 3 | — | ✅ (nativ trainiert) |

## Lesson

TQ1.0 ist für **nativ ternär kalibrierte** Modelle (Falcon3, BitNet).
Tritplane3 → TQ1.0 = double quantization = Signal-Collapse.
Per-row int8 ist die einzig stabile Brücke von tritplane3 zu ATLAS.
