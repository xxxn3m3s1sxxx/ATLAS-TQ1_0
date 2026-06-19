#!/usr/bin/env python3
"""parse_profile.py — Parse ARM64 FFN micro-profiling output from stderr.

Usage:
  python scripts/parse_profile.py profiling_stderr.txt
  python scripts/parse_profile.py --csv profiling_stderr.txt

Input formats:

  1. FFN Micro (accumulator blocks from profile_print_arm64()):
     ── ARM64 FFN micro (576M cycles) ──
       i4 unpack       4731901 (  0.8%)
       i4 FMA        481209204 ( 83.5%)
       ...
     ────────────────────────────────

  2. RAII scopes (ATLAS_PROFILE macro, quick & dirty):
     [Profile] KV_Shift: 12345 ticks
     [Profile] Tokenizer: 678 ticks

Output: Markdown tables for both formats.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict


# ─── Format 1: FFN Micro accumulator ───────────────────────────────────

PROFILE_RE = re.compile(
    r"─{2} ARM64 FFN micro \((\d+)M cycles\) ─{2}\s*"
    r"(.*?)"
    r"─{20,}",
    re.DOTALL,
)

LINE_RE = re.compile(
    r"\s*(\S[\S ]*?\S)\s+(\d+)\s+\((\s*[\d.]+)%\)"
)

CATEGORY_ORDER = [
    "i4 unpack",
    "i4 FMA",
    "f32 conv",
    "f32 FMA",
    "default conv",
    "default FMA",
]

CATEGORY_LABELS = {
    "i4 unpack": "Int4 Nibble Unpack",
    "i4 FMA": "Int4 vdotq_s32 FMA",
    "f32 conv": "F32 int8→f32 Convert",
    "f32 FMA": "F32 vfmaq_f32 FMA",
    "default conv": "Default XOR-0x80 Convert",
    "default FMA": "Default vdotq_s32 FMA",
}

# ─── Format 2: RAII ATLAS_PROFILE scopes ─────────────────────────────

RAII_RE = re.compile(r"\[Profile\] (.+?): (\d+) ticks?")


def parse_ffn_blocks(text: str):
    blocks = []
    for m in PROFILE_RE.finditer(text):
        total_cycles = int(m.group(1)) * 1_000_000
        cats = {}
        for lm in LINE_RE.finditer(m.group(2)):
            cats[lm.group(1)] = {"cycles": int(lm.group(2)), "pct": float(lm.group(3))}
        blocks.append({"total_cycles": total_cycles, "categories": cats})
    return blocks


def print_ffn_table(blocks, file=sys.stdout):
    if not blocks:
        return
    print("# ARM64 FFN Micro-Profiling", file=file)
    print(file=file)
    for i, blk in enumerate(blocks):
        total_m = blk["total_cycles"] // 1_000_000
        cats = blk["categories"]
        print(f"## Block {i + 1}: {total_m}M total cycles", file=file)
        print(file=file)
        print("| Category | Cycles | % |", file=file)
        print("|----------|-------:|--:|", file=file)
        for cat in CATEGORY_ORDER:
            if cat in cats:
                c = cats[cat]
                label = CATEGORY_LABELS.get(cat, cat)
                print(f"| {label} | {c['cycles']:>10,} | {c['pct']:>5.1f}% |", file=file)

        i4_total = sum(cats[k]["cycles"] for k in ["i4 unpack", "i4 FMA"] if k in cats)
        f32_total = sum(cats[k]["cycles"] for k in ["f32 conv", "f32 FMA"] if k in cats)
        def_total = sum(cats[k]["cycles"] for k in ["default conv", "default FMA"] if k in cats)
        gu_total = f32_total + def_total
        total = blk["total_cycles"]

        print(file=file)
        print("| **Rollup** | **Cycles** | **%** |", file=file)
        print("|------------|----------:|-----:|", file=file)
        print(f"| Int4 FFN    | {i4_total:>10,} | {i4_total / total * 100:>5.1f}% |", file=file)
        print(f"| Gate+Up F32 | {f32_total:>10,} | {f32_total / total * 100:>5.1f}% |", file=file)
        print(f"| Gate+Up Def | {def_total:>10,} | {def_total / total * 100:>5.1f}% |", file=file)
        print(f"| Gate+Up All | {gu_total:>10,} | {gu_total / total * 100:>5.1f}% |", file=file)
        print(file=file)


def print_ffn_csv(blocks, file=sys.stdout):
    if not blocks:
        return
    print("block,total_cycles_m," + ",".join(CATEGORY_ORDER), file=file)
    for i, blk in enumerate(blocks):
        cats = blk["categories"]
        row = [str(i + 1), str(blk["total_cycles"] // 1_000_000)]
        for cat in CATEGORY_ORDER:
            row.append(str(cats.get(cat, {}).get("cycles", 0)))
        print(",".join(row), file=file)


# ─── Format 2: RAII scopes ─────────────────────────────────────────────

def parse_raii(text: str):
    data = defaultdict(list)
    for m in RAII_RE.finditer(text):
        name = m.group(1).strip()
        ticks = int(m.group(2))
        data[name].append(ticks)
    return dict(data)


def print_raii_table(data: dict, file=sys.stdout):
    if not data:
        return
    print("# RAII ATLAS_PROFILE Scopes", file=file)
    print(file=file)
    print("| Scope | Calls | Avg Ticks | Min Ticks | Max Ticks |", file=file)
    print("| :--- | ----: | --------: | --------: | --------: |", file=file)

    for name, ticks in sorted(data.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True):
        avg = sum(ticks) / len(ticks)
        print(f"| {name} | {len(ticks)} | {int(avg):>10,} | {min(ticks):>9,} | {max(ticks):>9,} |", file=file)
    print(file=file)


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    ffn_blocks = parse_ffn_blocks(text)
    raii_data = parse_raii(text)

    if "--csv" in sys.argv:
        print_ffn_csv(ffn_blocks)
        return

    if ffn_blocks:
        print_ffn_table(ffn_blocks)
    if raii_data:
        print_raii_table(raii_data)
    if not ffn_blocks and not raii_data:
        print("No profiling data found. Was the binary built with -DPROFILE_MODE?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
