#!/usr/bin/env python3
"""parse_profile.py — Parse ARM64 FFN micro-profiling output from stderr.

Usage:
  python scripts/parse_profile.py profiling_stderr.txt
  python scripts/parse_profile.py --summary profiling_stderr.txt

Input format (from atlas_kernel_arm64.cpp profile_print_arm64()):
  ── ARM64 FFN micro (576M cycles) ──
    i4 unpack       4731901 (  0.8%)
    i4 FMA        481209204 ( 83.5%)
    f32 conv        5760919 (  1.0%)
    f32 FMA        28804594 (  5.0%)
    default conv   11521837 (  2.0%)
    default FMA    44051945 (  7.6%)
  ────────────────────────────────

Output: Markdown table + optional CSV
"""

import re
import sys
from pathlib import Path


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


def parse_stderr(text: str):
    """Parse all profile blocks from stderr text.

    Returns list of dicts with keys: total_cycles, categories (ordered dict)
    """
    blocks = []
    for m in PROFILE_RE.finditer(text):
        total_cycles = int(m.group(1)) * 1_000_000
        body = m.group(2)
        cats = {}
        for lm in LINE_RE.finditer(body):
            cat = lm.group(1)
            cycles = int(lm.group(2))
            pct = float(lm.group(3))
            cats[cat] = {"cycles": cycles, "pct": pct}
        blocks.append({"total_cycles": total_cycles, "categories": cats})
    return blocks


def print_summary_table(blocks, file=sys.stdout):
    """Print a Markdown summary table."""
    if not blocks:
        print("No profiling blocks found.", file=file)
        return

    print("# ARM64 FFN Micro-Profiling Summary", file=file)
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
                print(
                    f"| {label} | {c['cycles']:>10,} | {c['pct']:>5.1f}% |",
                    file=file,
                )

        # Computed rollups
        i4_total = sum(
            cats[k]["cycles"]
            for k in ["i4 unpack", "i4 FMA"]
            if k in cats
        )
        f32_total = sum(
            cats[k]["cycles"]
            for k in ["f32 conv", "f32 FMA"]
            if k in cats
        )
        default_total = sum(
            cats[k]["cycles"]
            for k in ["default conv", "default FMA"]
            if k in cats
        )

        gate_up_total = f32_total + default_total

        print(file=file)
        print("| **Rollup** | **Cycles** | **%** |", file=file)
        print("|------------|----------:|-----:|", file=file)
        print(
            f"| Int4 FFN    | {i4_total:>10,} | "
            f"{i4_total / blk['total_cycles'] * 100:>5.1f}% |",
            file=file,
        )
        print(
            f"| Gate+Up F32 | {f32_total:>10,} | "
            f"{f32_total / blk['total_cycles'] * 100:>5.1f}% |",
            file=file,
        )
        print(
            f"| Gate+Up Def | {default_total:>10,} | "
            f"{default_total / blk['total_cycles'] * 100:>5.1f}% |",
            file=file,
        )
        print(
            f"| Gate+Up All | {gate_up_total:>10,} | "
            f"{gate_up_total / blk['total_cycles'] * 100:>5.1f}% |",
            file=file,
        )
        print(file=file)


def print_csv(blocks, file=sys.stdout):
    """Print CSV for external analysis."""
    if not blocks:
        return
    print("block,total_cycles_m," + ",".join(CATEGORY_ORDER), file=file)
    for i, blk in enumerate(blocks):
        cats = blk["categories"]
        row = [str(i + 1), str(blk["total_cycles"] // 1_000_000)]
        for cat in CATEGORY_ORDER:
            row.append(str(cats.get(cat, {}).get("cycles", 0)))
        print(",".join(row), file=file)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    blocks = parse_stderr(text)

    if "--csv" in sys.argv:
        print_csv(blocks)
    else:
        print_summary_table(blocks)

    if not blocks:
        sys.exit(1)


if __name__ == "__main__":
    main()
