#!/usr/bin/env python3
"""Check coverage meets minimum thresholds. Exit 1 if below."""
import json
import sys

MIN_LINES = 65.0
MIN_FUNCS = 80.0

report = sys.argv[1] if len(sys.argv) > 1 else "coverage_summary.json"
with open(report) as f:
    d = json.load(f)
lines = d["line_percent"]
funcs = d["function_percent"]
branches = d["branch_percent"]

print(f"Lines: {lines:.1f}%  Functions: {funcs:.1f}%  Branches: {branches:.1f}%")
ok = lines >= MIN_LINES and funcs >= MIN_FUNCS
if not ok:
    print(f"FAIL: below thresholds (lines >= {MIN_LINES}%, functions >= {MIN_FUNCS}%)")
    sys.exit(1)
print("PASS")
