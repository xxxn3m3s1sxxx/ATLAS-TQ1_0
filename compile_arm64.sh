#!/bin/bash
# Build atlas.dylib for ARM64 macOS (Apple Silicon)
# Requires: Xcode CLT (clang with arm64 target)
#
# Usage:
#   ./compile_arm64.sh              Release build
#   ./compile_arm64.sh debug        Debug build (-O0 -g)
#
# NEON SDOT (vdotq_s32) requires ARMv8.2-A+, available on all Apple Silicon.

set -euo pipefail
CC=clang

if [ "${1:-}" = "debug" ]; then
    OPT="-O0 -g -DATLAS_DEBUG_MODE"
    echo "[Atlas] Compiling atlas.dylib (ARM64 debug)..."
else
    OPT="-O2 -ffast-math"
    echo "[Atlas] Compiling atlas.dylib (ARM64 release)..."
fi

$CC -shared -o atlas.dylib \
    atlas_api.cpp atlas_kernel_arm64.cpp \
    $OPT -march=armv8.2-a+dotprod -std=c++17 -fopenmp

echo "[Atlas] OK -- atlas.dylib built"
