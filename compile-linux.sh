#!/bin/bash
# Compile libatlas.so for Linux x86-64 (AVX2 + FMA)
# Requires: g++ with OpenMP support
# Usage: ./compile-linux.sh [CXX] [CXXFLAGS]
set -eu
SRC="atlas_api.cpp"
OUT="libatlas.so"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O2 -mavx2 -mfma -mf16c -ffast-math -fopenmp -fPIC -std=c++17 -fvisibility=hidden}"
echo "[Atlas] Compiling libatlas.so..."
$CXX -shared $CXXFLAGS -o "$OUT" "$SRC"
ls -lh "$OUT"
echo "[Atlas] OK -- $OUT built successfully"

# Build CLI (optional — requires atlas_cli.cpp)
if [ -f "atlas_cli.cpp" ]; then
    CLI_OUT="atlas"
    echo "[Atlas] Compiling $CLI_OUT (standalone CLI)..."
    $CXX -o "$CLI_OUT" atlas_cli.cpp -O2 -std=c++17
    ls -lh "$CLI_OUT"
    echo "[Atlas] OK -- $CLI_OUT built"
fi
