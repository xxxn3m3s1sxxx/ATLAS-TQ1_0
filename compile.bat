@echo off
REM Build atlas.dll + atlas.exe from source using Clang (LLVM MinGW)
REM Requires: llvm-mingw in PATH
REM
REM OpenMP: enabled with -fopenmp. libomp.dll must be discoverable at runtime.
REM The LLVM-MinGW distro ships libomp.dll in x86_64-w64-mingw32\bin — copy
REM it next to atlas.exe, or add that directory to PATH.

set CC=clang++

echo [Atlas] Compiling atlas.dll...
%CC% -shared -o atlas.dll atlas_api.cpp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -fopenmp
if %ERRORLEVEL% NEQ 0 (
    echo [Atlas] FAILED -- DLL build error
    exit /b 1
)
echo [Atlas] OK -- atlas.dll built

echo [Atlas] Compiling atlas.exe (standalone CLI)...
%CC% -o atlas.exe atlas_cli.cpp -O2 -std=c++17
if %ERRORLEVEL% EQU 0 (
    echo [Atlas] OK -- atlas.exe built
) else (
    echo [Atlas] WARNING -- atlas.exe build failed (CLI unavailable)
    echo [Atlas] atlas_cli.cpp is optional — DLL build succeeded
    exit /b 0
)
