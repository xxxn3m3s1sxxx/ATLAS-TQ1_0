@echo off
REM Build atlas.dll + atlas.exe from source using Clang (LLVM MinGW)
REM Requires: llvm-mingw in PATH
REM
REM OpenMP: enabled with -fopenmp. libomp.dll must be discoverable at runtime.
REM The LLVM-MinGW distro ships libomp.dll in x86_64-w64-mingw32\bin — copy
REM it next to atlas.exe, or add that directory to PATH.
REM
REM Usage:
REM   compile              Release build (atlas.dll)
REM   compile debug        Debug build (atlas_d.dll, -DATLAS_DEBUG_MODE -O0 -g)
REM   compile test         Release build + run fixture tests

set CC=clang++

if "%1"=="debug" goto build_debug
if "%1"=="test" goto build_test

:build_release
echo [Atlas] Compiling atlas.dll (release)...
%CC% -shared -o atlas.dll atlas_api.cpp -O2 -mavx2 -mfma -mf16c -ffast-math -std=c++17 -fopenmp
if %ERRORLEVEL% NEQ 0 (
    echo [Atlas] FAILED -- DLL build error
    exit /b 1
)
echo [Atlas] OK -- atlas.dll built

echo [Atlas] Compiling atlas.exe (standalone CLI)...
%CC% -o atlas.exe atlas_cli.cpp -O2 -std=c++17 -lshell32
if %ERRORLEVEL% EQU 0 (
    echo [Atlas] OK -- atlas.exe built
) else (
    echo [Atlas] WARNING -- atlas.exe build failed (CLI unavailable)
    echo [Atlas] atlas_cli.cpp is optional — DLL build succeeded
)
goto :eof

:build_debug
echo [Atlas] Compiling atlas_d.dll (debug)...
%CC% -shared -o atlas_d.dll atlas_api.cpp -DATLAS_DEBUG_MODE -O0 -g -mavx2 -mfma -mf16c -std=c++17 -fopenmp
if %ERRORLEVEL% NEQ 0 (
    echo [Atlas] FAILED -- Debug DLL build error
    exit /b 1
)
echo [Atlas] OK -- atlas_d.dll built
goto :eof

:build_test
call :build_release
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
echo [Atlas] Running fixture tests...
python -m pytest tests/test_fixtures.py -v
if %ERRORLEVEL% NEQ 0 (
    echo [Atlas] FAILED -- fixture tests
    exit /b 1
)
echo [Atlas] OK -- all tests passed
goto :eof
