@echo off
setlocal

:: Low-quality fast Gaussian splat export
set EXE=%~dp0target\release\nanotracer-rs.exe

if not exist "%EXE%" (
    echo Building release...
    cargo build --release
    if errorlevel 1 exit /b 1
)

"%EXE%" -n 64 --seed 123 ^
    -S scene_lo.ply ^
    --splat-density 512 ^
    --sh-samples 36 ^
    --sh-glossy-mult 1.5 ^
    --radiance-clamp 20 ^
    --detail-boost 1.5 ^
    --detail-boost-max 3.0 ^
    --light-sampling one ^
    --splat-scale 0.02 ^
    -m 8 -r 4 -f 6 ^
    %*
