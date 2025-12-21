@echo off
setlocal

:: Basic render test with fixed seed
set EXE=%~dp0target\release\nanotracer-rs.exe

if not exist "%EXE%" (
    echo Building release...
    cargo build --release
    if errorlevel 1 exit /b 1
)

"%EXE%" -n 220 --seed 2025 %*
