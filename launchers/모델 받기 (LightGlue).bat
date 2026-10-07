@echo off
call "%~dp0_env.bat"
title VR180 Install LightGlue
echo [VR180] Installing LightGlue (git clone + pip install).
echo         Used for Stage 0 (left/right feature matching) when SIFT is weak.
echo         No HF token required.
echo.
uv sync --extra lightglue
if errorlevel 1 (
  echo [VR180] Install failed.
  pause
  exit /b 1
)
echo.
echo [VR180] Done. To use LightGlue:
echo   -s stage_00_split.matcher=lightglue
echo.
pause
