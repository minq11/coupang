@echo off
call "%~dp0_env.bat"
title VR180 Install GeoCalib
echo [VR180] Installing GeoCalib (git clone + pip install).
echo         Used for Stage 1 (automatic HFOV estimation).
echo         No HF token required. Model weights auto-download on first use.
echo.
uv sync --extra geocalib
if errorlevel 1 (
  echo [VR180] Install failed.
  pause
  exit /b 1
)
echo.
echo [VR180] Done. To use GeoCalib for auto FOV estimation:
echo   -s stage_01_fov.auto_estimate=true
echo   -s stage_01_fov.estimator=geocalib
echo.
pause
