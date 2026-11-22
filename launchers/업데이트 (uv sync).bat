@echo off
call "%~dp0_env.bat"
title VR180 Update
echo [VR180] Syncing dependencies...
rem --inexact keeps optional extras (GeoCalib, LightGlue) installed separately
uv sync --inexact
echo.
pause
