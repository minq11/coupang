@echo off
call "%~dp0_env.bat"
title VR180 Update
echo [VR180] Syncing dependencies...
uv sync
echo.
pause
