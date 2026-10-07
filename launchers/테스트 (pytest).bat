@echo off
call "%~dp0_env.bat"
title VR180 pytest
uv run pytest
echo.
pause
