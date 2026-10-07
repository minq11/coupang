@echo off
call "%~dp0_env.bat"
title VR180 GUI
echo [VR180] Starting GUI server... your browser will open automatically.
echo         Close this window to stop the server.
echo.
start "" "http://127.0.0.1:7860"
uv run vr180-gui
echo.
echo [VR180] Server stopped.
pause
