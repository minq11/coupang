@echo off
call "%~dp0_env.bat"
title VR180 Make Sample
echo [VR180] Creating synthetic test image at samples\synthetic_sbs.png
uv run vr180 make-sample samples/synthetic_sbs.png
echo.
pause
