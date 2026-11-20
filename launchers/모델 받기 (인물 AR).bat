@echo off
call "%~dp0_env.bat"
title VR180 Fetch Models (Person AR)
echo [VR180] Downloading Person AR models: SAM 2.1 small, BiRefNet-matting,
echo         Depth Anything V2 Small, Video Depth Anything Small, RF-DETR-Seg.
echo         All are licensed for commercial use (Apache-2.0 / MIT).
echo.
pause
uv run vr180 fetch-models --person
echo.
echo [VR180] Done. Open the GUI and use the "Person AR" tab.
pause
