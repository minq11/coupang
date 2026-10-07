@echo off
call "%~dp0_env.bat"
title VR180 Fetch Models
echo [VR180] Downloading Depth Anything V2 + SDXL inpainting + RAFT-Stereo.
echo         This downloads several GB.
echo.
pause
uv run vr180 fetch-models --da2 --sdxl --raft-stereo
echo.
echo [VR180] Done. To use the models when converting:
echo   -s stage_03_outpaint.inpainter=sdxl
echo   -s stage_04_stereo.matcher=raft_stereo
echo   -s stage_05_depth.estimator=da2_large
echo.
pause
