@echo off
call "%~dp0_env.bat"
title VR180 Convert
if "%~1"=="" (
  echo [VR180] Drag a 3D SBS image file onto this icon.
  echo         Or use "VR180 GUI".
  echo.
  pause
  exit /b 1
)
echo [VR180] Input: %~1
echo.
uv run vr180 run -i "%~1"
echo.
echo [VR180] Output is in work\<latest-job>\07_output\.
echo         Press Enter to open the work folder.
pause >nul
start "" "%PROJECT_DIR%\work"
