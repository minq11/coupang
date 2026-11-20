@echo off
call "%~dp0_env.bat"
title VR180 Person AR Player
echo [VR180] Person AR player: http://localhost:8080
echo.
echo   Quest: connect USB, then this window runs "adb reverse tcp:8080 tcp:8080"
echo          and you open http://localhost:8080 in the Quest browser, press "AR".
echo.
where adb >nul 2>nul
if %ERRORLEVEL%==0 (
  adb reverse tcp:8080 tcp:8080
) else (
  echo   [!] adb not found. Install Meta Quest Developer Hub or Android platform-tools,
  echo       then run: adb reverse tcp:8080 tcp:8080
)
echo.
start "" "http://localhost:8080"
uv run vr180 player --port 8080
pause
