@echo off
rem Common env for vr180 launchers. Not run directly.
set "PROJECT_DIR=%~dp0.."
for %%I in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fI"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
cd /d "%PROJECT_DIR%"
