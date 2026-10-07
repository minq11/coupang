@echo off
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
title VR180 Fetch FLUX

rem --- Check if HF_TOKEN is already configured (via .env or environment)
set "HAS_TOKEN=0"
if exist ".env" (
  findstr /b /c:"HF_TOKEN=hf_" ".env" >nul 2>nul
  if !errorlevel! EQU 0 set "HAS_TOKEN=1"
)
if defined HF_TOKEN set "HAS_TOKEN=1"

if "!HAS_TOKEN!"=="1" (
  echo [VR180] HF_TOKEN already configured. Skipping token setup.
  goto :DOWNLOAD
)

echo.
echo ========================================================================
echo   FLUX.1 Fill dev is a GATED model and requires a Hugging Face token.
echo.
echo   Opening two browser tabs:
echo     1) Create a token  (any "read" token is enough)
echo     2) Accept the FLUX.1 Fill dev license (one-time, free)
echo.
echo   Then come back here and paste the token.
echo   NOTE: license is NON-COMMERCIAL use only.
echo ========================================================================
echo.
start "" "https://huggingface.co/settings/tokens"
start "" "https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
echo.
set /p TOKEN_INPUT=Paste token (starts with hf_) then press Enter:
if "!TOKEN_INPUT!"=="" (
  echo [VR180] No token entered. Aborting.
  pause
  exit /b 1
)
if not exist ".env" copy /y ".env.example" ".env" >nul
>>".env" echo HF_TOKEN=!TOKEN_INPUT!
echo [VR180] Token appended to .env. (You can edit the file to change it later.)
echo.

:DOWNLOAD
echo [VR180] Downloading FLUX.1 Fill dev GGUF Q4 + text encoder/VAE (~12 GB).
echo         This can take 20-60 minutes depending on your connection.
echo.
pause
uv run vr180 fetch-models --flux
if errorlevel 1 (
  echo.
  echo [VR180] Fetch failed. Common causes:
  echo   - Token has no access  ^(did you accept the license page?^)
  echo   - Disk space  ^(need ~15 GB free in HF cache / models folder^)
  pause
  exit /b 1
)
echo.
echo [VR180] Done. To use FLUX inpainting when converting:
echo   -s stage_03_outpaint.inpainter=flux
echo   -s stage_06_right.inpainter=flux
echo.
pause
