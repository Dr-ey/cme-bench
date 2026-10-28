@echo off
rem cme-bench installer (Windows cmd). Usage: install.cmd [118|121|128|cpu]
setlocal EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

set CUDA=%1
if "%CUDA%"=="" (
  set CUDA=cpu
  where nvidia-smi > nul 2>&1
  if !errorlevel! == 0 (
    set CUDA=118
    for /f "tokens=1 delims=." %%c in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader') do (
      if %%c GEQ 12 set CUDA=128
    )
  )
)
echo [install] torch-branch=%CUDA%

py -3.10 -m venv .venv 2> nul || py -3.11 -m venv .venv || python -m venv .venv
if errorlevel 1 ( echo Python 3.10/3.11 required & exit /b 1 )
call .venv\Scripts\activate.bat

python -m pip install -U pip
if "%CUDA%"=="cpu" (
  python -m pip install -r requirements\torch-cpu.txt
) else (
  python -m pip install -r requirements\torch-cu%CUDA%.txt
)
if errorlevel 1 exit /b 1
python -m pip install -r requirements\base.txt || exit /b 1
python -m pip install --no-deps -r requirements\extras.txt || exit /b 1
python -m pip install -e . || exit /b 1

python scripts\00_check_env.py
echo [install] done. Activate with: .venv\Scripts\activate.bat
endlocal
