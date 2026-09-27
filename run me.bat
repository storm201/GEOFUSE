@echo off
setlocal EnableDelayedExpansion
title GeoFUSE SentinelGuard Launcher
cd /d "%~dp0"

echo ===============================================================================
echo                GeoFUSE SentinelGuard - Unified Application Launcher
echo ===============================================================================
echo.

REM Detect Python environment (prefer .venv_gpu, then .venv, then venv, then system python)
if exist ".venv_gpu\Scripts\python.exe" (
    set "PY_CMD=.venv_gpu\Scripts\python.exe"
    echo [*] Using GPU Virtual Environment: .venv_gpu
) else if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
    echo [*] Using Virtual Environment: .venv
) else if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
    echo [*] Using Virtual Environment: venv
) else (
    set "PY_CMD=python"
    echo [!] Using System Python: python
)

REM Verify dependency readiness in selected Python environment
%PY_CMD% -c "import torch, fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo.
    echo ===============================================================================
    echo [!] Dependencies not yet installed in: %PY_CMD%
    echo ===============================================================================
    echo This appears to be a fresh clone. Would you like to automatically create a
    echo virtual environment (.venv) and install all dependencies?
    echo.
    set "setup_choice="
    set /p setup_choice="Create .venv and install requirements? [Y/n, default=Y]: "
    if "!setup_choice!"=="" set setup_choice=Y
    if /i "!setup_choice!"=="Y" (
        echo [*] Creating virtual environment in .venv...
        python -m venv .venv
        if exist ".venv\Scripts\python.exe" (
            set "PY_CMD=.venv\Scripts\python.exe"
            echo [*] Upgrading pip...
            !PY_CMD! -m pip install --upgrade pip
            echo [*] Installing dependencies from requirements.txt...
            !PY_CMD! -m pip install -r requirements.txt
            echo [✓] Environment setup complete.
        ) else (
            echo [✗] Error: Failed to create .venv. Ensure Python 3.10-3.12 is on your system PATH.
            pause
            exit /b 1
        )
    )
)

REM Verify PyTorch / GPU availability
%PY_CMD% -c "import torch; print('    PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())" 2>nul
echo.

REM If argument passed directly, forward to execution mode
if "%~1"=="1" goto :launch_webapp
if "%~1"=="2" goto :launch_streamlit
if "%~1"=="3" goto :launch_tests
if "%~1"=="4" goto :launch_pipeline
if /i "%~1"=="web" goto :launch_webapp
if /i "%~1"=="api" goto :launch_webapp
if /i "%~1"=="test" goto :launch_tests
if /i "%~1"=="streamlit" goto :launch_streamlit

echo Select an execution mode:
echo   [1] Launch GeoFUSE Web Application (FastAPI GPU Backend + React UI) [Default]
echo   [2] Launch Streamlit Interactive Dashboard
echo   [3] Run Full Verification Test Suite (136+ Pytest Checks)
echo   [4] Run Pipeline CLI / Benchmark Modes
echo.

set "choice="
set /p choice="Enter choice [1-4, default=1]: "
if "%choice%"=="" set choice=1
set "choice=%choice: =%"

if "%choice%"=="1" goto :launch_webapp
if "%choice%"=="2" goto :launch_streamlit
if "%choice%"=="3" goto :launch_tests
if "%choice%"=="4" goto :launch_pipeline
goto :launch_webapp

:launch_webapp
%PY_CMD% scripts\launch_server.py
goto :end

:launch_streamlit
echo.
echo [*] Starting Streamlit Interactive Dashboard...
if exist ".venv_gpu\Scripts\streamlit.exe" (
    .venv_gpu\Scripts\streamlit.exe run src\dashboard\app.py
) else (
    streamlit run src\dashboard\app.py
)
goto :end

:launch_tests
echo.
echo [*] Executing Full Verification Test Suite...
%PY_CMD% -m pytest tests/ -v
echo.
pause
goto :end

:launch_pipeline
call run_pipeline.bat
goto :end

:end
exit /b 0
