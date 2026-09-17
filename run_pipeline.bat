@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ===============================================================================
echo                GeoFUSE SentinelGuard - Pipeline Runner
echo ===============================================================================
echo.

REM Prefer GPU virtual environment if available
if exist ".venv_gpu\Scripts\python.exe" (
    set "PY_CMD=.venv_gpu\Scripts\python.exe"
    set "ST_CMD=.venv_gpu\Scripts\streamlit.exe"
) else (
    set "PY_CMD=python"
    set "ST_CMD=streamlit"
)

REM If arguments were passed directly from CLI, forward them straight to python
if not "%~1"=="" (
    %PY_CMD% scripts\reproduce_all.py %*
    goto :end
)

echo Select an execution mode:
echo   [1] Direct Real Sentinel-2 -> 4m Super Resolution (Live Direct Inference)
echo   [2] Synthetic Degradation Benchmark (Controlled Degrade-and-Recover Evaluation)
echo   [3] Full Verification Suite (Benchmark Pipeline + PyTest Suite)
echo   [4] Launch Streamlit Interactive Web Dashboard (Real & Benchmark Modes)
echo   [5] Generate Full-Scene Overview & Tile Locator Graphic
echo.

set /p choice="Enter choice [1-5, default=1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo.
    echo [RUNNING] Direct Real Sentinel-2 -> 4m Super Resolution...
    set /p real_input="Enter Sentinel-2 scene folder, 4 bands, or GeoTIFF [default=data\additional_datasets\urban_core]: "
    if "!real_input!"=="" set real_input=data\additional_datasets\urban_core
    set /p real_tile="Optional: Enter tile index (0..24) or press Enter for full scene: "
    set "tile_param="
    if not "!real_tile!"=="" set "tile_param=--tile !real_tile!"
    %PY_CMD% scripts\run_custom_input.py --input !real_input! --mode real !tile_param!
) else if "%choice%"=="2" (
    echo.
    echo [RUNNING] Synthetic Degradation Benchmark Pipeline...
    %PY_CMD% scripts\reproduce_all.py --skip-training --skip-tests
) else if "%choice%"=="3" (
    echo.
    echo [RUNNING] Full Verification Suite (Benchmark + PyTest Tests)...
    %PY_CMD% scripts\reproduce_all.py --skip-training
) else if "%choice%"=="4" (
    echo.
    echo [LAUNCHING] Streamlit Interactive Web Dashboard...
    %ST_CMD% run src\dashboard\app.py
) else if "%choice%"=="5" (
    echo.
    set /p tile_choice="Enter tile index to highlight [0..24, default=0]: "
    if "!tile_choice!"=="" set tile_choice=0
    echo [RUNNING] Generating Full-Scene Macro Overview for Tile #!tile_choice!...
    %PY_CMD% scripts\render_scene_overview.py --tile !tile_choice!
) else (
    echo.
    echo [RUNNING] Direct Real Sentinel-2 -> 4m Super Resolution...
    %PY_CMD% scripts\run_custom_input.py --input data\additional_datasets\urban_core --mode real
)


:end
echo.
echo ===============================================================================
echo Execution finished.
echo ===============================================================================
pause
