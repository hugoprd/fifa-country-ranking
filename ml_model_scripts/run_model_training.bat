@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"

for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI\"

set "PYTHON_EXEC=%ROOT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXEC%" (
    set "PYTHON_EXEC=python"
)

echo ======================================================
echo              STARTING MODEL PIPELINE                  
echo ======================================================

:: ==============================================================================
:: RUNNING THE STEPS
:: ==============================================================================

:: 1: Loads the best architecture (hyperparameters) for the model
call :run_step "[1/3] Loading best architecture..." "%SCRIPT_DIR%architecture_loader.py"
if %errorlevel% neq 0 exit /b %errorlevel%

:: 2: Train the model
call :run_step "[2/3] Starting the model train..." "%SCRIPT_DIR%train_model.py"
if %errorlevel% neq 0 exit /b %errorlevel%

:: 3: Generate the Ranking
call :run_step "[3/3] Generating the National Teams Ranking..." "%SCRIPT_DIR%generate_fifa_ranking.py"
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==========================================================
echo            MODEL PIPELINE CONCLUDED SUCCESSFULLY             
echo ==========================================================
exit /b 0

:: ==============================================================================
:: Helper function equivalent (Label) to run a step and log the progress
:: ==============================================================================
:run_step
set "step_message=%~1"
set "script_path=%~2"

echo.
echo ------------------------------------------------------
echo ▶ %step_message%
echo ------------------------------------------------------
echo.

"%PYTHON_EXEC%" "%script_path%"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Step failed. Script execution aborted.
    exit /b 1
)
exit /b 0