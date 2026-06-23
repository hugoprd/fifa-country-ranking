@echo off
setlocal enabledelayedexpansion

@echo off
setlocal enabledelayedexpansion

:: defines cmd dir
set "SCRIPT_DIR=%~dp0"
:: go up one level to define the root dir
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI\"

echo ======================================================
echo        STARTING THE FULL FIFA RANKING PIPELINE        
echo ======================================================

echo.
echo ▶ [ STEP 0 ] Environment Setup...

call "%ROOT_DIR%cmd\environment_setup.bat"

call "%ROOT_DIR%.venv\Scripts\activate.bat"

:: ==============================================================================
:: 1: DATA PIPELINE
:: ==============================================================================
echo.
echo ▶ [ STEP 1 ] Executing Data Extraction and Refinement...

call "%ROOT_DIR%data\run_data_pipeline.bat"

:: Verifica se houve erro no passo anterior
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Data pipeline failed. Aborting full execution.
    exit /b 1
)

:: ==============================================================================
:: 2: MACHINE LEARNING PIPELINE
:: ==============================================================================
echo.
echo ▶ [ STEP 2 ] Executing Model Training and Inference...

call "%ROOT_DIR%ml_model_scripts\run_model_training.bat"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Model training failed. Aborting full execution.
    exit /b 1
)

echo.
echo ======================================================
echo        FULL PIPELINE CONCLUDED SUCCESSFULLY!          
echo ======================================================