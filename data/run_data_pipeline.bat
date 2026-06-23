@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"

echo ======================================================
echo            STARTING FIFA DATA PIPELINE                
echo ======================================================

:: ==============================================================================
:: RUNNING THE STEPS
:: ==============================================================================

:: 1: EXTRACTION Wikipedia
call :run_step "[1/4] Extracting External Metadata (Wikipedia)..." "%SCRIPT_DIR%extract_external_metadata.py" "%SCRIPT_DIR%external_metadata\world_teams_metadata.csv"
if %errorlevel% neq 0 exit /b %errorlevel%

:: 2: EXTRACTION FBref e FootyStats
call :run_step "[2/4] Extracting Raw Data (World Cup/FBref)..." "%SCRIPT_DIR%extract_data.py" "%SCRIPT_DIR%raw\fbref_world_cup_games.csv"
if %errorlevel% neq 0 exit /b %errorlevel%

:: 3: PROCESSING AND ENRICHMENT
call :run_step "[3/4] Processing Data, Clubs and Weights..." "%SCRIPT_DIR%process_data.py" "%SCRIPT_DIR%processed\processed_players.csv"
if %errorlevel% neq 0 exit /b %errorlevel%

:: 4: REFINEMENT AND FEATURE ENGINEERING
call :run_step "[4/4] Calculating Efficiency and Synergy (Machine Learning)..." "%SCRIPT_DIR%refine_data.py" "%SCRIPT_DIR%refined\ml_national_synergy_features.csv"
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==========================================================
echo          DATA PIPELINE CONCLUDED SUCCESSFULLY           
echo ==========================================================
exit /b 0

:: =====================================================================================================================
:: Helper function equivalent (Label) to run the step and verify the validation file
:: =====================================================================================================================
:run_step
set "step_title=%~1"
set "python_script=%~2"
set "validation_file=%~3"

echo.
echo --^> %step_title%

if exist "%validation_file%" (
    echo    [!] The data for this step already exists in the directory.
    set "answer=N"
    set /p answer="   Would you like to run this script again? (y/N): "
    
    if /I "!answer!"=="y" (
        echo    Executando %python_script%...
        python "%python_script%"
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else (
        echo    ^>^> Pulando etapa ^(Cache mantido^).
    )
) else (
    echo    [!] Data not found. Executing script...
    python "%python_script%"
    if !errorlevel! neq 0 exit /b !errorlevel!
)
exit /b 0