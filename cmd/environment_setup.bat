@echo off

echo ======================================================
echo        AUTOMATIC ENVIRONMENT SETUP (UV) - WINDOWS     
echo ======================================================

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ▶ 'uv' not found. Downloading and installing astral-sh/uv...
    
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    echo ▶ 'uv' successfully installed!
) else (
    echo.
    echo ▶ 'uv' is already installed.
)

echo.
echo ▶ uv version:
uv --version

echo.
echo ▶ Syncing Python environment and installing dependencies...
uv sync

echo.
echo ======================================================
echo        ENVIRONMENT SETUP COMPLETED SUCCESSFULLY       
echo ======================================================