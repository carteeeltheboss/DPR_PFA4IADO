@echo off
setlocal

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "VENV_DIR=%ROOT_DIR%\DPR_MedFusionNet\venv"
set "ACTIVATE_SCRIPT=%VENV_DIR%\Scripts\activate.bat"
set "APP_FILE=%ROOT_DIR%\DPR_WebService\app.py"

if not exist "%ACTIVATE_SCRIPT%" (
    echo Virtual environment not found. Run run_first.bat first.
    exit /b 1
)

call "%ACTIVATE_SCRIPT%"
if errorlevel 1 exit /b %errorlevel%

cd /d "%ROOT_DIR%"
python "%APP_FILE%"
