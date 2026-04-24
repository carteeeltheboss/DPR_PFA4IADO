@echo off
setlocal

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "VENV_DIR=%ROOT_DIR%\DPR_MedFusionNet\venv"
set "ACTIVATE_SCRIPT=%VENV_DIR%\Scripts\activate.bat"
set "MED_REQ=%ROOT_DIR%\DPR_MedFusionNet\requirements.txt"
set "WEB_REQ=%ROOT_DIR%\DPR_WebService\requirements.txt"
set "APP_FILE=%ROOT_DIR%\DPR_WebService\app.py"

if not exist "%ACTIVATE_SCRIPT%" (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo Python 3 is required but was not found on PATH.
            exit /b 1
        )
        python -m venv "%VENV_DIR%"
    )
)

call "%ACTIVATE_SCRIPT%"
if errorlevel 1 exit /b %errorlevel%

python -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

python -m pip install -r "%MED_REQ%"
if errorlevel 1 exit /b %errorlevel%

python -m pip install -r "%WEB_REQ%"
if errorlevel 1 exit /b %errorlevel%

cd /d "%ROOT_DIR%"
python "%APP_FILE%"
