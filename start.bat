@echo off
cd /d "%~dp0"
set VENV_PYTHON=venv\Scripts\python.exe

if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Is Python installed and in PATH?
        pause
        exit /b 1
    )
)

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python was not found.
    echo [ERROR] Delete the incomplete venv folder and run start.bat again.
    pause
    exit /b 1
)

rem Keep an existing virtual environment in sync with requirements.txt.
set "REQUIREMENTS_MARKER=venv\.requirements.sha256"
set "REQUIREMENTS_HASH="
for /f %%H in ('%VENV_PYTHON% -c "import hashlib; print(hashlib.sha256(open('requirements.txt','rb').read()).hexdigest())"') do set "REQUIREMENTS_HASH=%%H"
set "INSTALLED_HASH="
if exist "%REQUIREMENTS_MARKER%" set /p INSTALLED_HASH=<"%REQUIREMENTS_MARKER%"
if not "%REQUIREMENTS_HASH%"=="%INSTALLED_HASH%" goto :install_dependencies
%VENV_PYTHON% -c "import fastapi, uvicorn, pydantic, litellm, multipart, httpx, anyio, PIL" >nul 2>&1
if errorlevel 1 goto :install_dependencies
goto :dependencies_ready

:install_dependencies
echo [INFO] Installing or updating dependencies...
%VENV_PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check network or requirements.txt.
    pause
    exit /b 1
)
>"%REQUIREMENTS_MARKER%" echo %REQUIREMENTS_HASH%

:dependencies_ready

echo [INFO] Starting LLM Gateway...
echo [INFO] Admin UI: http://localhost:8000
echo [INFO] API Docs:  http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo.

%VENV_PYTHON% main.py
pause
