@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%..\.."
set "DEFAULT_CONFIG=%SCRIPT_DIR%live-eval.config.local.json"
set "FALLBACK_CONFIG=%SCRIPT_DIR%live-eval.config.json"

cd /d "%REPO_DIR%" || goto :error

if exist "%DEFAULT_CONFIG%" (
  set "CONFIG=%DEFAULT_CONFIG%"
) else if exist "%FALLBACK_CONFIG%" (
  set "CONFIG=%FALLBACK_CONFIG%"
) else (
  echo Missing config file.
  echo Create tools\live_eval\live-eval.config.local.json from tools\live_eval\live-eval.config.example.json first.
  goto :error
)

echo Running live gateway evaluation...
echo Config: %CONFIG%
echo.

if not "%~1"=="" goto :run_with_args

echo Choose evaluation mode:
echo   1. Quick smoke test: first model from config, streaming, skip multimodal
echo   2. All models from config / /v1/models
echo   3. Config defaults exactly
echo.
set /p "MODE=Select 1, 2, or 3 [1]: "
if "%MODE%"=="" set "MODE=1"

if "%MODE%"=="2" (
  python "%SCRIPT_DIR%live_eval.py" --config "%CONFIG%" --skip-multimodal
  goto :after_run
)

if "%MODE%"=="3" (
  python "%SCRIPT_DIR%live_eval.py" --config "%CONFIG%"
  goto :after_run
)

python "%SCRIPT_DIR%live_eval.py" --config "%CONFIG%" --limit 1 --skip-multimodal
goto :after_run

:run_with_args
python "%SCRIPT_DIR%live_eval.py" --config "%CONFIG%" %*

:after_run
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Live evaluation completed successfully.
) else (
  echo Live evaluation finished with failures. Exit code: %EXIT_CODE%
)
echo Reports are written to reports\live-eval\.
echo.
pause
exit /b %EXIT_CODE%

:error
echo.
pause
exit /b 1
