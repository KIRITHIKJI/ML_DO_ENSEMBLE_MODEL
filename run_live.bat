@echo off
REM ===========================================================================
REM run_live.bat - one production run of the DO early-warning system.
REM
REM   fetch the live window  ->  forecast every station  ->  write
REM   results\live\forecast_<UTC timestamp>.csv and append to
REM   results\live\log.parquet  ->  update the monitoring csv.
REM
REM Schedule it hourly with Windows Task Scheduler (see README.md):
REM
REM   schtasks /create /tn "DO early warning" /tr "A:\ML_MAIN_V2\run_live.bat" ^
REM            /sc hourly /mo 1 /st 00:05 /rl LIMITED
REM
REM Everything is read from config.yaml and config_deploy.yaml; the only
REM thing this file decides is where the project and its virtualenv live.
REM Edit PROJECT and PYTHON if yours are elsewhere.
REM ===========================================================================

setlocal

set "PROJECT=%~dp0"
cd /d "%PROJECT%"

REM The virtualenv's python if there is one, otherwise whatever is on PATH.
set "PYTHON=%PROJECT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if not exist "%PROJECT%results\live" mkdir "%PROJECT%results\live"
set "LOG=%PROJECT%results\live\run_live.log"

echo. >> "%LOG%"
echo ======================================================== >> "%LOG%"
echo run_live.bat starting %DATE% %TIME% >> "%LOG%"

REM The forecast. Deep members on the GPU; --allow-cpu would let it fall
REM back silently, which is exactly what this project does not do, so it is
REM left off on purpose: a missing CUDA build should be noticed.
"%PYTHON%" -u -m src.deploy.run >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo run_live.bat: the forecast failed with exit code %RC% >> "%LOG%"
    echo run_live.bat: the forecast failed with exit code %RC%
    endlocal
    exit /b %RC%
)

REM Scoring whatever forecasts the sensors have now caught up with. This is
REM cheap and reads only files, so it runs every time.
"%PYTHON%" -u -m src.deploy.monitor >> "%LOG%" 2>&1

echo run_live.bat finished %DATE% %TIME% >> "%LOG%"
endlocal
exit /b 0
