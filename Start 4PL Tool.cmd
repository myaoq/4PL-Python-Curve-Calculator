@echo off
cd /d "%~dp0"
set "CALIBRATION_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%CALIBRATION_PY%" (
    "%CALIBRATION_PY%" calibration.py
) else (
    python calibration.py
)
if errorlevel 1 pause
