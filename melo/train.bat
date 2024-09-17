@echo off
setlocal enabledelayedexpansion

:: Set the relative path to the config file as a parameter
set "RELATIVE_CONFIG_PATH=data\example\config.json"

:: Get the directory of the batch file
set "SCRIPT_DIR=%~dp0"

:: Construct the full path to the config file
set "CONFIG=%SCRIPT_DIR%%RELATIVE_CONFIG_PATH%"

:: Extract the model name from the config path
for %%F in ("%CONFIG%") do set "MODEL_NAME=%%~dpnF"

:loop
python train.py -c "%CONFIG%" -m "%MODEL_NAME%"
for /f "tokens=2" %%a in ('tasklist ^| findstr python') do (
    taskkill /F /PID %%a
)
timeout /t 30
goto loop