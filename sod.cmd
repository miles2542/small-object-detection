@echo off
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%.venv\Scripts\sod.exe" (
    "%SCRIPT_DIR%.venv\Scripts\sod.exe" %*
) else (
    uv run sod %*
)
