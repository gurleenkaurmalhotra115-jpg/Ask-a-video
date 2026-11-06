@echo off
title VidQuery - Ask a Video Launcher
echo ======================================================================
echo Starting VidQuery Interactive Web Application...
echo ======================================================================

cd /d "%~dp0"

if not exist ".\venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    echo Please ensure the venv directory exists.
    pause
    exit /b 1
)

echo.
echo [1/2] Launching Streamlit Server on http://localhost:8502 ...
start "" "http://localhost:8502"
.\venv\Scripts\streamlit.exe run app.py --server.port 8502 --server.headless false

pause
