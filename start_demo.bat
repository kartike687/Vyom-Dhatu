@echo off
echo ======================================================================
echo   STARTING MOIL MANGANESE INTELLIGENCE COMMAND CENTRE & DIGITAL TWIN
echo   Smart India Hackathon 2026
echo ======================================================================
echo.
echo Checking Python environment...
python --version
echo.
echo Launching Digital Twin on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python run_server.py
pause
