@echo off
echo ===================================================
echo   Starting Deep Learning Vision Decision Agent
echo ===================================================

set "PY=C:\Users\syeda\AppData\Local\Programs\Python\Python312\python.exe"
echo [OK] Using Python: %PY%

if exist "venv\Scripts\python.exe" goto :START_SERVER

echo Creating virtual environment...
"%PY%" -m venv venv

echo Installing deep learning requirements...
call venv\Scripts\python.exe -m pip install --upgrade pip
call venv\Scripts\python.exe -m pip install -r requirements.txt

:START_SERVER
echo.
echo ===================================================
echo   Launching Agent Server on http://127.0.0.1:8000
echo ===================================================
call venv\Scripts\python.exe app.py
pause
