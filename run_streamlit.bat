@echo off
echo ===================================================
echo   Launching Vision Decision Agent on Streamlit
echo ===================================================

cd /d "%~dp0"

if not exist "venv\Scripts\streamlit.exe" (
    echo Installing Streamlit into virtual environment...
    venv\Scripts\python.exe -m pip install streamlit
)

echo Starting Streamlit App on http://localhost:8501 ...
venv\Scripts\python.exe -m streamlit run streamlit_app.py
pause
