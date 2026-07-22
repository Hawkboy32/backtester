@echo off
cd /d "%~dp0"
echo.| ".venv\Scripts\streamlit.exe" run app.py
pause
