@echo off
REM Starts the dashboard (run setup_windows.bat once before this).
call venv\Scripts\activate.bat
streamlit run app.py
pause
