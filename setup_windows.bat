@echo off
REM One-time setup for Windows: creates a virtual environment and installs packages.
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Now double-click run_app.bat
pause
