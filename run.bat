@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo First run: installing... this takes a few minutes.
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
start "" ".venv\Scripts\pythonw.exe" app.py
