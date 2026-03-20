@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
echo Starting ML service at http://127.0.0.1:8000 ...
echo Keep this window open. Use Ctrl+C to stop.
python main.py
pause
