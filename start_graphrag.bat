@echo off
setlocal

REM Run this file from the project root (the folder containing .venv, api, and ui).
set "PROJECT_DIR=%~dp0"

if not exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    echo ERROR: Could not find .venv\Scripts\activate.bat
    echo Place this BAT file in your project root and make sure the virtual environment exists.
    pause
    exit /b 1
)

 echo Starting GraphRAG API and Streamlit UI...

start "GraphRAG API - port 8000" /D "%PROJECT_DIR%" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn api.server:app --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak >nul

start "GraphRAG Streamlit UI" /D "%PROJECT_DIR%" cmd /k "call .venv\Scripts\activate.bat && streamlit run ui/app.py"

endlocal
