@echo off
REM Musical Weather Development Helper
REM Usage: dev.bat [command]

set PYTHON_EXE=D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe
set PROJECT_DIR=D:\Backup\repos\Musical-Weather\musical_weather_app

cd /d "%PROJECT_DIR%"

if "%1"=="" goto dev
if "%1"=="dev" goto dev
if "%1"=="prod" goto prod
if "%1"=="test" goto test
if "%1"=="install" goto install
if "%1"=="help" goto help
goto help

:dev
echo 🚀 Starting development server...
echo    URL: http://localhost:5000
echo    Dev Info: http://localhost:5000/dev
echo    Health: http://localhost:5000/health
echo.
"%PYTHON_EXE%" dev_main.py
goto end

:prod
echo 🏭 Starting production server...
"%PYTHON_EXE%" main.py
goto end

:test
echo 🧪 Running minimal test...
"%PYTHON_EXE%" minimal_test.py
goto end

:install
echo 📦 Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
goto end

:help
echo Musical Weather Development Helper
echo ================================
echo Available commands:
echo   dev     - Start development server (default)
echo   prod    - Start production server
echo   test    - Start minimal test server
echo   install - Install Python dependencies
echo   help    - Show this help
echo.
echo Usage: dev.bat [command]
goto end

:end