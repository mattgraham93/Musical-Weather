# Musical Weather Development Helper Script
# Run this script from the musical_weather_app directory

param(
    [string]$Command = "dev"
)

$ProjectDir = "D:\Backup\repos\Musical-Weather\musical_weather_app"
$PythonExe = "D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe"

Write-Host "🎵 Musical Weather Development Helper" -ForegroundColor Cyan
Write-Host "=" * 40 -ForegroundColor Cyan

# Change to project directory
Set-Location $ProjectDir

switch ($Command.ToLower()) {
    "dev" {
        Write-Host "🚀 Starting development server..." -ForegroundColor Green
        Write-Host "   URL: http://localhost:5000" -ForegroundColor Yellow
        Write-Host "   Dev Info: http://localhost:5000/dev" -ForegroundColor Yellow
        Write-Host "   Health: http://localhost:5000/health" -ForegroundColor Yellow
        Write-Host ""
        & $PythonExe dev_main.py
    }
    "prod" {
        Write-Host "🏭 Starting production server..." -ForegroundColor Green
        & $PythonExe main.py
    }
    "test" {
        Write-Host "🧪 Running minimal test..." -ForegroundColor Green
        & $PythonExe minimal_test.py
    }
    "install" {
        Write-Host "📦 Installing dependencies..." -ForegroundColor Green
        & $PythonExe -m pip install -r requirements.txt
    }
    "health" {
        Write-Host "🏥 Testing health endpoint..." -ForegroundColor Green
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 5
            Write-Host "✅ App is healthy!" -ForegroundColor Green
            Write-Host $response.Content
        }
        catch {
            Write-Host "❌ App is not responding. Is it running?" -ForegroundColor Red
        }
    }
    "stop" {
        Write-Host "🛑 Stopping Python processes..." -ForegroundColor Red
        Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Host "✅ Stopped" -ForegroundColor Green
    }
    "help" {
        Write-Host "Available commands:" -ForegroundColor Yellow
        Write-Host "  dev     - Start development server (default)" -ForegroundColor White
        Write-Host "  prod    - Start production server" -ForegroundColor White
        Write-Host "  test    - Start minimal test server" -ForegroundColor White
        Write-Host "  install - Install Python dependencies" -ForegroundColor White
        Write-Host "  health  - Test if app is running" -ForegroundColor White
        Write-Host "  stop    - Stop all Python processes" -ForegroundColor White
        Write-Host "  help    - Show this help" -ForegroundColor White
        Write-Host ""
        Write-Host "Usage: .\dev.ps1 [command]" -ForegroundColor Gray
    }
    default {
        Write-Host "❌ Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run '.\dev.ps1 help' for available commands" -ForegroundColor Yellow
    }
}