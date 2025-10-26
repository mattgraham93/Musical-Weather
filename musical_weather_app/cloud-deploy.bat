@echo off
REM Musical Weather - Cloud Run Deployment Helper (Batch Version)
REM For full functionality, use cloud-deploy.ps1

set PROJECT_ID=musical-weather
set SERVICE_NAME=musical-weather
set REGION=us-west1

if "%1"=="" goto deploy
if "%1"=="deploy" goto deploy
if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="logs" goto logs
if "%1"=="status" goto status
if "%1"=="help" goto help
goto help

:deploy
echo 🚀 Deploying to Google Cloud Run...
echo    Project: %PROJECT_ID%
echo    Service: %SERVICE_NAME%
echo    Region: %REGION%
echo.
gcloud run deploy %SERVICE_NAME% ^
  --source . ^
  --project %PROJECT_ID% ^
  --region %REGION% ^
  --platform managed ^
  --port 8080 ^
  --memory 2Gi ^
  --cpu 1 ^
  --timeout 300 ^
  --allow-unauthenticated

if %ERRORLEVEL% EQU 0 (
  echo ✅ Deployment successful!
  echo.
  echo Getting service URL...
  for /f %%i in ('gcloud run services describe %SERVICE_NAME% --project=%PROJECT_ID% --region=%REGION% --format="value(status.url)" 2^>nul') do set SERVICE_URL=%%i
  if defined SERVICE_URL (
    echo 🌐 Service URL: %SERVICE_URL%
    echo 🏥 Health Check: %SERVICE_URL%/health
    echo 📊 Status: %SERVICE_URL%/status
  )
) else (
  echo ❌ Deployment failed!
  exit /b 1
)
goto end

:build
echo 🐳 Building Docker image locally...
docker build -t %SERVICE_NAME% .
if %ERRORLEVEL% EQU 0 (
  echo ✅ Docker build successful!
  echo.
  echo To test locally, run:
  echo docker run -p 8080:8080 -e PORT=8080 %SERVICE_NAME%
) else (
  echo ❌ Docker build failed!
  exit /b 1
)
goto end

:test
echo 🧪 Testing Docker container locally...
docker stop %SERVICE_NAME%-test 2>nul
docker rm %SERVICE_NAME%-test 2>nul
docker build -t %SERVICE_NAME% .
if %ERRORLEVEL% EQU 0 (
  docker run -d --name %SERVICE_NAME%-test -p 8081:8080 -e PORT=8080 %SERVICE_NAME%
  if %ERRORLEVEL% EQU 0 (
    echo ✅ Container started successfully!
    echo.
    echo 🌐 Test URL: http://localhost:8081
    echo 🏥 Health Check: http://localhost:8081/health
    echo.
    echo To stop: docker stop %SERVICE_NAME%-test
  ) else (
    echo ❌ Container failed to start!
    exit /b 1
  )
) else (
  echo ❌ Docker build failed!
  exit /b 1
)
goto end

:logs
echo 📋 Fetching Cloud Run logs...
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=%SERVICE_NAME%" --project=%PROJECT_ID% --limit=30
goto end

:status
echo 📊 Service Status...
gcloud run services describe %SERVICE_NAME% --project=%PROJECT_ID% --region=%REGION%
goto end

:help
echo Musical Weather - Cloud Run Deployment Helper
echo =============================================
echo Available commands:
echo   deploy - Deploy to Google Cloud Run (default)
echo   build  - Build Docker image locally
echo   test   - Build and test Docker container locally
echo   logs   - Show Cloud Run logs
echo   status - Show service status
echo   help   - Show this help
echo.
echo Usage: cloud-deploy.bat [command]
echo.
echo For advanced options, use cloud-deploy.ps1
goto end

:end