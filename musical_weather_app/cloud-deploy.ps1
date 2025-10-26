# Musical Weather - Google Cloud Run Deployment Script
# This script automates the deployment process to Google Cloud Run

param(
    [string]$ProjectId = "musical-weather",
    [string]$ServiceName = "musical-weather",
    [string]$Region = "us-west1",
    [string]$Command = "deploy",
    [switch]$AllowUnauthenticated = $true,
    [switch]$MockMode = $false
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Musical Weather - Cloud Run Deployment" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

function Test-GCloudAuth {
    try {
        $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
        if ($account) {
            Write-Host "✅ Authenticated as: $account" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Not authenticated with Google Cloud" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ Google Cloud CLI not found or not working" -ForegroundColor Red
        return $false
    }
}

function Test-DockerRunning {
    try {
        docker ps >$null 2>&1
        Write-Host "✅ Docker is running" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ Docker is not running" -ForegroundColor Red
        return $false
    }
}

function Deploy-ToCloudRun {
    Write-Host "🚀 Deploying to Google Cloud Run..." -ForegroundColor Green
    Write-Host "   Project: $ProjectId" -ForegroundColor Yellow
    Write-Host "   Service: $ServiceName" -ForegroundColor Yellow
    Write-Host "   Region: $Region" -ForegroundColor Yellow
    Write-Host ""

    $deployArgs = @(
        "run", "deploy", $ServiceName,
        "--source", ".",
        "--project", $ProjectId,
        "--region", $Region,
        "--platform", "managed",
        "--port", "8080",
        "--memory", "2Gi",
        "--cpu", "1",
        "--timeout", "300",
        "--max-instances", "10",
        "--min-instances", "0",
        "--concurrency", "80"
    )

    if ($AllowUnauthenticated) {
        $deployArgs += "--allow-unauthenticated"
    }

    # Add environment variables
    $envVars = @()
    if ($MockMode) {
        $envVars += "FORCE_MOCK_MODE=true"
        Write-Host "🔧 Mock mode enabled" -ForegroundColor Yellow
    }
    
    if ($envVars.Count -gt 0) {
        $deployArgs += "--set-env-vars"
        $deployArgs += ($envVars -join ",")
    }

    Write-Host "Executing: gcloud $($deployArgs -join ' ')" -ForegroundColor Gray
    & gcloud @deployArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Deployment successful!" -ForegroundColor Green
        
        # Get the service URL
        $serviceUrl = gcloud run services describe $ServiceName --project=$ProjectId --region=$Region --format="value(status.url)" 2>$null
        if ($serviceUrl) {
            Write-Host ""
            Write-Host "🌐 Service URL: $serviceUrl" -ForegroundColor Cyan
            Write-Host "🏥 Health Check: $serviceUrl/health" -ForegroundColor Cyan
            Write-Host "📊 Status: $serviceUrl/status" -ForegroundColor Cyan
            Write-Host "🎵 Data Table: $serviceUrl/table" -ForegroundColor Cyan
        }
    } else {
        Write-Host "❌ Deployment failed!" -ForegroundColor Red
        exit 1
    }
}

function Build-LocalDocker {
    Write-Host "🐳 Building Docker image locally..." -ForegroundColor Green
    
    docker build -t $ServiceName .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker build successful!" -ForegroundColor Green
        Write-Host ""
        Write-Host "To test locally, run:" -ForegroundColor Yellow
        Write-Host "docker run -p 8080:8080 -e PORT=8080 $ServiceName" -ForegroundColor Gray
    } else {
        Write-Host "❌ Docker build failed!" -ForegroundColor Red
        exit 1
    }
}

function Test-LocalDocker {
    Write-Host "🧪 Testing Docker container locally..." -ForegroundColor Green
    
    # Stop any existing container
    docker stop $ServiceName-test 2>$null
    docker rm $ServiceName-test 2>$null
    
    # Run the container
    Write-Host "Starting container..." -ForegroundColor Yellow
    docker run -d --name $ServiceName-test -p 8081:8080 -e PORT=8080 $ServiceName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Container started successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Test URL: http://localhost:8081" -ForegroundColor Cyan
        Write-Host "🏥 Health Check: http://localhost:8081/health" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "To stop the test container:" -ForegroundColor Yellow
        Write-Host "docker stop $ServiceName-test && docker rm $ServiceName-test" -ForegroundColor Gray
        
        # Wait a moment and test the health endpoint
        Start-Sleep 5
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8081/health" -TimeoutSec 10
            Write-Host "✅ Health check passed!" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Health check failed, but container is running" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Container failed to start!" -ForegroundColor Red
        exit 1
    }
}

function Show-Logs {
    Write-Host "📋 Fetching Cloud Run logs..." -ForegroundColor Green
    gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName" --project=$ProjectId --limit=50 --format="table(timestamp,severity,textPayload)"
}

function Show-Status {
    Write-Host "📊 Service Status..." -ForegroundColor Green
    gcloud run services describe $ServiceName --project=$ProjectId --region=$Region
}

function Show-Help {
    Write-Host "Musical Weather - Cloud Run Deployment Helper" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor White
    Write-Host "  deploy       - Deploy to Google Cloud Run (default)" -ForegroundColor Gray
    Write-Host "  build        - Build Docker image locally" -ForegroundColor Gray
    Write-Host "  test         - Build and test Docker container locally" -ForegroundColor Gray
    Write-Host "  logs         - Show Cloud Run logs" -ForegroundColor Gray
    Write-Host "  status       - Show service status" -ForegroundColor Gray
    Write-Host "  help         - Show this help" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Parameters:" -ForegroundColor White
    Write-Host "  -ProjectId            Google Cloud Project ID (default: musical-weather)" -ForegroundColor Gray
    Write-Host "  -ServiceName          Cloud Run service name (default: musical-weather)" -ForegroundColor Gray
    Write-Host "  -Region               Deployment region (default: us-west1)" -ForegroundColor Gray
    Write-Host "  -AllowUnauthenticated Allow public access (default: true)" -ForegroundColor Gray
    Write-Host "  -MockMode             Deploy in mock mode (default: false)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor White
    Write-Host "  .\cloud-deploy.ps1" -ForegroundColor Gray
    Write-Host "  .\cloud-deploy.ps1 -Command build" -ForegroundColor Gray
    Write-Host "  .\cloud-deploy.ps1 -ProjectId my-project -MockMode" -ForegroundColor Gray
}

# Main execution
switch ($Command.ToLower()) {
    "deploy" {
        if (!(Test-GCloudAuth)) {
            Write-Host "Please run: gcloud auth login" -ForegroundColor Yellow
            exit 1
        }
        
        # Set the project
        gcloud config set project $ProjectId
        
        Deploy-ToCloudRun
    }
    "build" {
        if (!(Test-DockerRunning)) {
            Write-Host "Please start Docker Desktop" -ForegroundColor Yellow
            exit 1
        }
        
        Build-LocalDocker
    }
    "test" {
        if (!(Test-DockerRunning)) {
            Write-Host "Please start Docker Desktop" -ForegroundColor Yellow
            exit 1
        }
        
        Build-LocalDocker
        Test-LocalDocker
    }
    "logs" {
        if (!(Test-GCloudAuth)) {
            Write-Host "Please run: gcloud auth login" -ForegroundColor Yellow
            exit 1
        }
        
        Show-Logs
    }
    "status" {
        if (!(Test-GCloudAuth)) {
            Write-Host "Please run: gcloud auth login" -ForegroundColor Yellow
            exit 1
        }
        
        Show-Status
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "❌ Unknown command: $Command" -ForegroundColor Red
        Show-Help
        exit 1
    }
}