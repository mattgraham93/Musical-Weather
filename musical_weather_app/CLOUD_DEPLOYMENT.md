# Musical Weather - Google Cloud Run Deployment Guide

## 🚀 Quick Deployment

### Prerequisites
1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI** installed and configured
3. **Docker Desktop** (for local testing)

### One-Command Deployment
```powershell
# Deploy with mock data (no APIs required)
.\cloud-deploy.ps1 -MockMode

# Deploy with real functionality (requires API setup)
.\cloud-deploy.ps1
```

## 📋 Step-by-Step Setup

### 1. Google Cloud Setup

#### Install Google Cloud CLI
```bash
# Download from: https://cloud.google.com/sdk/docs/install
# Or use chocolatey (Windows):
choco install gcloudsdk
```

#### Authenticate and Setup
```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 2. Project Configuration

#### Option A: Mock Mode (Immediate Deployment)
```powershell
# Deploy immediately with mock data
.\cloud-deploy.ps1 -ProjectId YOUR_PROJECT_ID -MockMode
```

#### Option B: Full Functionality Setup
1. **Set up Secret Manager** (for API keys):
   ```bash
   # Create secrets for your API keys
   echo "your_spotify_client_id" | gcloud secrets create spotify-client-id --data-file=-
   echo "your_spotify_client_secret" | gcloud secrets create spotify-client-secret --data-file=-
   echo "your_mongodb_connection" | gcloud secrets create mongodb-connection --data-file=-
   ```

2. **Deploy with full functionality**:
   ```powershell
   .\cloud-deploy.ps1 -ProjectId YOUR_PROJECT_ID
   ```

### 3. Local Testing (Optional but Recommended)

#### Test Docker Build
```powershell
# Build and test locally
.\cloud-deploy.ps1 -Command test
```

#### Manual Docker Commands
```bash
# Build the image
docker build -t musical-weather .

# Test locally
docker run -p 8080:8080 -e PORT=8080 -e FORCE_MOCK_MODE=true musical-weather

# Visit: http://localhost:8080
```

## 🔧 Configuration Options

### Environment Variables

Set these in Cloud Run for customization:

| Variable | Description | Default |
|----------|-------------|---------|
| `FORCE_MOCK_MODE` | Force mock data mode | `false` |
| `GOOGLE_CLOUD_PROJECT` | Your GCP project ID | auto-detected |
| `DEBUG_LOGGING` | Enable detailed logging | `false` |

### Resource Configuration

The deployment script uses these optimized settings:
- **Memory**: 2Gi
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Max Instances**: 10
- **Min Instances**: 0 (cost-effective)
- **Concurrency**: 80 requests per instance

## 🌐 Available Endpoints

After deployment, your service will have these endpoints:

| Endpoint | Description |
|----------|-------------|
| `/` | Main application |
| `/health` | Health check |
| `/status` | Detailed status information |
| `/ping` | Simple connectivity test |
| `/version` | Version and environment info |
| `/table` | Main data visualization |

## 📊 Monitoring and Debugging

### View Logs
```powershell
# View recent logs
.\cloud-deploy.ps1 -Command logs

# Or use gcloud directly
gcloud logging read "resource.type=cloud_run_revision" --limit=50
```

### Check Service Status
```powershell
# Get service details
.\cloud-deploy.ps1 -Command status

# Or check health endpoint
curl https://YOUR_SERVICE_URL/health
```

### Performance Monitoring
- Use Google Cloud Monitoring for metrics
- Check the `/status` endpoint for service health
- Monitor resource usage in Cloud Console

## 🛠️ Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### 2. Permission Errors
```bash
# Ensure your account has these roles:
# - Cloud Run Admin
# - Service Account User
# - Secret Manager Secret Accessor (if using secrets)
```

#### 3. Container Build Failures
```powershell
# Test locally first
.\cloud-deploy.ps1 -Command build
```

#### 4. Service Not Responding
- Check logs: `.\cloud-deploy.ps1 -Command logs`
- Verify health endpoint: `https://YOUR_SERVICE_URL/health`
- Check resource limits in Cloud Console

### Debug Mode

Enable detailed logging:
```bash
gcloud run services update musical-weather \
  --set-env-vars DEBUG_LOGGING=true \
  --region us-west1
```

## 💰 Cost Optimization

### Free Tier Benefits
- Cloud Run includes 2 million requests per month free
- 360,000 GB-seconds of memory free
- No charges when not serving requests (scales to zero)

### Cost Control Settings
```bash
# Set maximum instances to control costs
gcloud run services update musical-weather \
  --max-instances 5 \
  --region us-west1
```

## 🔄 Deployment Workflow

### Development to Production
1. **Develop locally** using `dev_main.py`
2. **Test with Docker** using `.\cloud-deploy.ps1 -Command test`
3. **Deploy to staging** using mock mode first
4. **Configure APIs** and secrets
5. **Deploy to production** with full functionality

### Continuous Deployment
Consider setting up:
- GitHub Actions for automatic deployment
- Cloud Build triggers
- Environment-specific deployments (dev/staging/prod)

## 📱 Mobile and API Access

Your Cloud Run service supports:
- **Web interface** for human users
- **JSON API endpoints** for programmatic access
- **Mobile-friendly** responsive design
- **CORS enabled** for web applications

## 🎯 Next Steps

1. **Deploy in mock mode** to verify everything works
2. **Add your API keys** to Secret Manager
3. **Configure custom domain** (optional)
4. **Set up monitoring alerts**
5. **Implement CI/CD pipeline** (optional)

---

**Need help?** Check the service status at `https://YOUR_SERVICE_URL/status` or review the logs with `.\cloud-deploy.ps1 -Command logs`.