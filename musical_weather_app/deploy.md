# Musical Weather - Cloud Run Deployment Guide

## Prerequisites

1. **Google Cloud Authentication**: Ensure you have proper service account credentials
2. **Docker Desktop**: Install and start Docker Desktop for local testing

## Step 1: Test Locally with Flask

```bash
# Navigate to the app directory
cd musical_weather_app

# Install dependencies
pip install -r requirements.txt

# Set environment variable for local testing
set PORT=8080  # Windows
export PORT=8080  # Linux/Mac

# Run the app
python main.py
```

Visit `http://localhost:8080/health` to check if the app starts correctly.

## Step 2: Test with Docker Locally

```bash
# Build the Docker image
docker build -t musical-weather-local .

# Run the container
docker run -p 8080:8080 -e PORT=8080 musical-weather-local
```

Test the endpoints:
- `http://localhost:8080/health` - Health check
- `http://localhost:8080/ping` - Simple ping
- `http://localhost:8080/` - Main app

## Step 3: Deploy to Cloud Run

### Option A: Deploy from Source (Recommended)
```bash
gcloud run deploy musical-weather \
  --source . \
  --region us-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 10
```

### Option B: Deploy from Container Registry
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/musical-weather/musical-weather

# Deploy the image
gcloud run deploy musical-weather \
  --image gcr.io/musical-weather/musical-weather \
  --region us-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300
```

## Step 4: Verify Deployment

After deployment, test these endpoints:
- `https://your-service-url.run.app/health`
- `https://your-service-url.run.app/ping`
- `https://your-service-url.run.app/version`

## Troubleshooting

### If the app still fails:

1. **Check Logs**:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=musical-weather" --limit 50
   ```

2. **Service Account Issues**:
   - Ensure your Cloud Run service has the proper IAM roles:
     - Secret Manager Secret Accessor
     - Cloud SQL Client (if using Cloud SQL)

3. **Memory Issues**:
   - Increase memory allocation: `--memory 4Gi`

4. **Timeout Issues**:
   - Increase timeout: `--timeout 600`

## Environment Variables

Set these in Cloud Run if needed:
- `GOOGLE_CLOUD_PROJECT`: Your project ID
- `GAE_ENV`: Set to 'production' for production behavior

## Performance Optimizations

1. **Cold Start Optimization**:
   - Consider using Cloud Run minimum instances: `--min-instances 1`
   
2. **Resource Allocation**:
   - Monitor usage and adjust CPU/memory as needed
   
3. **Caching**:
   - Consider implementing Redis for caching weather/music data

## Security Considerations

1. **Authentication**: Remove `--allow-unauthenticated` for production
2. **IAM**: Use principle of least privilege for service accounts
3. **Secrets**: Ensure all API keys are in Secret Manager, not in code