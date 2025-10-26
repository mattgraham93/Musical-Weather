# Musical Weather - Local Development Setup Guide

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Navigate to the project directory
cd D:\Backup\repos\Musical-Weather\musical_weather_app

# Create your local environment file
copy .env.example .env

# Edit .env file with your API keys (optional for basic testing)
```

### 2. Install Dependencies
```bash
# Install Python dependencies (already done)
D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 3. Run the Development Server
```bash
# Option 1: Run the development version (recommended for local dev)
D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe dev_main.py

# Option 2: Run the production version (if Google Cloud is configured)
D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe main.py
```

### 4. Test the Application
Open your browser and visit:
- **Home**: http://localhost:5000/
- **Health Check**: http://localhost:5000/health
- **Development Info**: http://localhost:5000/dev
- **Data Table**: http://localhost:5000/table
- **Simple Test**: http://localhost:5000/ping

## 🔧 Configuration Options

### Local Development Mode (No API Keys Required)
The `dev_main.py` runs in mock mode by default:
- ✅ Flask app works immediately
- ✅ Mock weather and music data
- ✅ All endpoints functional
- ✅ No external API dependencies

### Full Functionality Mode
To enable real data, you'll need API keys:

1. **Spotify API** (for music data)
   - Go to: https://developer.spotify.com/dashboard
   - Create an app
   - Add CLIENT_ID and CLIENT_SECRET to .env

2. **Google Cloud** (for data storage)
   - Set up Google Cloud project
   - Configure Secret Manager
   - Set up service account

3. **Weather API** (if using external weather service)
   - OpenWeatherMap: https://openweathermap.org/api
   - Add API key to .env

## 📁 Project Structure
```
musical_weather_app/
├── main.py              # Production version
├── dev_main.py          # Development version (recommended)
├── minimal_test.py      # Basic Flask test
├── .env.example         # Environment template
├── .env                 # Your local config (create this)
├── requirements.txt     # Python dependencies
├── Dockerfile           # For Cloud Run deployment
├── engine/              # Core business logic
├── templates/           # HTML templates
├── blueprints/          # Flask blueprints
└── static/              # Static files
```

## 🐛 Troubleshooting

### App Won't Start
1. Make sure you're in the right directory:
   ```bash
   cd D:\Backup\repos\Musical-Weather\musical_weather_app
   ```

2. Try the development version first:
   ```bash
   D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe dev_main.py
   ```

3. Check the health endpoint:
   ```bash
   # In browser: http://localhost:5000/health
   ```

### Import Errors
- The development version bypasses problematic imports
- Check `/dev` endpoint for service status
- All external services are optional in dev mode

### Port Issues
- Default port is 5000
- Change with: `set PORT=8080` (Windows) or `export PORT=8080` (Linux/Mac)
- Or edit the .env file

## 🔄 Development Workflow

1. **Start Development Server**:
   ```bash
   D:/Backup/repos/Musical-Weather/.venv/Scripts/python.exe dev_main.py
   ```

2. **Make Changes**: Edit files and they'll auto-reload

3. **Test Endpoints**:
   - Basic: http://localhost:5000/ping
   - Status: http://localhost:5000/health
   - Dev Info: http://localhost:5000/dev
   - Main App: http://localhost:5000/table

4. **View Logs**: Check the terminal for debug information

## 🚀 Moving to Production

When ready for full functionality:

1. **Configure APIs**: Add real API keys to .env
2. **Test with main.py**: `python main.py`
3. **Deploy to Cloud Run**: See deploy.md

## 📝 Environment Variables

Create a `.env` file with:
```bash
# Required for local development
DEBUG=true
DISABLE_GOOGLE_CLOUD=true
PORT=5000

# Optional - add when ready for full functionality
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GOOGLE_CLOUD_PROJECT=your_project_id
```

## 🎯 Next Steps

1. **Run dev_main.py** - Should work immediately
2. **Test all endpoints** - Verify functionality
3. **Add API keys gradually** - Enable services one by one
4. **Deploy to Cloud Run** - When ready for production

---

**Need help?** Check the `/dev` endpoint for current service status and configuration.