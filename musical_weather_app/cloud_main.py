import os
from flask import Flask, jsonify, Response, g, request, render_template
import config
import json
import time
import uuid
import random

import io
import urllib
import base64

from blueprints.activities import activities

import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check environment type
IS_CLOUD_RUN = bool(os.environ.get('K_SERVICE'))
IS_GAE = bool(os.environ.get('GAE_ENV'))
IS_PRODUCTION = IS_CLOUD_RUN or IS_GAE

print(f"🌍 Environment: {'Cloud Run' if IS_CLOUD_RUN else 'GAE' if IS_GAE else 'Development'}")

# NLTK data handling for production
if IS_PRODUCTION:
    print("📦 Production environment: NLTK data should be pre-downloaded in container")
    try:
        # Test if NLTK data is available
        nltk.data.find('vader_lexicon')
        print("✅ NLTK data available")
    except LookupError:
        print("⚠️  NLTK data not found, some features may be limited")
else:
    # Development environment
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")

# Import matplotlib with fallback
try:
    import matplotlib
    # Use Agg backend for headless environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib available")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("❌ Warning: matplotlib not available")

# Import musical_weather with error handling
MUSICAL_WEATHER_AVAILABLE = False
musical_weather = None

# Check if we should try to load real services or stay in mock mode
FORCE_MOCK_MODE = os.environ.get('FORCE_MOCK_MODE', 'false').lower() == 'true'

if FORCE_MOCK_MODE:
    print("🔧 Mock mode forced via environment variable")
    MUSICAL_WEATHER_AVAILABLE = False
else:
    try:
        print("🔄 Attempting to import musical_weather...")
        import musical_weather
        MUSICAL_WEATHER_AVAILABLE = True
        print("✅ musical_weather imported successfully")
    except Exception as e:
        MUSICAL_WEATHER_AVAILABLE = False
        print(f"❌ Warning: musical_weather module not available: {e}")
        print("   App will continue with mock data")

def create_app():
    app = Flask(__name__, static_folder='templates/assets')
    
    # Register blueprints with error handling
    try:
        app.register_blueprint(activities, url_prefix="/api/v1/activities")
        print("✅ Activities blueprint registered")
    except Exception as e:
        print(f"⚠️  Warning: Could not register activities blueprint: {e}")

    # Error handlers
    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify(error=str(e)), 404
    
    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify(error=str(e)), 405
    
    @app.errorhandler(401)
    def unauthorized(error):
        return Response("API Key required.", 401)
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify(error="Internal server error", details=str(e)), 500

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        try:
            if path != "" and os.path.exists("templates/" + path + ".html"):
                return render_template(path + ".html")
            else:
                return render_template("index.html")
        except Exception as e:
            print(f"Error in catch_all route: {e}")
            return jsonify(error="Template not found", path=path), 404

    @app.route("/version", methods=["GET"], strict_slashes=False)
    def version():
        response_body = {
            "success": 1,
            "version": config.VERSION,
            "environment": "production" if IS_PRODUCTION else "development",
            "platform": "cloud-run" if IS_CLOUD_RUN else "gae" if IS_GAE else "local",
            "timestamp": int(time.time())
        }
        return jsonify(response_body)

    @app.route("/ping")
    def ping():
        return "pong"

    @app.route("/health")
    def health_check():
        status = {
            "status": "healthy",
            "environment": "production" if IS_PRODUCTION else "development",
            "platform": {
                "is_cloud_run": IS_CLOUD_RUN,
                "is_gae": IS_GAE,
                "is_production": IS_PRODUCTION
            },
            "services": {
                "musical_weather_available": MUSICAL_WEATHER_AVAILABLE,
                "matplotlib_available": MATPLOTLIB_AVAILABLE,
                "force_mock_mode": FORCE_MOCK_MODE
            },
            "config": {
                "port": os.environ.get('PORT', '8080'),
                "google_cloud_project": os.environ.get('GOOGLE_CLOUD_PROJECT', 'not-set')
            },
            "timestamp": int(time.time())
        }
        return jsonify(status)

    @app.route('/table')
    def table(force_mock=False):
        if not MUSICAL_WEATHER_AVAILABLE or force_mock:
            # Mock data mode
            mock_weather = [{
                "temperature_2m_max": 72.5,
                "condition": "Sunny",
                "season": "fall",
                "location": "Seattle, WA",
                "note": f"Mock data - Running in {'production' if IS_PRODUCTION else 'development'} mode"
            }]
            
            mock_songs = [
                {"title": "Autumn Leaves", "artist": "Miles Davis", "genre": "Jazz", "note": "Mock data"},
                {"title": "September", "artist": "Earth, Wind & Fire", "genre": "Funk", "note": "Mock data"},
                {"title": "October", "artist": "U2", "genre": "Rock", "note": "Mock data"},
                {"title": "Harvest Moon", "artist": "Neil Young", "genre": "Folk Rock", "note": "Mock data"},
                {"title": "California Gurls", "artist": "Katy Perry", "genre": "Pop", "note": "Mock data"}
            ]
            
            mock_histogram = ""
            
            if MATPLOTLIB_AVAILABLE:
                try:
                    # Create a more sophisticated mock histogram
                    import numpy as np
                    
                    # Create realistic weather score distribution
                    data = np.concatenate([
                        np.random.normal(0.2, 0.8, 60),  # Most common weather
                        np.random.normal(-1.5, 0.5, 20), # Cold weather
                        np.random.normal(1.8, 0.3, 20)   # Hot weather
                    ])
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(data, bins=25, alpha=0.7, color='skyblue', edgecolor='navy')
                    plt.axvline(0.2, color='red', linestyle='dashed', linewidth=2, label='Today')
                    plt.title('Weather Score Distribution (Mock Data)')
                    plt.xlabel('Weather Score')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    png_image = io.BytesIO()
                    plt.savefig(png_image, format='png', dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    mock_histogram = "data:image/png;base64,"
                    mock_histogram += urllib.parse.quote(base64.b64encode(png_image.getvalue()))
                    print("✅ Mock histogram generated successfully")
                except Exception as e:
                    print(f"⚠️  Error creating mock histogram: {e}")
            
            return render_template("tables.basic-table.html", 
                                 weather=mock_weather, 
                                 songs=mock_songs, 
                                 histogram=mock_histogram)
        
        # Real data mode (if musical_weather is available)
        try:
            historical_weather, historical_summary = musical_weather.get_stored_weather()
            todays_forecast = musical_weather.get_forecast(historical_weather)
            
            todays_forecast['temperature_2m_max'] = todays_forecast['temperature_2m_max'].astype(float).round(1)
            
            weather_data = {
                'historical_weather': historical_weather.to_dict(orient='records'),
                'historical_summary': historical_summary.to_dict(orient='records'),
                'todays_forecast': todays_forecast.to_dict(orient='records')
            }
            
            # Create histogram if matplotlib is available
            png_image_b64_string = ""
            if MATPLOTLIB_AVAILABLE:
                try:
                    todays_season = todays_forecast['season'].iloc[0]
                    filtered_historical_weather = historical_weather[historical_weather['season'] == todays_season]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(filtered_historical_weather['average_t_score'], bins=30, alpha=0.5, label='Historical')
                    plt.axvline(todays_forecast['average_t_score'].iloc[0], color='r', linestyle='dashed', linewidth=2, label='Today')
                    
                    plt.xlabel('Average T Score')
                    plt.ylabel('Frequency')
                    plt.legend()
                    plt.title(f'Weather Score Distribution - {todays_season.title()} Season')
                    plt.grid(True, alpha=0.3)
                    
                    png_image = io.BytesIO()
                    plt.savefig(png_image, format='png', dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    png_image_b64_string = "data:image/png;base64,"
                    png_image_b64_string += urllib.parse.quote(base64.b64encode(png_image.getvalue()))
                except Exception as e:
                    print(f"Error creating histogram: {e}")

            # Get songs data
            try:
                selected_songs_weather, selected_songs_season = musical_weather.main()
                songs_data = {
                    'selected_songs_weather': selected_songs_weather.to_dict(orient='records'),
                    'selected_songs_season': selected_songs_season.to_dict(orient='records')
                }
                
                all_songs = [song for songs_list in songs_data.values() for song in songs_list]
                if len(all_songs) > 50:
                    all_songs = random.sample(all_songs, 50)
            except Exception as e:
                print(f"Error getting songs data: {e}")
                all_songs = [{"error": f"Song service error: {str(e)}"}]

            return render_template("tables.basic-table.html", 
                                 weather=weather_data["todays_forecast"], 
                                 songs=all_songs, 
                                 histogram=png_image_b64_string)
            
        except Exception as e:
            print(f"Error in table route: {e}")
            print("Falling back to mock data due to error in real data functions")
            # Force fallback to mock data by calling with force_mock=True
            return table(force_mock=True)

    @app.route("/basic_table.html")
    def basic_table():
        try:
            return render_template("tables.basic-table.html")
        except Exception as e:
            return jsonify(error="Template not found", details=str(e)), 404

    @app.route("/status")
    def detailed_status():
        """Detailed status endpoint for monitoring and debugging"""
        status = {
            "app": {
                "name": "Musical Weather",
                "version": config.VERSION,
                "status": "running"
            },
            "environment": {
                "type": "production" if IS_PRODUCTION else "development",
                "platform": "cloud-run" if IS_CLOUD_RUN else "gae" if IS_GAE else "local",
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            },
            "services": {
                "musical_weather": {
                    "available": MUSICAL_WEATHER_AVAILABLE,
                    "status": "enabled" if MUSICAL_WEATHER_AVAILABLE else "mock_mode"
                },
                "matplotlib": {
                    "available": MATPLOTLIB_AVAILABLE,
                    "status": "enabled" if MATPLOTLIB_AVAILABLE else "disabled"
                }
            },
            "configuration": {
                "port": os.environ.get('PORT', '8080'),
                "google_cloud_project": os.environ.get('GOOGLE_CLOUD_PROJECT'),
                "force_mock_mode": FORCE_MOCK_MODE
            },
            "endpoints": [
                "/", "/ping", "/health", "/status", "/version", "/table"
            ],
            "timestamp": int(time.time())
        }
        return jsonify(status)

    @app.before_request
    def before_request_func():
        execution_id = uuid.uuid4()
        g.start_time = time.time()
        g.execution_id = execution_id
        
        # Only log in development or if debug is enabled
        if not IS_PRODUCTION or os.environ.get('DEBUG_LOGGING'):
            print(f"{g.execution_id} ROUTE CALLED {request.url}")

    @app.after_request
    def after_request(response):
        try:
            if response and response.get_json():
                data = response.get_json()
                data["time_request"] = int(time.time())
                data["version"] = config.VERSION
                response.set_data(json.dumps(data))
        except Exception:
            # Don't break the response if there's an issue with after_request
            pass
        return response

    return app

app = create_app()

if __name__ == "__main__":
    print("🚀 Starting Musical Weather App")
    print("=" * 50)
    print(f"📍 Environment: {'Production' if IS_PRODUCTION else 'Development'}")
    print(f"🏗️  Platform: {'Cloud Run' if IS_CLOUD_RUN else 'GAE' if IS_GAE else 'Local'}")
    print(f"🎵 Musical Weather: {'Available' if MUSICAL_WEATHER_AVAILABLE else 'Mock Mode'}")
    print(f"📊 Matplotlib: {'Available' if MATPLOTLIB_AVAILABLE else 'Not Available'}")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 8080))
    debug = not IS_PRODUCTION and os.environ.get('DEBUG', 'false').lower() == 'true'
    
    app.run(host="0.0.0.0", port=port, debug=debug)