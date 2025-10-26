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

# Load environment variables for local development
load_dotenv()

# Only download NLTK data if not in production (Cloud Run)
if not os.environ.get('GAE_ENV') and not os.environ.get('K_SERVICE'):
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
else:
    print("Running in production environment, skipping NLTK downloads")

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

# Import musical_weather with error handling for local development
MUSICAL_WEATHER_AVAILABLE = False
musical_weather = None

# Check if Google Cloud services should be disabled for local dev
DISABLE_GOOGLE_CLOUD = os.environ.get('DISABLE_GOOGLE_CLOUD', 'false').lower() == 'true'

if DISABLE_GOOGLE_CLOUD:
    print("🔧 Local development mode: Google Cloud services disabled")
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
        print("   App will continue with limited functionality")

def create_app():
    app = Flask(__name__, static_folder='templates/assets')
    app.register_blueprint(activities, url_prefix="/api/v1/activities")

    # Error handlers
    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify(error=str(e)), 404
    
    @app.errorhandler(405)
    def resource_not_found(e):
        return jsonify(error=str(e)), 405
    
    @app.errorhandler(401)
    def custom_401(error):
        return Response("API Key required.", 401)

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        if path != "" and os.path.exists("templates/" + path + ".html"):
            return render_template(path + ".html")
        else:
            return render_template("index.html")

    @app.route("/version", methods=["GET"], strict_slashes=False)
    def version():
        response_body = {
            "success": 1,
            "version": config.VERSION,
            "environment": "development" if os.environ.get('DEBUG') else "production"
        }
        return jsonify(response_body)

    @app.route("/ping")
    def hello_world():
        return "pong"

    @app.route("/health")
    def health_check():
        status = {
            "status": "healthy",
            "environment": "development" if os.environ.get('DEBUG') else "production",
            "services": {
                "musical_weather_available": MUSICAL_WEATHER_AVAILABLE,
                "matplotlib_available": MATPLOTLIB_AVAILABLE,
                "google_cloud_disabled": DISABLE_GOOGLE_CLOUD
            },
            "config": {
                "port": os.environ.get('PORT', '5000'),
                "debug": os.environ.get('DEBUG', 'False'),
                "google_cloud_project": os.environ.get('GOOGLE_CLOUD_PROJECT', 'not-set')
            }
        }
        return jsonify(status)

    @app.route('/table')
    def table():
        if not MUSICAL_WEATHER_AVAILABLE:
            # Return mock data for local development
            mock_weather = [{
                "temperature_2m_max": 72.5,
                "condition": "Sunny",
                "season": "fall",
                "note": "Mock data - enable musical_weather for real data"
            }]
            mock_songs = [
                {"title": "Autumn Leaves", "artist": "Miles Davis", "note": "Mock data"},
                {"title": "September", "artist": "Earth, Wind & Fire", "note": "Mock data"},
                {"title": "October", "artist": "U2", "note": "Mock data"}
            ]
            mock_histogram = ""
            
            if MATPLOTLIB_AVAILABLE:
                try:
                    # Create a simple mock histogram
                    import numpy as np
                    data = np.random.normal(0, 1, 100)
                    plt.figure(figsize=(8, 6))
                    plt.hist(data, bins=20, alpha=0.7, color='skyblue')
                    plt.title('Mock Weather Data Distribution')
                    plt.xlabel('Weather Score')
                    plt.ylabel('Frequency')
                    
                    png_image = io.BytesIO()
                    plt.savefig(png_image, format='png')
                    plt.close()
                    
                    mock_histogram = "data:image/png;base64,"
                    mock_histogram += urllib.parse.quote(base64.b64encode(png_image.getvalue()))
                except Exception as e:
                    print(f"Error creating mock histogram: {e}")
            
            return render_template("tables.basic-table.html", 
                                 weather=mock_weather, 
                                 songs=mock_songs, 
                                 histogram=mock_histogram)
        
        try:
            # Get weather data (if musical_weather is available)
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
                    
                    plt.hist(filtered_historical_weather['average_t_score'], bins=30, alpha=0.5, label='Historical')
                    plt.axvline(todays_forecast['average_t_score'].iloc[0], color='r', linestyle='dashed', linewidth=2, label='Today')
                    
                    plt.xlabel('Average T Score')
                    plt.ylabel('Frequency')
                    plt.legend()
                    
                    png_image = io.BytesIO()
                    plt.savefig(png_image, format='png')
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
            return render_template("tables.basic-table.html", 
                                 weather=[{"error": f"Weather service error: {str(e)}"}], 
                                 songs=[{"error": "Song service unavailable"}], 
                                 histogram="")

    @app.route("/basic_table.html")
    def basic_table():
        return render_template("tables.basic-table.html")

    @app.route("/dev")
    def dev_info():
        """Development information endpoint"""
        info = {
            "environment": "development",
            "services_status": {
                "musical_weather": MUSICAL_WEATHER_AVAILABLE,
                "matplotlib": MATPLOTLIB_AVAILABLE,
                "google_cloud_disabled": DISABLE_GOOGLE_CLOUD
            },
            "environment_variables": {
                "GOOGLE_CLOUD_PROJECT": os.environ.get('GOOGLE_CLOUD_PROJECT'),
                "DEBUG": os.environ.get('DEBUG'),
                "PORT": os.environ.get('PORT'),
                "DISABLE_GOOGLE_CLOUD": os.environ.get('DISABLE_GOOGLE_CLOUD')
            },
            "endpoints": [
                "/", "/ping", "/health", "/version", "/table", "/dev"
            ]
        }
        return jsonify(info)

    @app.before_request
    def before_request_func():
        execution_id = uuid.uuid4()
        g.start_time = time.time()
        g.execution_id = execution_id
        
        if os.environ.get('DEBUG'):
            print(f"{g.execution_id} ROUTE CALLED {request.url}")

    @app.after_request
    def after_request(response):
        if response and response.get_json():
            data = response.get_json()
            data["time_request"] = int(time.time())
            data["version"] = config.VERSION
            response.set_data(json.dumps(data))
        return response

    return app

app = create_app()

if __name__ == "__main__":
    print("🚀 Starting Musical Weather App (Local Development)")
    print("=" * 50)
    print(f"📍 Environment: {'Development' if os.environ.get('DEBUG') else 'Production'}")
    print(f"🔧 Google Cloud: {'Disabled' if DISABLE_GOOGLE_CLOUD else 'Enabled'}")
    print(f"🎵 Musical Weather: {'Available' if MUSICAL_WEATHER_AVAILABLE else 'Mock Mode'}")
    print(f"📊 Matplotlib: {'Available' if MATPLOTLIB_AVAILABLE else 'Not Available'}")
    print("=" * 50)
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    app.run(host="0.0.0.0", port=port, debug=debug)