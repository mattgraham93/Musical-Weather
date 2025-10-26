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
import os

# Only download NLTK data if not in production (Cloud Run)
if not os.environ.get('GAE_ENV') and not os.environ.get('K_SERVICE'):
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")
else:
    print("Running in production environment, skipping NLTK downloads")

# Import matplotlib with fallback
try:
    import matplotlib
    # Use Agg backend for headless environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")

# Skip musical_weather import for testing
MUSICAL_WEATHER_AVAILABLE = False
print("Skipping musical_weather import for testing")

def create_app():
  app = Flask(__name__, static_folder='templates/assets')
  app.register_blueprint(activities, url_prefix="/api/v1/activities")

  # Error 404 handler
  @app.errorhandler(404)
  def resource_not_found(e):
    return jsonify(error=str(e)), 404
  # Error 405 handler
  @app.errorhandler(405)
  def resource_not_found(e):
    return jsonify(error=str(e)), 405
  # Error 401 handler
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
    }
    return jsonify(response_body)
  
  @app.route("/ping")
  def hello_world():
    return "pong"
  
  @app.route("/health")
  def health_check():
    status = {
        "status": "healthy",
        "musical_weather_available": MUSICAL_WEATHER_AVAILABLE,
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
        "port": os.environ.get('PORT', '5000'),
        "environment": {
            "GAE_ENV": os.environ.get('GAE_ENV'),
            "K_SERVICE": os.environ.get('K_SERVICE')
        }
    }
    return jsonify(status)

  @app.route('/table')
  def table():
    # Return mock data for testing
    mock_weather = [{"temperature": "72°F", "condition": "Sunny", "error": "Mock data - musical_weather unavailable"}]
    mock_songs = [{"title": "Test Song", "artist": "Test Artist", "error": "Mock data - song service unavailable"}]
    mock_histogram = ""
    
    return render_template("tables.basic-table.html", weather=mock_weather, songs=mock_songs, histogram=mock_histogram)
        
  @app.route("/basic_table.html")
  def basic_table():
    return render_template("tables.basic-table.html")
  
  @app.before_request
  def before_request_func():
    execution_id = uuid.uuid4()
    g.start_time = time.time()
    g.execution_id = execution_id

    print(g.execution_id, "ROUTE CALLED ", request.url)

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
  print(" Starting test app...")
  port = int(os.environ.get('PORT', 5000))
  app.run(host="0.0.0.0", port=port, debug=True)