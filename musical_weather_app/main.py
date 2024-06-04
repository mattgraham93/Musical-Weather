import os
from flask import Flask, jsonify, Response, g, request, render_template
import config
import json
import time
import uuid
import musical_weather

from blueprints.activities import activities

# gcloud run deploy musical-weather --region=us-west1 --source=$(pwd) --allow-unauthenticated    
# 
# Flask app creation 
# https://medium.com/google-cloud/deploy-a-python-flask-server-using-google-cloud-run-d47f728cc864

# site template courtesy of pro-dev-ph
# https://github.com/pro-dev-ph/bootstrap-responsive-web-application-template

# web app tutorial:
# https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3

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
  
  @app.route('/get_weather')
  def get_weather():
      historical_weather, historical_summary = musical_weather.get_stored_weather()
      todays_forecast = musical_weather.get_forecast(historical_weather)
      weather = {
          'historical_weather': historical_weather.to_dict(orient='records'),
          'historical_summary': historical_summary.to_dict(orient='records'),
          'todays_forecast': todays_forecast.to_dict(orient='records')
      }
      return jsonify(weather)
  
  @app.route('/get_songs')
  def get_songs():
      selected_songs_weather, selected_songs_season = musical_weather.main()
      songs = {
          'selected_songs_weather': selected_songs_weather.to_dict(orient='records'),
          'selected_songs_season': selected_songs_season.to_dict(orient='records')
      }
      return jsonify(songs)

  @app.route('/table')
  def table():
      return render_template("tables.basic-table.html")      
      
  '''
    what needs to happen:
    get and load:
      - weather data
      
    develop and load transformed data
    
    get user input
    run model
    
    return list of songs with spotify uris / links
    
    ideally: find way to save to playlist
  '''
  
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
  #    app = create_app()
  print(" Starting app...")
  app.run(host="0.0.0.0", port=5000)
