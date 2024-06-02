'''
    need to:
    - reach out and store the weather data -- done
    - if first time, store the data in a database -- done
    - create ability to get forecast data -- done
    - store the forecast data for that day -- done
    
    - observe how the weather event scores within its t-scores
    
    - create model (DONE) and predict weather event
    
    - return the results to the music algorithm
    - with the weather event, get range of scores for that event in relation to how it scores in the music algorithm (t-score)
    
    - return selection of songs within 2 standard deviations of the mean weather event score
    
    nice to haves:
    - stuff from last.fm (seattle stuff)
    - anything outside of weather events (seasons)
    - ability to select a city
'''


import weather
import spotify_enrichment
import mongodb
import pandas as pd
import numpy as np 
from datetime import datetime


def setup_first_time(is_first_time):
    first_time = is_first_time
    if first_time:
        historical_weather, condensed, forecast = weather.weather_main()
        print("Storing weather data...")
        weather.store_weather_data(historical_weather, "historical_raw")
        weather.store_weather_data(condensed, "historical_summary")
        print("Weather data stored.")
        
        print("Getting spotify data...")
        playlist_df, season_music, weather_music, playlists_without_track_info, playlists_without_data, le_event = spotify_enrichment.get_playlist_data()

        # playlist_df.to_csv("spotify_data.csv", index=False)

        print("Storing spotify data...")
        spotify_enrichment.store_music_data(season_music, weather_music)
        print("Spotify data stored.")

def pull_forecast():
    todays_forecast = weather.get_forecast()
    if 'ObjectId' in todays_forecast.columns:
        todays_forecast = todays_forecast.drop(columns=['ObjectId'])
    return todays_forecast

def get_stored_weather():
    historical_weather = pd.DataFrame(weather.get_stored_weather('weather.historical_raw', 'seattle'))
    historical_summary = pd.DataFrame(weather.get_stored_weather('weather.historical_summary', 'seattle'))
    if 'ObjectId' in historical_weather.columns:
        historical_weather = historical_weather.drop(columns=['ObjectId'])
    if 'ObjectId' in historical_summary.columns:
        historical_summary = historical_summary.drop(columns=['ObjectId'])
    return historical_weather, historical_summary

def store_forecast(todays_forecast):
    weather.store_weather_data(todays_forecast, "forecast")

def get_forecast():
    todays_forecast = weather.get_stored_weather('weather.forecast', 'seattle')
    todays_forecast = pd.DataFrame(todays_forecast)
    
    # If the DataFrame is empty, pull a new forecast and store it
    if todays_forecast.empty:
        new_forecast = pull_forecast()
        weather.store_weather_data(new_forecast, 'forecast')
        todays_forecast = new_forecast
    else:
        if 'ObjectId' in todays_forecast.columns:
            todays_forecast = todays_forecast.drop(columns=['ObjectId'])
        
        # Convert 'date' column to datetime
        todays_forecast['date'] = pd.to_datetime(todays_forecast['date'])
        
        # Check if the latest date in the forecast is old
        if todays_forecast['date'].max() < datetime.today():
            # Pull new forecast and store in the database
            new_forecast = pull_forecast()
            weather.store_weather_data(new_forecast, 'forecast')
            todays_forecast = new_forecast

    return todays_forecast

def get_todays_score(todays_forecast):
    return todays_forecast['weather_score_weighted'][0]

def create_weather_model(historical_weather):
    return weather.create_model(historical_weather)

def predict_weather_event(todays_forecast, weather_model):
    todays_forecast['weather_event'] = weather.predict_forecasted_event(weather_model, todays_forecast)
    return todays_forecast

def select_weather_songs(weather_score, range_width, weather_music):
    lower_bound = weather_score - range_width
    upper_bound = weather_score + range_width
    selected_songs = weather_music[(weather_music['score'] >= lower_bound) & (weather_music['score'] <= upper_bound)]
    selected_songs = selected_songs.drop_duplicates(subset='track_id', keep='last')
    return selected_songs

def get_music_selection(todays_forecast, weather_summary, weather_event_score):
    todays_score = todays_forecast['weather_score_weighted'][0]
    todays_season = weather_summary[0]['season']
    
    weather_music = pd.DataFrame(spotify_enrichment.get_stored_music('spotify.weather_music'))
    '''
    - get the weather event score
    - get the range of scores for that event
    - get the songs within 2 standard deviations of the mean
    - return the songs
    '''
    pass

def get_last_fm():
    pass

def get_season():
    pass

def get_city():
    pass

def main():
    # setup_first_time(False) # set to True if first time
    
    historical_weather, historical_summary = get_stored_weather()
    todays_forecast = get_forecast()
        
    model_base, model_fit = create_weather_model(historical_weather)
    
    # todays_forecast = predict_weather_event(todays_forecast, weather_model)
    
    # store_forecast(todays_forecast)
    
    todays_score = get_todays_score(todays_forecast)
    
    # get_music_selection(todays_forecast, todays_score)
    
    return todays_score, historical_summary

