'''
    need to:
    - reach out and store the weather data
    - if first time, store the data in a database
    - create ability to get forecast data
    - store the forecast data for that day
    - observe how the weather event scores within its t-scores
    
    - create model and predict weather event
    
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

first_time = True

if first_time:
    historical_weather, condensed, forecast = weather.weather_main()
    print("Storing weather data...")
    weather.store_weather_data(historical_weather, "historical_raw")
    weather.store_weather_data(condensed, "historical_summary")
    print("Weather data stored.")
    
    print("Storing spotify data...")
    playlist_data = spotify_enrichment.get_playlist_data()
    
    season_music = spotify_enrichment.get_season_music(playlist_data)
    weather_music = spotify_enrichment.get_weather_music(playlist_data)
    
    spotify_enrichment.store_music_data(season_music, weather_music)
    print("Spotify data stored.")

def get_stored_weather():
    historical_weather = weather.get_stored_weather('weather.historical_raw', 'seattle')
    historical_summary = weather.get_stored_weather('weather.historical_summary', 'seattle')
    return historical_weather, historical_summary

def pull_forecast():
    todays_forecast = weather.get_todays_weather()
    return todays_forecast

def store_forecast(todays_forecast):
    weather.store_weather_data(todays_forecast, "forecast")
    
def get_forecast():
    todays_forecast = weather.get_stored_weather('weather.forecast', 'seattle')
    return todays_forecast

def get_todays_score(todays_forecast):
    return todays_forecast['weather_score_weighted']

def create_model(historical_weather):
    return weather.create_model(historical_weather)

def predict_weather_event(todays_forecast, weather_model):
    todays_forecast['weather_event'] = weather_model.predict(todays_forecast)
    return todays_forecast

def get_music_selection(todays_forecast, weather_event_score):
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

