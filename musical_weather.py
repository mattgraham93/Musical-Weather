'''
    need to:
    - reach out and store the weather data -- done
    - if first time, store the data in a database -- done
    - create ability to get forecast data -- done
    - store the forecast data for that day -- done
    - observe how the weather event scores within its t-scores -- done
    - create model (DONE) and predict weather event - done
    - return the results to the music algorithm - done
    - with the weather event, get range of scores for that event in relation to how it scores in the music algorithm (t-score) - done
    - return selection of songs within 2 standard deviations of the mean weather event score - done
    
    nice to haves:
    - stuff from last.fm (seattle stuff)
    - anything outside of weather events (seasons)
    - ability to select a city
'''

import weather
import spotify_enrichment
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
    return todays_forecast

def get_stored_weather():
    historical_weather = pd.DataFrame(weather.get_stored_weather('weather.historical_raw', 'seattle'))
    historical_summary = pd.DataFrame(weather.get_stored_weather('weather.historical_summary', 'seattle'))
    if 'ObjectId' in historical_weather.columns:
        historical_weather = historical_weather.drop(columns=['ObjectId'])
    if 'ObjectId' in historical_summary.columns:
        historical_summary = historical_summary.drop(columns=['ObjectId'])
    return historical_weather, historical_summary

def get_forecast(historical_weather):
    todays_forecast = weather.get_stored_weather('weather.forecast', 'seattle')
    
    # Convert todays_forecast to a DataFrame if it's a list
    if isinstance(todays_forecast, list):
        todays_forecast = pd.DataFrame(todays_forecast)

    if 'ObjectId' in todays_forecast.columns:
        todays_forecast = todays_forecast.drop(columns=['ObjectId'])

    if todays_forecast.empty:
        todays_forecast = pull_forecast()
        weather.store_weather_data(todays_forecast, "forecast")
        
    if todays_forecast['date'][0].date() != datetime.today().date():
        todays_forecast = pull_forecast()
        weather.store_weather_data(todays_forecast, "forecast")

    # Only keep columns that exist in both todays_forecast and historical_weather
    common_columns = set(todays_forecast.columns).intersection(set(historical_weather.columns))
    todays_forecast = todays_forecast[list(common_columns)]
    
    return todays_forecast

def store_forecast(todays_forecast):
    weather.store_weather_data(todays_forecast, "forecast")

def get_todays_score(todays_forecast):
    return todays_forecast['weather_score_weighted'][0]

def create_weather_model(historical_weather):
    return weather.create_model(historical_weather)

def predict_weather_event(todays_forecast, weather_model):
    return weather.predict_forecasted_event(todays_forecast, weather_model)

def select_songs(weather_score, range_width, track_df):
    lower_bound = weather_score - range_width
    upper_bound = weather_score + range_width
    selected_songs = track_df[(track_df['average_t_score'] >= lower_bound) & (track_df['average_t_score'] <= upper_bound)]
    selected_songs = selected_songs.drop_duplicates(subset='track_id', keep='last')
    return selected_songs

def get_music_selection(todays_forecast, historical_weather, todays_t_score, music_type):
    
    if music_type == 'season':
        todays_event = todays_forecast['season'].iloc[0].lower()
        matching_column = 'season'
    else:
        todays_event = todays_forecast['event'].iloc[0].lower()
        matching_column = 'event'

    # Filter historical_weather for rows where 'event' or 'season' matches todays_event
    matching_historical_weather = historical_weather[historical_weather[matching_column].str.lower() == todays_event]

    # Calculate mean and standard deviation of 'average_t_score' for the matching rows
    weather_std = matching_historical_weather['average_t_score'].std()
    
    music_data = pd.DataFrame(spotify_enrichment.get_stored_music_data(f'{music_type}_playlists'))
    music_data = spotify_enrichment.calculate_average_t_score(music_data)

    # Filter music_data for rows where 'event' or 'season' matches todays_event
    todays_music_data = music_data[music_data['event'].str.lower() == todays_event]
    
    selected_songs = select_songs(todays_t_score, weather_std, todays_music_data)
    
    return weather_std, music_data, selected_songs

def get_last_fm():
    pass

def get_season():
    pass

def get_city():
    pass

def main():
    # setup_first_time(False) # set to True if first time
    
    historical_weather, historical_summary = get_stored_weather()
    todays_forecast = get_forecast(historical_weather)
    
    todays_score = get_todays_score(todays_forecast) # for debugging    
    
    model_base, model_fit = create_weather_model(historical_weather) # use model_base for debugging
    
    todays_forecast = predict_weather_event(todays_forecast, model_fit) 
    todays_t_score = todays_forecast['average_t_score'][0]
    
    weather_std, weather_music, selected_songs_weather = get_music_selection(todays_forecast, historical_weather, todays_t_score, 'weather')
    season_std, season_music, selected_songs_season = get_music_selection(todays_forecast, historical_weather, todays_t_score, 'season')
    
    return selected_songs_weather, selected_songs_season

