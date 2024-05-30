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
import mongodb
import pandas as pd
import numpy as np 

first_time = True

if first_time:
    historical_weather, condensed = weather.weather_main()
    print("Storing weather data...")
    weather.store_weather_data(historical_weather)
    weather.store_weather_data(condensed)
    print("Weather data stored.")
    
def get_forecast():
    pass

def store_forecast():
    pass

def get_weather_event_score():
    pass

def create_model():
    pass

def predict_weather_event():
    pass

def get_music_selection():
    pass

def get_last_fm():
    pass

def get_season():
    pass

def get_city():
    pass

