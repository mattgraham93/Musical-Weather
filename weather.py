from datetime import datetime, timedelta
from matplotlib.dates import relativedelta
import matplotlib.pyplot as plt
import mongodb
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
import senitment_analysis as sa
import weather_historical as wh
import weather_today as wt
from datetime import datetime

from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder # for standardizing the Data
from sklearn.decomposition import PCA # for PCA calculation
from scipy.stats import yeojohnson
from statsmodels.tsa.arima.model import ARIMA


def create_model(historical_weather):
    model = ARIMA(historical_weather, order=(5,1,0))
    return model

def get_forecast():
    return wt.get_todays_weather()

def get_todays_score(todays_forecast):
    todays_forecast['weather_score'] = todays_forecast['weather_score'].fillna(0).astype(int)

    todays_forecast['base'] = todays_forecast['daylight_duration'] + todays_forecast['temperature_2m_max']
    todays_forecast['good'] = todays_forecast['sunshine_duration'] + todays_forecast['shortwave_radiation_sum']

    todays_forecast['bad'] = (todays_forecast['daylight_duration'] - todays_forecast['sunshine_duration']
        ) + (
            todays_forecast['rain_sum'] * todays_forecast['precipitation_hours']
            ) + (
                (todays_forecast['snowfall_sum'] * todays_forecast['precipitation_hours'])**2
                )

    todays_forecast['weight'] = todays_forecast['base'] + todays_forecast['good'] - todays_forecast['bad']

    todays_forecast['weight'] = np.where((todays_forecast['precipitation_sum'] > 0) & (todays_forecast['weight'] > 0), 
                                todays_forecast['weight'] * -1 ,
                                todays_forecast['weight']
                                ) 

    todays_forecast['weather_score_weighted'] = np.where(todays_forecast['weather_score'] < 0,
                                    (todays_forecast['weather_score'] * abs(todays_forecast['weight'])) + todays_forecast['weight'],
                                    (todays_forecast['weather_score'] * todays_forecast['weight']) + todays_forecast['weight'])

    todays_forecast.drop(columns=['base', 'good', 'bad'], inplace=True)
    return todays_forecast


def finalize_historical_weather(historical_weather):
    # Select numerical columns once
    numerical_cols = historical_weather.select_dtypes(include=[np.number]).columns

    # Perform Yeo-Johnson transformation
    y = historical_weather['weather_score_weighted']
    y, fitted_lambda = yeojohnson(y, lmbda=None)
    pt = PowerTransformer(method='yeo-johnson')
    data = pt.fit_transform(historical_weather[numerical_cols])
    data = pd.DataFrame(data, columns=numerical_cols)

    # Standardize the data
    sc = StandardScaler() 
    X_std = sc.fit_transform(data) 

    # Perform PCA
    pca = PCA(n_components = 7)
    X_pca = pca.fit_transform(X_std)

    # Get the most important features
    n_pcs= pca.n_components_
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = [numerical_cols[most_important[i]] for i in range(n_pcs)]

    # Create final DataFrame
    final_hist = data[most_important_names]
    final_hist_pca = pca.transform(final_hist)
    final_hist_pca_df = pd.DataFrame(final_hist_pca, columns=[f'PC{i}' for i in range(1, min(final_hist.shape)+1)])

    # Add the independent variable to the DataFrame
    final_hist_pca_df['description'] = historical_weather['description'].values
    return final_hist_pca_df

def get_historical_scores(historical_weather):
    # Calculate the weather score       
    historical_weather['weather_score'] = historical_weather['weather_score'].fillna(0).astype(int)

    historical_weather['base'] = historical_weather['daylight_duration'] + historical_weather['temperature_2m_mean']
    historical_weather['good'] = historical_weather['sunshine_duration'] + historical_weather['shortwave_radiation_sum']
    
    historical_weather['bad'] = (historical_weather['daylight_duration'] - historical_weather['sunshine_duration']
        ) + (
            historical_weather['rain_sum'] * historical_weather['precipitation_hours']
            ) + (
                (historical_weather['snowfall_sum'] * historical_weather['precipitation_hours'])**2
                )

    historical_weather['weight'] = historical_weather['base'] + historical_weather['good'] - historical_weather['bad']

    historical_weather['weight'] = np.where((historical_weather['precipitation_sum'] > 0) & (historical_weather['weight'] > 0), 
                                historical_weather['weight'] * -1 ,
                                historical_weather['weight']
                                ) 

    historical_weather['weather_score_weighted'] = np.where(historical_weather['weather_score'] < 0,
                                    (historical_weather['weather_score'] * abs(historical_weather['weight'])) + historical_weather['weight'],
                                    (historical_weather['weather_score'] * historical_weather['weight']) + historical_weather['weight'])

    historical_weather.drop(columns=['base', 'good', 'bad'], inplace=True)
    return historical_weather

def analyze_condensed_weather(historical_weather):
    condensed = pd.DataFrame(
            historical_weather.groupby(['description', 'season']).agg(
            {'temperature_2m_max': 'mean', 
            'temperature_2m_min': 'mean', 
            'temperature_2m_mean': 'mean',
            'precipitation_sum': 'mean', 
            'rain_sum': 'mean',
            'daylight_duration': 'mean',  
            'sunshine_duration': 'mean',
            'precipitation_hours': 'mean',
            'wind_speed_10m_max': 'mean', 
            'wind_gusts_10m_max': 'mean', 
            'shortwave_radiation_sum': 'mean',
            'snowfall_sum': 'mean',
            'weather_score': 'mean'}
            ).reset_index()
        )
    condensed = get_historical_scores(condensed)
    return condensed.sort_values('weather_score_weighted', ascending=False)

def store_weather_data(data, subject):
    df = pd.DataFrame(data)  # Convert the dictionary to a DataFrame
    mongodb.store_collection(f'weather.{subject}', 'seattle', df.to_dict('records'))
    return f'weather.{subject}'

def get_stored_weather(database_name, city):
    return mongodb.get_stored_data(database_name, city)

def get_weather_score(weather_event):
    # get the sentiment score
    return sa.get_score(weather_event)

def get_weather_codes():
    weather_codes = {}
    wc_df = pd.DataFrame()
    
    with urlopen('https://gist.githubusercontent.com/stellasphere/9490c195ed2b53c707087c8c2db4ec0c/raw/76b0cb0ef0bfd8a2ec988aa54e30ecd1b483495d/descriptions.json') as response:
        weather_codes = json.load(response)

    for key in weather_codes.keys():
        wc_df = pd.concat([wc_df, pd.DataFrame(weather_codes[key]['day'], index=[key])])
    wc_df = wc_df.reset_index()
    wc_df.columns = ['weather_code', 'description', 'image']
    wc_df['weather_score'] = wc_df['description'].apply(get_weather_score)
    return wc_df

def map_weather(description):
    if 'Sunny' in description:
        return 'Sun'
    elif 'Cloud' in description:
        return 'Cloud'
    elif 'Snow' in description:
        return 'Snow'
    elif 'Rain' in description or 'Heavy Drizzle' in description:
        return 'Rain'
    elif 'Drizzle' in description:
        return 'Drizzle'
    else:
        return 'Storm'  # If none of the above conditions are met, return 'Storm'

def get_season(date):
    now = (date.month, date.day)
    if (3, 1) <= now < (5, 31):
        season = 'spring'
    elif (6, 1) <= now < (8, 30):
        season = 'summer'
    elif (9, 1) <= now < (11, 30):
        season = 'fall'
    else:
        season = 'winter'
    return season

def weather_main():
    # Get today's date
    today = datetime.now()

    # Subtract one day
    yesterday = today - timedelta(days=1)
    start = yesterday - relativedelta(years=10)

    # Convert to string
    end = yesterday.strftime('%Y-%m-%d')
    start = start.strftime('%Y-%m-%d')

    # get and store latest data
    print('Getting weather data for Seattle')
    historical_weather = wh.get_historical_weather(start, end)
    historical_weather['season'] = pd.Series(historical_weather['date']).apply(lambda date: get_season(date.date()))

    historical_weather = pd.DataFrame(historical_weather)
    historical_weather['weather_code'] = historical_weather['weather_code'].astype(int)
    print(f'Getting weather codes')
    weather_codes = get_weather_codes()
    weather_codes['weather_code'] = weather_codes['weather_code'].astype(int)
    weather_codes.drop(columns='image', inplace=True)
    print(f'Analyzing and weighing weather data')
    joined = historical_weather.merge(weather_codes, on='weather_code', how='left')

    historical_weather = get_historical_scores(joined)
    historical_weather = finalize_historical_weather(historical_weather)
    condensed = analyze_condensed_weather(joined)

    # Get today's forecast
    print('Getting today\'s forecast')
    todays_forecast = get_forecast()
    todays_forecast['season'] = get_season(today)
    todays_forecast = pd.DataFrame(todays_forecast, index=[0])
    todays_forecast = todays_forecast.merge(weather_codes, on='weather_code', how='left')
    todays_forecast = get_todays_score(todays_forecast)
    # Only keep columns in final_cols that exist in todays_forecast
    final_cols = [col for col in final_cols if col in todays_forecast.columns]
    todays_forecast = todays_forecast[final_cols]

    return historical_weather, condensed, todays_forecast

# if __name__ == '__main__':
#     weather_main()
