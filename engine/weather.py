from datetime import timedelta
from matplotlib.dates import relativedelta
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
from engine import senitment_analysis as sa, mongodb
import sys
sys.path.insert(0, '..')

from engine import mongodb
from weather_files import weather_today as wt, weather_historical as wh
from datetime import datetime

from statsmodels.tsa.arima.model import ARIMA


# def predict_forecasted_event(todays_forecast, forecast_model):
#     # Ensure that 'date' is of datetime type
#     if 'date' in todays_forecast.columns:
#         todays_forecast['date'] = pd.to_datetime(todays_forecast['date'])

#         # Set 'date' as the index of the DataFrame
#         todays_forecast.set_index('date', inplace=True)

#     # Convert all columns to numeric data types, if possible
#     for col in todays_forecast.columns:
#         todays_forecast[col] = pd.to_numeric(todays_forecast[col], errors='coerce')

#     # Drop any columns that still have non-numeric data types
#     todays_forecast = todays_forecast.select_dtypes(include=[np.number])

#     # Fit the VAR model
#     model = VAR(todays_forecast)
#     fitted_model = model.fit()

#     # Make predictions
#     start_date = todays_forecast.index.min()
#     end_date = todays_forecast.index.max()
#     forecasted_values = fitted_model.forecast(fitted_model.y, steps=len(todays_forecast))

#     # Add the forecasted values to the DataFrame
#     todays_forecast['forecasted_event'] = forecasted_values

#     return todays_forecast

def predict_forecasted_event(todays_forecast, forecast_model):
    # Ensure that 'date' is of datetime type
    if 'date' in todays_forecast.columns:
        todays_forecast['date'] = pd.to_datetime(todays_forecast['date'])

        # Set 'date' as the index of the DataFrame
        todays_forecast.set_index('date', inplace=True)

    # Make predictions
    start_date = todays_forecast.index.min()
    end_date = todays_forecast.index.max()
    forecasted_values = forecast_model.predict(start=start_date, end=end_date)

    # Add the forecasted values to the DataFrame
    todays_forecast['forecasted_event'] = forecasted_values

    # If 'forecasted_event' is null, set it to the 'event'
    if 'event' in todays_forecast.columns:
        todays_forecast['forecasted_event'].fillna(todays_forecast['event'], inplace=True)

    return todays_forecast

def create_model(historical_weather):
    # Ensure that 'date' is of datetime type
    historical_weather['date'] = pd.to_datetime(historical_weather['date'])

    # Set 'date' as the index of the DataFrame
    historical_weather.set_index('date', inplace=True)

    # Fit the ARIMA model
    model_base = ARIMA(historical_weather['weather_score_weighted'], order=(5,1,0))
    model_fit = model_base.fit()

    return model_base, model_fit

def calculate_average_t_score(todays_forecast, historical_raw):
    # Initialize an empty DataFrame to store the t-scores
    t_scores_df = pd.DataFrame()

    # Select numerical columns only
    numerical_columns = todays_forecast.select_dtypes(include=[np.number]).columns

    for index, row in todays_forecast.iterrows():
        event = row['event']
        season = row['season']

        # Filter the DataFrame based on the event and season
        filtered_df = historical_raw[(historical_raw['event'] == event) & (historical_raw['season'] == season)]

        # If the filtered dataframe is empty or has less than two rows, get the average for the season only
        if filtered_df.empty or len(filtered_df) < 2:
            filtered_df = historical_raw[historical_raw['season'] == season]
            if filtered_df.empty or len(filtered_df) < 2:
                continue

        # Get the mean and standard deviation of all columns
        mean_df = pd.DataFrame(filtered_df.mean(numeric_only=True)).T
        std_df = pd.DataFrame(filtered_df.std(numeric_only=True)).T

        # Only keep columns that are present in mean_df and std_df
        numerical_columns = [col for col in numerical_columns if col in mean_df.columns and col in std_df.columns]

        # Now calculate the t-score
        t_scores = (row[numerical_columns] - mean_df[numerical_columns].squeeze()) / (std_df[numerical_columns].squeeze() + 1e-7)

        t_scores_df = pd.concat([t_scores_df, t_scores.to_frame().T], ignore_index=True)

    # Reset the index of todays_forecast before assigning the average t-scores
    todays_forecast = todays_forecast.reset_index(drop=True)
    todays_forecast['average_t_score'] = t_scores_df.mean(axis=1)

    return todays_forecast

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

def get_description(x, weather_codes):
    descriptions = weather_codes[weather_codes['weather_code'] == x]['description'].values
    return descriptions[0] if descriptions.size > 0 else None

def get_forecast():
    todays_forecast = wt.get_todays_weather()
    weather_codes = get_weather_codes()
    
    # Ensure 'weather_code' in both dataframes is of the same type
    todays_forecast['weather_code'] = todays_forecast['weather_code'].astype(float).astype(int)
    weather_codes['weather_code'] = weather_codes['weather_code'].astype(float).astype(int)
    
    todays_forecast['season'] = get_season(datetime.now())
    
    todays_forecast = pd.DataFrame(todays_forecast, index=[0])
    todays_forecast = todays_forecast.merge(weather_codes, on='weather_code', how='left')

    todays_forecast['weather_score'] = todays_forecast['description'].apply(get_weather_score)
    todays_forecast['event'] = todays_forecast['description'].apply(map_weather)
    todays_forecast = get_todays_score(todays_forecast)  
    
    historical_weather = pd.DataFrame(get_stored_weather('weather.historical_raw', 'seattle'))
    
    if historical_weather.empty:
        print("No historical summary data available. Setting t-score to 0...")
        todays_forecast['average_t_score'] = 0
    else:
        todays_forecast = calculate_average_t_score(todays_forecast, historical_weather)

    return todays_forecast

def get_historical_scores(historical_weather):
    # Calculate the weather score       
    historical_weather['weather_score'] = historical_weather['weather_score'].fillna(0).astype(int)

    # Replace 'weather_score' values of 0 with 1
    historical_weather['weather_score'] = historical_weather['weather_score'].replace(0, 1)

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
            historical_weather.groupby(['event', 'season']).agg(
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
    df = pd.DataFrame(data)  # Convert the list to a DataFrame
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
    elif 'Rain' in description or 'Heavy Drizzle' or 'Light Showers' in description:
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
    
    historical_weather = pd.DataFrame(historical_weather)
    historical_weather['season'] = pd.Series(historical_weather['date']).apply(lambda date: get_season(date.date()))
    # Fill NA/NaN values with a specific value, e.g., -1
    historical_weather['weather_code'] = historical_weather['weather_code'].fillna(1)
    print(f'Getting weather codes')
    weather_codes = get_weather_codes()
    weather_codes.drop(columns='image', inplace=True)
    
    # Ensure 'weather_code' in both dataframes is of the same type
    historical_weather['weather_code'] = historical_weather['weather_code'].astype(int)
    weather_codes['weather_code'] = weather_codes['weather_code'].astype(int)

    print(f'Analyzing and weighing weather data')
    joined = historical_weather.merge(weather_codes, on='weather_code', how='left')
    joined['event'] = joined['description'].apply(map_weather)
    historical_weather = get_historical_scores(joined)
    condensed = analyze_condensed_weather(joined)
    
    historical_weather = calculate_average_t_score(historical_weather, historical_weather)
    
    final_cols = historical_weather.columns
    
    # Get today's forecast
    print('Getting today\'s forecast')
    todays_forecast = get_forecast()

    # Only keep columns in final_cols that exist in todays_forecast
    final_cols = [col for col in final_cols if col in todays_forecast.columns]
    todays_forecast = todays_forecast[final_cols]

    return historical_weather, condensed, todays_forecast

# if __name__ == '__main__':
#     weather_main()


# def finalize_historical_weather(historical_weather):
#     # Select numerical columns once
#     numerical_cols = historical_weather.select_dtypes(include=[np.number]).columns
#     # Check for extreme values and handle them
#     extreme_value_threshold = 1e+10
#     if np.any(historical_weather[numerical_cols] > extreme_value_threshold):  # adjust the threshold as needed
#         # applies a log(1 + x) transformation
#         historical_weather[numerical_cols] = np.log1p(historical_weather[numerical_cols]) 

#     # Add a small constant to ensure there are no zero values
#     historical_weather[numerical_cols] += 1e-3

#     # Apply a different transformation to handle negative values
#     historical_weather[numerical_cols] = historical_weather[numerical_cols].apply(lambda x: np.where(x > 0, x, x**2))

#     # Now apply the transformation
#     pt = PowerTransformer(method='yeo-johnson')
#     try:
#         data = pt.fit_transform(historical_weather[numerical_cols])
#     except ValueError:
#         # If yeo-johnson fails, try box-cox transformation
#         pt = PowerTransformer(method='box-cox')
#         data = pt.fit_transform(historical_weather[numerical_cols])

#     data = pd.DataFrame(data, columns=numerical_cols)

#     # Standardize the data
#     sc = StandardScaler() 
#     X_std = sc.fit_transform(data) 

#     # Perform PCA
#     n_components = min(X_std.shape)
#     pca = PCA(n_components = n_components)
#     X_pca = pca.fit_transform(X_std)

#     # Get the most important features
#     n_pcs= pca.n_components_
#     most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
#     most_important_names = [numerical_cols[most_important[i]] for i in range(n_pcs)]

#     # Create final DataFrame
#     final_hist = data[most_important_names]
#     final_hist_pca = pca.transform(final_hist)
#     final_hist_pca_df = pd.DataFrame(final_hist_pca, columns=[f'PC{i}' for i in range(1, min(final_hist.shape)+1)])

#     # Add the independent variable to the DataFrame
#     final_hist_pca_df['description'] = historical_weather['description'].values
#     return final_hist_pca_df