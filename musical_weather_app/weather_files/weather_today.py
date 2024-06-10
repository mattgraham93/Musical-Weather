import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"

def get_todays_weather():
	params = {
		"latitude": 47.688,
		"longitude": -122.255,
		"daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
		"temperature_unit": "fahrenheit",
		"wind_speed_unit": "mph",
		"precipitation_unit": "inch",
		"timezone": "America/Los_Angeles",
		"forecast_days": 1
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation {response.Elevation()} m asl")
	print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_weather_code = daily.Variables(0).ValuesAsNumpy()
	daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
	daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
	daily_apparent_temperature_max = daily.Variables(3).ValuesAsNumpy()
	daily_apparent_temperature_min = daily.Variables(4).ValuesAsNumpy()
	daily_sunrise = daily.Variables(5).ValuesAsNumpy()
	daily_sunset = daily.Variables(6).ValuesAsNumpy()
	daily_daylight_duration = daily.Variables(7).ValuesAsNumpy()
	daily_sunshine_duration = daily.Variables(8).ValuesAsNumpy()
	daily_uv_index_max = daily.Variables(9).ValuesAsNumpy()
	daily_uv_index_clear_sky_max = daily.Variables(10).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(11).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(12).ValuesAsNumpy()
	daily_showers_sum = daily.Variables(13).ValuesAsNumpy()
	daily_snowfall_sum = daily.Variables(14).ValuesAsNumpy()
	daily_precipitation_hours = daily.Variables(15).ValuesAsNumpy()
	daily_precipitation_probability_max = daily.Variables(16).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(17).ValuesAsNumpy()
	daily_wind_gusts_10m_max = daily.Variables(18).ValuesAsNumpy()
	daily_wind_direction_10m_dominant = daily.Variables(19).ValuesAsNumpy()
	daily_shortwave_radiation_sum = daily.Variables(20).ValuesAsNumpy()
	daily_et0_fao_evapotranspiration = daily.Variables(21).ValuesAsNumpy()

	today_weather = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}
	today_weather["weather_code"] = daily_weather_code
	today_weather["temperature_2m_max"] = daily_temperature_2m_max
	today_weather["temperature_2m_min"] = daily_temperature_2m_min
	today_weather["apparent_temperature_max"] = daily_apparent_temperature_max
	today_weather["apparent_temperature_min"] = daily_apparent_temperature_min
	today_weather["sunrise"] = daily_sunrise
	today_weather["sunset"] = daily_sunset
	today_weather["daylight_duration"] = daily_daylight_duration
	today_weather["sunshine_duration"] = daily_sunshine_duration
	today_weather["uv_index_max"] = daily_uv_index_max
	today_weather["uv_index_clear_sky_max"] = daily_uv_index_clear_sky_max
	today_weather["precipitation_sum"] = daily_precipitation_sum
	today_weather["rain_sum"] = daily_rain_sum
	today_weather["showers_sum"] = daily_showers_sum
	today_weather["snowfall_sum"] = daily_snowfall_sum
	today_weather["precipitation_hours"] = daily_precipitation_hours
	today_weather["precipitation_probability_max"] = daily_precipitation_probability_max
	today_weather["wind_speed_10m_max"] = daily_wind_speed_10m_max
	today_weather["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
	today_weather["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
	today_weather["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
	today_weather["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

	today_weather = pd.DataFrame(data = today_weather)
	return today_weather