import weather
import cityscraper
import lastfm
import pprint

def main():

    historical_weather = weather.weather_main()
    if historical_weather is not None:
        historical_weather, condensed = historical_weather
    else:
        historical_weather, condensed = weather.pd.DataFrame(), weather.pd.DataFrame()

    print(f"Top 5 weather conditions for Seattle:")
    print(condensed.head(5))
    
    # TODO: plot and normalize weather data
    # TODO: determine weather events and normal weather

    seattle_artists = cityscraper.main()
    if seattle_artists is not None:
        seattle_artists, seattle_albums = historical_weather
    else:
        seattle_artists, seattle_albums = weather.pd.DataFrame(), weather.pd.DataFrame()

    top_tags = lastfm.lastfm_get('chart.gettoptags')
    print(top_tags.status_code)
    lastfm.jprint(top_tags.json())

    top_artists = lastfm.lastfm_get('chart.gettopartists')
    print(top_artists.status_code)
    lastfm.jprint(top_artists.json())

    top_tracks = lastfm.lastfm_get('chart.gettoptracks')
    print(top_tracks.status_code)
    lastfm.jprint(top_tracks.json())

    geo_method_artists = 'geo.gettopartists'
    geo_method_tracks = 'geo.gettoptracks'
    country = 'United States'
    city = 'Seattle'

    seattle_top_artists = lastfm.lastfm_get_geo(geo_method_artists, country, city)
    print(seattle_top_artists.status_code)
    lastfm.jprint(seattle_top_artists.json())

    seattle_top_tracks = lastfm.lastfm_get_geo(geo_method_tracks, country, city)
    print(seattle_top_tracks.status_code)
    lastfm.jprint(seattle_top_tracks.json())

    # TODO: get music data or define genres and relevant weather events
    '''
    https://www.accuweather.com/en/press/63672538
    https://www.thomann.de/blog/en/how-the-weather-affects-our-music-listening/

    https://www.kaggle.com/code/varunsaikanuri/spotify-data-visualization/input?select=songs_normalize.csv

    '''
    '''
    DONE: finish html scraping for music data
    DONE: get music data from last.fm and store in mongoDB
    DONE: get genres and similar artists
    '''

    # TODO: combine weather and music data on weather event and music genre

    # TODO: develop a model to recommend music based on weather
    # TODO: validate the model

    # TODO: make model find artists and songs 1 or 2 degrees of separation from top artists
    # TODO: test the model with different weather data

    # TODO: output the recommended music
    
    # optional steps
    # TODO: output playlist to Spotify or Tidal
    # TODO: make model reach out to Spotify for top artists and songs in Seattle
    # TODO: make model reach out to Tidal for top artists and songs in Seattle
    # TODO: make model reach out to Spotify for recommendations
    # TODO: make model reach out to Tidal for recommendations
    
'''
    primary psuedocode:
    get validated weather data model
    -- is there a way to "store" the model?
    get today's weather
    --get weather_score_weighted for day
    get:
        top artists and albums in Seattle
        top genres
        similar artists
        top tags
        top artists
        top tracks
    
    get music data
    -- get scores from spotify
    -- get lyric scores from genius

'''


if __name__ == '__main__':
    main()
# Compare this snippet from musical_weather/weather.py: