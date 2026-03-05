import polars as pl
import time

import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timezone, timedelta


def get_leitregion_weather_csv(file_path:str, leitregion_number: str, longitude: pl.Float32, latitude: pl.Float32):
    """

    This method creates a csv file with weather data from the open meteo api. It writes the csv to the chosen path.
    
    NOTE: Part of this function was created with the help of chatgpt.

    in: filepath (str), leitregionen_number (str), longitude (pl.Float32), latitude (pl.Float32)
    out: csv file written in chosen directory
    """
    print(f"Sende API-Call für Region {leitregion_number}.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2014-06-08",
        "end_date": "2026-03-01",
        "hourly": ["temperature_2m", "precipitation", "weather_code"],
        "timezone": "UTC"
    }

    
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = 'https://archive-api.open-meteo.com/v1/archive'
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
    
        # Process hourly data. The order of variables needs to be the same as requested.
        # The following section is from the openmeteo api website. Originally the data was put into a pandas df. 
        # We used chatgpt to change the code to create a polars df for the data.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(2).ValuesAsNumpy()

        start = datetime.fromtimestamp(hourly.Time(), tz=timezone.utc)
        end = datetime.fromtimestamp(hourly.TimeEnd(), tz=timezone.utc)
        interval = timedelta(seconds=hourly.Interval())

        date_range = pl.datetime_range(
            start,
            end,
            interval,
            closed="left",
            eager=True
        )
        # ---- Create Polars DataFrame ----
        df = pl.DataFrame({
            "date": date_range,
            "temperature_2m": hourly_temperature_2m,
            "precipitation": hourly_precipitation,
            "weather_code": hourly_weather_code,
        })
        print(f"Erfolg! {df.height:,} Zeilen geladen.")

        path_csv = file_path + 'weather_region' + leitregion_number + '.csv'

        print(f"Path: {path_csv}")
        print(df)

        df.write_csv(path_csv)

    except Exception as e:
        print(f"An error occured during API request: {e}")

    time.sleep(10)

    