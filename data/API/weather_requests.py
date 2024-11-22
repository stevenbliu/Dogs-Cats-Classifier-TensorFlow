import requests
import json
import pandas as pd
from datetime import datetime

def extract_weather_data(api_key, city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Example usage
api_key = "YOUR_API_KEY"
city = "San Francisco"
data = extract_weather_data(api_key, city)
