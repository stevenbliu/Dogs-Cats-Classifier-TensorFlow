import requests

def fetch_weather_data(city):
    api_key = 'your_api_key'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    response = requests.get(url)
    return response.json()