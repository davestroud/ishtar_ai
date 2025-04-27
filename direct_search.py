#!/usr/bin/env python3
"""
Direct search script to test weather queries
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_weather_in_location(location):
    """Get current weather for a location using a weather API"""
    try:
        # Use OpenWeatherMap API (free tier)
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY"),
            "units": "metric",  # Use metric units (Celsius)
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            return {
                "location": location,
                "temperature": f"{temp}°C",
                "condition": condition,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind_speed} m/s",
                "source": "OpenWeatherMap",
            }
        else:
            return {
                "error": f"Error fetching weather: {response.status_code}",
                "response": response.text,
            }
    except Exception as e:
        return {"error": f"Exception fetching weather: {str(e)}"}


def search_using_duckduckgo(query):
    """Search using DuckDuckGo API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json"}

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Error searching: {response.status_code}",
                "response": response.text,
            }
    except Exception as e:
        return {"error": f"Exception searching: {str(e)}"}


def main():
    """Main function"""
    # Try to get weather for Iran
    query = "What is the weather like in Iran today?"
    location = "Tehran,IR"  # Use Tehran as a representative city

    print(f"Query: {query}")
    print("\n1. Using OpenWeatherMap API:")
    weather_result = get_weather_in_location(location)
    print(json.dumps(weather_result, indent=2))

    print("\n2. Using DuckDuckGo search:")
    search_result = search_using_duckduckgo(query)
    print(json.dumps(search_result, indent=2))


if __name__ == "__main__":
    main()
