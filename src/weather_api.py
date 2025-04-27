#!/usr/bin/env python3
"""
Weather API Integration module for Ishtar AI
"""

import os
import logging
import requests
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAPI:
    """Client for Weather API interactions"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Weather API client"""
        self.api_key = api_key or os.environ.get("OPENWEATHER_API_KEY")
        if not self.api_key:
            logger.warning(
                "OpenWeather API key not set. Weather data will be unavailable."
            )
        else:
            logger.info(
                f"Weather API client initialized with key: {self.api_key[:5]}..."
            )

        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_current_weather(
        self, location: str, units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get current weather for a location

        Args:
            location: City name, state code and country code divided by comma
            units: Temperature unit (metric for Celsius, imperial for Fahrenheit)

        Returns:
            Dictionary with weather data or error information
        """
        if not self.api_key:
            return {"error": "OpenWeather API key not set"}

        try:
            url = f"{self.base_url}/weather"
            params = {"q": location, "appid": self.api_key, "units": units}

            logger.info(f"Fetching weather for: {location}")
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Extract and format relevant data
                result = {
                    "location": f"{data.get('name', location)}, {data.get('sys', {}).get('country', '')}",
                    "temperature": (
                        f"{data['main']['temp']}°C"
                        if units == "metric"
                        else f"{data['main']['temp']}°F"
                    ),
                    "feels_like": (
                        f"{data['main']['feels_like']}°C"
                        if units == "metric"
                        else f"{data['main']['feels_like']}°F"
                    ),
                    "condition": data["weather"][0]["description"],
                    "humidity": f"{data['main']['humidity']}%",
                    "wind_speed": (
                        f"{data['wind']['speed']} m/s"
                        if units == "metric"
                        else f"{data['wind']['speed']} mph"
                    ),
                    "pressure": f"{data['main']['pressure']} hPa",
                    "timestamp": data["dt"],
                    "source": "OpenWeatherMap",
                }

                logger.info(f"Successfully fetched weather for {location}")
                return result
            else:
                error_data = (
                    response.json() if response.text else {"message": "Unknown error"}
                )
                logger.error(
                    f"Error fetching weather: {response.status_code}, {error_data}"
                )
                return {
                    "error": f"Error fetching weather data: {response.status_code}",
                    "message": error_data.get("message", "Unknown error"),
                }

        except Exception as e:
            logger.error(f"Exception fetching weather: {e}")
            return {"error": f"Exception fetching weather: {str(e)}"}

    def format_weather_message(self, weather_data: Dict[str, Any]) -> str:
        """
        Format weather data into a readable message

        Args:
            weather_data: Weather data dictionary

        Returns:
            Formatted message string
        """
        if "error" in weather_data:
            return f"Could not retrieve weather information: {weather_data.get('message', weather_data['error'])}"

        return f"""
Weather in {weather_data['location']}:
Temperature: {weather_data['temperature']} (feels like {weather_data['feels_like']})
Condition: {weather_data['condition']}
Humidity: {weather_data['humidity']}
Wind: {weather_data['wind_speed']}
Pressure: {weather_data['pressure']}

Source: {weather_data['source']}
        """


if __name__ == "__main__":
    # Test the Weather API
    weather_client = WeatherAPI()

    # Test locations
    locations = ["Tehran,IR", "London,GB", "New York,US"]

    for loc in locations:
        print(f"\nTesting weather for: {loc}")
        result = weather_client.get_current_weather(loc)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(weather_client.format_weather_message(result))
