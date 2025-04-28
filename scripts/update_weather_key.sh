#!/bin/bash
# Script to update OpenWeather API key in .env file

if [ -z "$1" ]; then
    echo "Error: No API key provided"
    echo "Usage: ./update_weather_key.sh YOUR_OPENWEATHER_API_KEY"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    touch .env
    echo "Created new .env file"
fi

# Check if OPENWEATHER_API_KEY already exists in the file
if grep -q "^OPENWEATHER_API_KEY=" .env; then
    # Replace existing OPENWEATHER_API_KEY 
    sed -i.bak "s/^OPENWEATHER_API_KEY=.*/OPENWEATHER_API_KEY=$1/" .env
    echo "Updated OPENWEATHER_API_KEY in .env file"
else
    # Add OPENWEATHER_API_KEY to file
    echo "OPENWEATHER_API_KEY=$1" >> .env
    echo "Added OPENWEATHER_API_KEY to .env file"
fi

echo "OpenWeather API key has been set. You can now run the application with weather data enabled." 