#!/bin/bash
# Script to update NewsAPI key

if [ "$#" -ne 1 ]; then
    echo "Usage: ./update_newsapi.sh YOUR_NEWSAPI_KEY"
    exit 1
fi

API_KEY=$1

update_env_var() {
    local var_name=$1
    local var_value=$2
    
    if [ ! -f .env ]; then
        echo "# Created by update_newsapi.sh" > .env
    fi
    
    if grep -q "^${var_name}=" .env; then
        # Replace existing variable
        sed -i '' "s|^${var_name}=.*|${var_name}=${var_value}|" .env
    else
        # Add new variable
        echo "${var_name}=${var_value}" >> .env
    fi
    
    echo "✅ Updated ${var_name} in .env file"
}

update_env_var "NEWSAPI_KEY" "$API_KEY"

echo "✨ NewsAPI key updated"
echo "To test fetching news, run: python utils/news_fetcher.py --page-size 5 --verbose"
echo "To restart the Docker containers with updated credentials, run: docker compose down && docker compose up -d" 