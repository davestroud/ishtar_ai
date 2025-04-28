#!/bin/bash
# Script to update Pinecone credentials

if [ "$#" -lt 1 ]; then
    echo "Usage: ./update_pinecone.sh [--api-key YOUR_API_KEY] [--host YOUR_HOST_URL]"
    exit 1
fi

update_env_var() {
    local var_name=$1
    local var_value=$2
    
    if [ ! -f .env ]; then
        echo "# Created by update_pinecone.sh" > .env
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

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./update_pinecone.sh [--api-key YOUR_API_KEY] [--host YOUR_HOST_URL]"
            exit 1
            ;;
    esac
done

if [ ! -z "$API_KEY" ]; then
    update_env_var "PINECONE_API_KEY" "$API_KEY"
fi

if [ ! -z "$HOST" ]; then
    update_env_var "PINECONE_HOST" "$HOST"
fi

echo "✨ Pinecone credentials updated"
echo "To verify the connection, run: python3 test_pinecone.py"
echo "To restart the Docker containers with updated credentials, run: docker compose down && docker compose up -d" 