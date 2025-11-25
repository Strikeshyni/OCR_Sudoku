#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing dependencies..."
pip install -r api/requirements.txt

echo "Starting API..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
