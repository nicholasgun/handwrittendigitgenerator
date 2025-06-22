#!/bin/bash

# Setup script for deploying Streamlit app

# Install dependencies
pip install -r requirements.txt

# Make directories if they don't exist
mkdir -p ~/.streamlit/

# Create Streamlit config if it doesn't exist already
echo "[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml

echo "Setup complete!"
