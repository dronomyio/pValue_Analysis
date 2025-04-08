#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting the Lightweight P-Value Analysis application..."
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
