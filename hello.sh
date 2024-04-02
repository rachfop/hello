#!/bin/bash

# Install pip
echo "Installing pip..."
apt-get update
apt-get install -y python3-pip

# Install uv using pip
echo "Installing uv using pip..."
pip install uv

# Create and activate virtual environment
echo "Creating and activating virtual environment..."
uv venv
source .venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Prompt the user for the OpenAI API key
read -p "Enter your OpenAI API key: " openai_api_key

# Set the OpenAI API key as an environment variable
export OPENAI_API_KEY=$openai_api_key

# Run the main Python script
echo "Running the main Python script..."
python3 main.py