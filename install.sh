#!/bin/bash

# Install Python
apt-get -y update
apt-get install -y python3-pip python3-venv


# Create empty virtual environment
python3 -m venv summary_generation
source summary_generation/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Install spacy model
python -m spacy download en_core_web_sm
