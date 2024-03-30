#!/bin/bash
# Authorize Access
Set-ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
# source venv/bin/activate
.\venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt

# Set environment variables
# export FLASK_APP=app.py
# export FLASK_ENV=development

set FLASK_APP=app.py
set FLASK_ENV=development

# Run the Flask application
flask run
