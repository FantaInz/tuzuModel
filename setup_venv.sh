#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <name>"
  exit 1
fi

VENV_NAME=$1

echo "Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"

echo "Activating virtual environment: $VENV_NAME"
source "$VENV_NAME/bin/activate"

if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt"
  pip3 install -r requirements.txt
else
  echo "requirements.txt not found, skipping dependency installation."
fi

echo "Installing Jupyter kernel for virtual environment: $VENV_NAME"
python -m ipykernel install --user --name="$VENV_NAME"

echo "Setup complete. The virtual environment '$VENV_NAME' is ready."
