#!/bin/bash

# Azure App Service Startup Script for Lore Lantern
# Uses the virtual environment's Python directly

cd /home/site/wwwroot

# Set paths explicitly to use venv
export VIRTUAL_ENV="/home/site/wwwroot/antenv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="/home/site/wwwroot:$VIRTUAL_ENV/lib/python3.11/site-packages"

# Debug: Show what we're using
echo "=== Startup Debug ==="
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python location: $(which python)"
echo "Pip location: $(which pip)"
echo "Gunicorn location: $(which gunicorn)"

# List installed packages
echo "=== Installed Packages ==="
$VIRTUAL_ENV/bin/pip list 2>&1 | head -20

# Verify uvicorn
echo "=== Uvicorn Check ==="
$VIRTUAL_ENV/bin/python -c "import uvicorn; print(f'Uvicorn version: {uvicorn.__version__}')" 2>&1

# Start gunicorn using the venv's gunicorn directly
echo "=== Starting Gunicorn ==="
exec $VIRTUAL_ENV/bin/gunicorn \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    src.main:app
