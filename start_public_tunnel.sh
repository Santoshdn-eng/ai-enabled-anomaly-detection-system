#!/bin/bash
echo "🌐 Starting Instant Public Tunnel for AI Surveillance Anomaly Detection System..."
echo "Make sure backend app is running on port 8000 (python app_launcher.py or python -m backend.main)"
echo "----------------------------------------------------------------------------------"

npx -y localtunnel --port 8000
