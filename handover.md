# DriveEasy Rentals - Email Triage Assistant

## Overview
An automated email classification web application powered by Flask, a Hugging Face transformer model (`distilbert-base-uncased-finetuned-sst-2-english`), and a reliable regex fallback engine to streamline customer inquiry processing for car rental operations.

## Live Application
- **Live Demo URL:** `https://device-vocally-accompany.ngrok-free.dev`
*(Note: This is a temporary live tunnel URL generated via ngrok for mentor verification purposes).*

## Deployment Instructions

### Prerequisites
- Python 3.10+
- `ngrok` installed on your local machine

### 🛠️ How to Deploy Locally with ngrok (Temporary Tunnel)
Since free-tier cloud platforms currently restrict deployments behind payment gateways or strict quotas, this project uses ngrok for live evaluation. Follow these steps:

1. Ensure the Flask app is running locally on port 5000:
   ```bash
   python app.py