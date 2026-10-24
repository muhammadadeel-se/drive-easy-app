# Handover Note: DriveEasy Rentals - Email Triage Assistant

## 📌 Project Overview
An automated email classification web application designed to streamline customer inquiries for DriveEasy Rentals. It uses a hybrid approach, combining a Hugging Face Transformer model (`distilbert-base-uncased-finetuned-sst-2-english`) for sentiment-based urgency detection and a reliable Regex fallback engine for business category routing.

## 🛠️ Tech Stack
- **Backend Framework:** Python 3.10+, Flask
- **Machine Learning:** Hugging Face `transformers`
- **Data Handling:** Pandas
- **Production Server:** Gunicorn WSGI
- **Deployment Platform:** Railway

## 1. Local Run Instructions (Development)
To run this application on your local machine, follow these steps:
1. Clone the repository and open your terminal in the project folder.
2. *(Optional but recommended)* Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate