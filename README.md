# Email Triage System

A robust Flask web application designed for automated email classification and batch triage using a hybrid approach of Hugging Face transformer models and a reliable regex fallback engine.

## 🚀 Features
- **Hybrid Classification:** Attempts to use Hugging Face transformers (`distilbert-base-uncased-finetuned-sst-2-english`) first, with an automated fallback to a keyword/regex pattern-matching engine.
- **Batch Processing:** Supports CSV file uploads containing email subjects and bodies.
- **Structured Results:** Displays a clean, tabular output with email IDs, subjects, predicted categories (e.g., Reservation Inquiry, Pricing & Billing, Customer Support, Cancellation Request), and confidence scores.
- **Error Handling & Aliases:** Fully configured route aliases (`home`, `index`, `upload_file`) and safe data casting to prevent any runtime errors.

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Gunicorn
- **Machine Learning / AI:** PyTorch, Hugging Face Transformers (`transformers`, `pipeline`)
- **Frontend:** HTML5, Bootstrap 5, Jinja2 Templates

## 📥 Installation & Local Setup

1. **Clone or open the repository folder:**
   ```bash
   cd drive-easy-app
## 🌐 Live Demo & Deployment Instructions
## 🌐 Live Demo & Deployment

- **Live Demo URL:** `https://drive-easy-app-production.up.railway.app`

### 🚀 How to Deploy to Railway (Production)
This application is fully configured for permanent cloud hosting on Railway.
1. Log in to [Railway.app](https://railway.app/) and click **New Project** > **Deploy from GitHub repo**.
2. Select your repository. Railway will automatically detect the `requirements.txt` and install dependencies.
3. The included `Procfile` (`web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`) will automatically configure the Gunicorn WSGI server.
4. Go to **Settings** > **Networking** in your Railway project and click **Generate Domain** to get your live public URL.