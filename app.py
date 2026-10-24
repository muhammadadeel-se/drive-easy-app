import os
import csv
import io
import re
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Initialize Hugging Face pipeline safely with fallback
try:
    from transformers import pipeline
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    classifier = None
    print(f"Warning: HF pipeline could not be loaded: {e}")

def classify_email_with_hf(text):
    """
    Try HF model first; if model not loaded or returns unknown label,
    fall back to simple regex engine.
    """
    try:
        if classifier is None:
            raise ValueError("Classifier not loaded")
        result = classifier(text[:512])
        label = result[0]["label"]
        score = float(result[0]["score"])
        
        if label in ("POSITIVE", "NEGATIVE"):
            raise ValueError("Sentiment label not useful")
        return label, score
    except Exception:
        # ---- Regex & Keyword Pattern-Matching Engine ----
        patterns = {
            "Reservation Inquiry": r"\b(reserve|booking|schedule|rent|car)\b",
            "Pricing & Billing": r"\b(price|cost|billing|invoice|rate|fee)\b",
            "Cancellation Request": r"\b(cancel|termination|stop|refund)\b",
            "Customer Support": r"\b(help|support|issue|problem|assist)\b",
        }
        for cat, pat in patterns.items():
            if re.search(pat, text, re.I):
                return cat, 0.85
        return "Uncertain", 0.55

# Multi-route aliases to prevent any BuildError ('home', 'index', 'upload_file')
@app.route("/", methods=["GET"], endpoint="home")
@app.route("/index", methods=["GET"], endpoint="index")
@app.route("/upload", methods=["GET", "POST"], endpoint="upload_file")
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            stream = io.TextIOWrapper(file.stream, encoding="UTF8")
            reader = csv.DictReader(stream)
            
            results = []
            for row in reader:
                email_content = None
                for col in ["body", "content", "email_body", "message"]:
                    if col in row and row[col]:
                        email_content = row[col]
                        break
                if not email_content:
                    email_content = row.get("subject", "")
                
                category, confidence = classify_email_with_hf(email_content)
                
                try:
                    conf_float = float(confidence)
                except (ValueError, TypeError):
                    conf_float = 0.55

                results.append({
                    "id": row.get("id", "N/A"),
                    "subject": row.get("subject", "N/A"),
                    "category": category,
                    "confidence": conf_float  # Pass as clean float number
                })
            
            return render_template("results.html", results=results)
            
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)