import os
import re
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Hugging Face pipeline for GPT-2 text generation
print("Loading Hugging Face GPT-2 model pipeline...")
generator = pipeline("text-generation", model="gpt2")

VALID_CATEGORIES = [
    "Reservation Inquiry",
    "Cancellation Request",
    "Payment Issue",
    "Vehicle Damage Report",
    "General Feedback"
]

def classify_email_with_gpt2(email_text):
    try:
        # Prompt structure guiding the base model
        prompt = (
            f"Classify the following car rental email into one of these categories: "
            f"{VALID_CATEGORIES}.\n"
            f"Email: \"{email_text}\"\n"
            f"Output strictly in JSON format with keys 'category' and 'confidence' (float between 0 and 1).\n"
            f"JSON:"
        )
        
        # Run text generation
        response = generator(prompt, max_new_tokens=50, num_return_sequences=1, pad_token_id=50256)
        raw_output = response[0]['generated_text']
        
        # Extract JSON payload using regex
        json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            category = data.get("category", "Uncertain")
            confidence = float(data.get("confidence", 0.0))
            
            # Apply confidence fallback threshold (< 0.6) or validate category
            if confidence < 0.6 or category not in VALID_CATEGORIES:
                return "Uncertain", confidence
            
            return category, confidence
        
        # Fallback if no valid JSON structure is found in generated text
        return "Uncertain", 0.0

    except (json.JSONDecodeError, ValueError, KeyError, Exception):
        # Robustly catch any parsing, missing key, or type conversion failures without crashing
        return "Uncertain", 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Read uploaded CSV using pandas
        df = pd.read_csv(filepath)
        
        results = []
        for index, row in df.iterrows():
            # Handle different possible column names for email content
            email_content = row.get('email_text', row.get('email', str(row.values[0])))
            category, confidence = classify_email_with_gpt2(email_content)
            results.append({
                'email': email_content,
                'category': category,
                'confidence': confidence
            })
            
        return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)