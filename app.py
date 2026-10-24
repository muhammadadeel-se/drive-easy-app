import os
import re
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. Initialize the Hugging Face DistilBERT model
print("Loading Hugging Face model...")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Reliable Regex Fallback Engine
def regex_fallback(text):
    text = str(text).lower()
    if re.search(r'\b(reserve|booking|rent|car|available)\b', text):
        return "Reservation Inquiry"
    elif re.search(r'\b(cancel|refund|change|modify)\b', text):
        return "Cancellation / Modification"
    elif re.search(r'\b(price|cost|quote|how much)\b', text):
        return "Pricing Inquiry"
    elif re.search(r'\b(issue|problem|broken|accident|complain)\b', text):
        return "Customer Support / Issue"
    else:
        return "General Inquiry"

# 3. Hybrid Classification Logic
def classify_email(text):
    try:
        # Pass first 512 characters to DistilBERT
        result = classifier(text[:512])[0]
        
        # If the model detects a strongly negative sentiment, flag as Urgent/Complaint
        if result['label'] == 'NEGATIVE' and result['score'] > 0.7:
            return "Urgent / Complaint"
        else:
            # Otherwise, use the regex engine to determine business category
            return regex_fallback(text)
   except Exception as e:
        # Fallback to regex in case of any model error
        return regex_fallback(text)

# 4. Routing Architecture
@app.route('/')
def index():
    # Redirect root URL directly to the upload page
    return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "Error: No file part in the request", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Error: No file selected for uploading", 400
            
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            try:
                # Process the CSV file for Batch Triage
                df = pd.read_csv(filepath)
                
                # Find the column containing email text (adjust 'Body' or 'Email' based on your CSV structure)
                text_column = None
                for col in ['Body', 'body', 'Email', 'email', 'Text', 'text']:
                    if col in df.columns:
                        text_column = col
                        break
                
                if not text_column:
                    text_column = df.columns[0] # Fallback to the first column if no standard name is found

                # Apply classification
                df['Predicted Category'] = df[text_column].apply(classify_email)
                
                # Convert results to dictionary for the HTML template
                results = df.to_dict(orient='records')
                return render_template('results.html', results=results)
                
            except Exception as e:
                return f"Error processing file: {str(e)}", 500
                
    # Render upload form for GET requests
    return render_template('upload.html')

# 5. Production Server Configuration (Railway)
if __name__ == '__main__':
    # Railway dynamically assigns a PORT environment variable.
    # We use os.getenv to fetch it, defaulting to 5000 for local testing.
    port = int(os.getenv("PORT", 5000))
    
    # debug=False is REQUIRED for production to prevent exposing stack traces
    app.run(host='0.0.0.0', port=port, debug=False)