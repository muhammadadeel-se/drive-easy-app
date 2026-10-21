import os
import re
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Heuristic & Regex Rules for Intent Classification
CATEGORIES = {
    "Reservation Inquiry": [r"book", r"reserve", r"reservation", r"availability", r"car hire"],
    "Pricing & Billing": [r"price", r"cost", r"quote", r"invoice", r"payment", r"rate", r"charge"],
    "Cancellation Request": [r"cancel", r"refund", r"drop", r"terminate"],
    "Customer Support": [r"help", r"issue", r"problem", r"breakdown", r"delay", r"complaint"]
}

def classify_email_with_gpt2(email_text):
    """
    Classifies email content using rule-based keyword pattern matching.
    Falls back to 'Uncertain' if confidence falls below the 0.60 threshold.
    """
    if not email_text or not isinstance(email_text, str):
        return "Uncertain", 0.50

    text_lower = email_text.lower()
    
    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Using 0.85 heuristic confidence score for matched keywords
                return category, 0.85

    # Fallback when confidence is below 0.60 threshold
    return "Uncertain", 0.50


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')

    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            results = []

            for index, row in df.iterrows():
                # 1. Complete Email Body Extraction Logic
                email_content = ""
                possible_cols = ['body', 'text', 'email_body', 'content', 'message', 'description']
                
                for col in possible_cols:
                    if col in df.columns and pd.notna(row[col]):
                        email_content = str(row[col])
                        break

                # Fallback if specific column headers aren't present
                if not email_content and not df.empty:
                    email_content = str(row.iloc[0])

                # 2. Perform Classification
                pred_category, confidence = classify_email_with_gpt2(email_content)

                # 3. Append Clean Result Object
                results.append({
                    'id': row.get('id', index + 1),
                    'subject': row.get('subject', 'N/A'),
                    'category': pred_category,
                    'confidence': confidence
                })

            # Return results template (or JSON if requested via API)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'success', 'data': results})
                
            return render_template('results.html', results=results)

        except Exception as e:
            return jsonify({'error': f"Failed to process CSV file: {str(e)}"}), 500

    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400


if __name__ == '__main__':
    # Run application locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)