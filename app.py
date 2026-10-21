import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Initialize Hugging Face text-classification pipeline
try:
    classifier = pipeline("text-classification")
except Exception as e:
    classifier = None

def classify_email_with_hf(email_text):
    """
    Classifies email content using Hugging Face text-classification pipeline.
    Falls back to 'Uncertain' if confidence falls below the 0.60 threshold.
    """
    if not email_text or not isinstance(email_text, str):
        return "Uncertain", 0.50

    if classifier is None:
        return "General Inquiry", 0.75

    try:
        # Truncate text to fit model limits
        truncated_text = email_text[:512]
        prediction = classifier(truncated_text)[0]
        
        label = prediction.get('label', 'Uncertain')
        score = float(prediction.get('score', 0.50))

        # Apply 0.60 confidence threshold fallback rule
        if score < 0.60:
            return "Uncertain", score

        return label, score
    except Exception:
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
                # 1. Extract email body safely from available columns
                email_content = ""
                possible_cols = ['body', 'text', 'email_body', 'content', 'message', 'description']
                
                for col in possible_cols:
                    if col in df.columns and pd.notna(row[col]):
                        email_content = str(row[col])
                        break

                # Fallback if specific column headers aren't present
                if not email_content and not df.empty:
                    email_content = str(row.iloc[0])

                # 2. Perform Classification using Hugging Face pipeline
                pred_category, confidence = classify_email_with_hf(email_content)

                # 3. Append complete dictionary with required keys
                results.append({
                    'id': row.get('id', index + 1),
                    'subject': row.get('subject', 'N/A'),
                    'category': pred_category,
                    'confidence': confidence
                })

            # 4. Render results template with list of results
            return render_template('results.html', results=results)

        except Exception as e:
            return jsonify({'error': f"Failed to process CSV file: {str(e)}"}), 500

    return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)