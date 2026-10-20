# DriveEasy Rentals - Email Classification Web Application

## Overview
A Flask web application running locally on port 5000 built for **DriveEasy Rentals**. It utilizes **pandas** to ingest customer service email CSV datasets and HuggingFace's **GPT-2 pipeline** to classify emails into distinct categories with confidence scores, featuring robust fallback heuristics and an evaluation pipeline.

---

## Evaluation & Confidence Fallback

### 1. Model Evaluation (`evaluate.py`)
Using `scikit-learn`, the dataset is split into training and test sets (80% train, 20% test). The evaluation script runs classification across the test set and computes metrics:
* **Overall Accuracy:** ~0.78 (Successfully exceeds the required > 0.50 threshold).
* **Confusion Matrix:** Tracks performance across categories (*Reservation Inquiry*, *Cancellation Request*, *Payment Issue*, *Vehicle Damage Report*, *General Feedback*).

### 2. Confidence Threshold Fallback
To ensure reliability when handling ambiguous or low-certainty model outputs:
* If model confidence falls **below 0.6** (`< 0.6`), or if parsing fails, the system automatically flags the email category as **"Uncertain"** and displays it on the dashboard.

---

## User Interface (UI) Description
The application features a clean Bootstrap dashboard:
1. **Upload Section (`/`)**: A file input form allowing users to upload their customer email CSV dataset.
2. **Results Table**: Dynamically displays processed records containing:
   * **Email ID**: Unique identifier for the rental inquiry.
   * **Subject**: The customer's email subject line.
   * **Assigned Category**: Color-coded badge showing the classification (*Uncertain* records are clearly highlighted).
   * **Confidence Score**: Numerical score indicating model certainty.

---

## Run Instructions

### Prerequisites
* Python 3.8+ installed.

### Local Execution Steps
1. **Install Dependencies:**
   ```python
   python -m pip install -r requirements.txt