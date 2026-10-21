# DriveEasy Rentals - Email Triage Assistant

## Live Application
- **Deployment URL:** [RAILWAY_BUILD_PAUSED_PENDING_RESUME] *(Tested & verified locally at http://127.0.0.1:5000/upload)*

## Overview
An automated email classification web application powered by Flask, Hugging Face GPT-2, and scikit-learn to streamline customer inquiry processing for car rental operations.

## Evaluation & Performance
- **Evaluation Script:** `evaluate.py`
- **Achieved Accuracy:** 0.75
- **Confidence Threshold:** 0.60 (Predictions below 0.60 are automatically routed to the "Uncertain" category for human review)

## Local Setup & Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/muhammadadeel-se/drive-easy-app.git](https://github.com/muhammadadeel-se/drive-easy-app.git)
   cd drive-easy-app