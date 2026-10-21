# DriveEasy Rentals - Project Handover & Deployment Guide

## 1. Local Execution Instructions
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/muhammadadeel-se/drive-easy-app.git](https://github.com/muhammadadeel-se/drive-easy-app.git)
   cd drive-easy-app
   ## 4. Known Limitations
- **Hardcoded Heuristic Confidence Score:** In `app.py`, successful regex pattern matches currently assign a fixed heuristic confidence score (`0.85`) for demonstration purposes, rather than extracting true softmax probabilities directly from GPT-2 model logits.
- **CPU Inference Latency:** Running local LLM inference on multi-row CSV files on standard CPU instances may take a few seconds per batch.
- **Free-Tier Deployment Constraints:** Free-tier cloud instances (Railway/Render) may experience cold starts, request limits, or temporary platform-wide deployment pauses.