---
title: Cassava Disease Detection CNN
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.12.0
app_file: backend/main.py
pinned: false
---
"Add Space configuration"

Cassava Guard — Cassava Leaf Disease Classification

Overview

- End-to-end system for cassava leaf disease diagnosis using a ConvNeXt-based FastAI model, a FastAPI backend, and a Next.js frontend.
- Supports upload of JPG/PNG images, returns predicted disease with confidence and actionable recommendations, and records predictions in SQLite for analytics.

Quick Start

- Prerequisites
  - Python 3.10+ (3.13 tested)
  - Node.js 18+ and npm
  - Git
  - Optional: GPU with CUDA if you want accelerated training/inference

- Clone and set up
  - Fork this repository on GitHub and clone your fork:
    - git clone https://github.com/JoshAyo/Cassava-Disease-Detection-Using-CNN.git
    - cd cassava-project
  - Create and activate a Python virtual environment:
    - Windows PowerShell: python -m venv venv; .\venv\Scripts\Activate.ps1
    - Bash: python -m venv venv; source venv/bin/activate
  - Install Python dependencies:
    - pip install -r requirements.txt

- Obtain model weights
  - Option A: Train locally (requires dataset)
    - Prepare the dataset as image folders under data/:
      - data/
        - Cassava\_\_\_bacterial_blight/
        - Cassava\_\_\_brown_streak_disease/
        - Cassava\_\_\_green_mottle/
        - Cassava\_\_\_healthy/
        - Cassava\_\_\_mosaic_disease/
    - Start training:
      - python model/train.py
    - Outputs are written to model/models/:
      - model.pkl (exported learner used by the backend)
      - weights.pth, classes.json, metrics.json
  - Option B: Use pre-trained weights
    - If shared externally (e.g., release or file transfer), place files under model/models/ so that model/models/model.pkl exists.

- Run the backend (FastAPI)
  - From the project root with your venv active:
    - python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  - Verify:
    - http://localhost:8000/health → should show model_loaded: true
    - http://localhost:8000/docs → interactive API docs

- Run the frontend (Next.js)
  - cd frontend
  - npm install --legacy-peer-deps
  - If the API runs at a non-default URL:
    - PowerShell: $env:NEXT_PUBLIC_API_URL="http://localhost:8000"; npm run dev
    - Bash: NEXT_PUBLIC_API_URL="http://localhost:8000" npm run dev
  - Start the dev server:
    - npm run dev
  - Open http://localhost:3000 and ensure the header shows “API Connected” once the backend is reachable.

Testing Predictions

- From the UI
  - Go to the Classify tab and upload a JPG/PNG image (< 10 MB). You’ll see the predicted disease, confidence, severity, and suggested actions.
- With curl
  - Windows PowerShell example:
    - curl -Method Post -Uri http://localhost:8000/predict -Form @{file="C:\path\to\leaf.jpg"}
  - Bash example:
    - curl -X POST http://localhost:8000/predict -F "file=@/path/to/leaf.jpg"

Collecting Test Images

- From local farmers
  - Capture clear images of individual leaves in good lighting. Avoid busy backgrounds; fill the frame with the leaf.
  - Ensure JPG or PNG format; file size < 10 MB.
  - Create a local folder test_images/ and copy images there.
  - Renaming is optional but helps organization. Example:
    - test_images/healthy_001.jpg
    - test_images/mosaic_candidate_002.jpg
- From public sources
  - You can source cassava leaf images from public datasets or academic resources where permitted. Ensure images are legal to use and do not contain personal/identifying information.

Project Structure

- backend/
  - main.py — FastAPI app (endpoints: /health, /info, /predict, /analytics, export/clear history)
  - predictor.py — Loads FastAI model and performs inference
  - predictions.db — SQLite DB created at runtime
- model/
  - train.py — Training script (handles class imbalance, training phases, TTA eval)
  - models/ — Stores model artifacts (model.pkl, weights.pth, classes.json, metrics.json)
- frontend/
  - Next.js app (UI for classification and analytics)
- data/
  - Training data (folder-per-class images) if training is performed locally

API Reference

- GET /health
  - Returns status and whether the model is loaded.
- GET /info
  - Returns model metadata and supported formats.
- POST /predict
  - Multipart form-data with “file”: JPG/PNG image. Returns disease, confidence, and class probabilities.
- GET /analytics
  - Returns total predictions, breakdown by disease, and recent history.
- GET /predictions/export
  - Returns all stored predictions as JSON.
- DELETE /predictions/clear
  - Clears the prediction history (admin caution).

Troubleshooting

- Frontend shows “Offline”
  - Ensure the backend is running on http://localhost:8000.
  - If your backend runs elsewhere, set NEXT_PUBLIC_API_URL to its URL before npm run dev.
  - Check browser DevTools → Network for /health requests and CORS or mixed content blocks.
- “Failed to process image” from /predict
  - Ensure the file is a valid JPG/PNG image and not corrupted.
  - Try a sample image from the dataset or a freshly captured photo.
- “Model not found” during backend startup
  - Ensure model/models/model.pkl exists. Train (model/train.py) or place the pre-trained file there.
- Port conflicts
  - Change backend port: python -m uvicorn backend.main:app --port 8001
  - Frontend: set NEXT_PUBLIC_API_URL to match the backend URL.
- GPU/CPU notes
  - The requirements install compatible torch builds by default. If you need CUDA builds, consult PyTorch’s official install instructions and reinstall torch/torchvision accordingly.

Production Notes (Optional)

- Backend: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
- Frontend: npm run build; npm start
- Configure CORS origins in production and place the frontend URL in NEXT_PUBLIC_API_URL.

Undergraduate Thesis — Remaining Chapters Guide

- Assumption: Chapters 1–2 completed (e.g., Introduction and Literature Review). Below is a detailed scaffold for the remaining chapters with the information you should prepare and include.

- Chapter 3 — Methodology
  - Problem definition and scope
    - Define the classification task (five cassava disease categories including “healthy”).
    - Describe target users (farmers, extension workers) and intended deployment context.
  - Dataset and data handling
    - Source of images, class distribution, and data volume per class.
    - Data preprocessing (resizing, normalization), handling of duplicates/near-duplicates, and data cleaning steps.
    - Train/validation split strategy and rationale (stratified split; imbalance considerations).
  - Model architecture and approach
    - Transfer learning with ConvNeXt Small (or the exact backbone used).
    - Training phases: feature extraction vs. fine-tuning with discriminative learning rates.
    - Loss/optimizer: weighted loss for class imbalance; learning rate schedule; batch size; augmentations.
  - Evaluation strategy
    - Validation metrics: accuracy, per-class precision/recall/F1, confusion matrix.
    - TTA usage for evaluation and its trade-off.
    - Reproducibility controls (random seeds, environment versions).
  - Tools and libraries
    - FastAI, PyTorch, timm for modeling; FastAPI for serving; Next.js for UI; SQLite for lightweight analytics.

- Chapter 4 — System Design and Implementation
  - High-level architecture
    - Diagram and explanation of data flow: frontend → backend → model → DB → frontend analytics.
  - Backend implementation
    - Endpoints (/health, /info, /predict, /analytics, export/clear).
    - Model loading via exported FastAI learner (model.pkl).
    - Input validation (MIME types, size limits) and error handling.
    - Prediction logging to SQLite and schema explanation.
    - Security and privacy considerations (CORS, input validation, data retention).
  - Frontend implementation
    - UI features: upload flow, progress feedback, results display, analytics dashboard.
    - API client design and environment configuration (NEXT_PUBLIC_API_URL).
    - Handling online/offline states and error messaging.
  - DevOps considerations
    - Project structure, environment setup, running servers.
    - Optional packaging/deployment pathways (containerization, PaaS).

- Chapter 5 — Experiments and Results
  - Experimental setup
    - Hardware/software environment, dataset split percentages, training configuration.
    - Hyperparameters (learning rates, epochs, weight decay, augmentations).
  - Baselines and ablations
    - Baseline without class weighting vs. with weighted loss.
    - With vs. without TTA; freeze vs. fine-tune; different backbones if tested.
  - Results
    - Overall validation accuracy and error rate.
    - Per-class metrics and confusion matrix.
    - Charts from metrics.json (accuracy/loss curves) and analytics snapshots.
  - Analysis
    - Error analysis with representative misclassifications and low-confidence cases.
    - Impact of class imbalance; where the model performs strongly/poorly and why.

- Chapter 6 — Discussion
  - Practical implications for farmers and extension services.
  - Limitations and risks: dataset bias, generalization to new environments, lighting conditions.
  - Ethical and privacy concerns: consent for farmer images, data stewardship, responsible use.
  - Comparison to related work and potential improvements.

- Chapter 7 — Conclusion and Future Work
  - Summary of contributions: end-to-end system, accuracy achieved, usability.
  - Future work: larger and more diverse datasets, field trials, mobile app, on-device inference, calibration, active learning, and MLOps (CI/CD monitoring).

Artifacts and Evidence to Include

- Tables: class distribution, hyperparameters, per-class metrics.
- Figures: model architecture diagram, system architecture diagram, training curves, confusion matrix, UI screenshots.
- Listings: key API endpoints and usage examples, environment and version table.
- Data sheets: describe data sources, consent process, anonymization steps.

Reproducibility Checklist

- Record environment:
  - Python, torch, fastai, timm, FastAPI, Node, Next.js versions.
- Fix random seeds where possible (Python, NumPy, PyTorch, FastAI).
- Document exact data split and any filtering steps.
- Provide command lines used for training and serving.
- Archive metrics.json, classes.json, and final model artifacts.

Data Collection Guidance (Local Farmers)

- Obtain informed consent; explain intended use and storage policies.
- Capture leaves in good natural light, avoiding shadows and blur.
- Prefer plain backgrounds and close-up framing of individual leaves.
- Avoid personal identifiers; store images in well-labeled folders.

License and Attribution

- Ensure that any datasets and third-party images used comply with their licenses and data-use policies. Attribute appropriately in your thesis and repository.
"# Cassava-Disease-Detection-Using-CNN"