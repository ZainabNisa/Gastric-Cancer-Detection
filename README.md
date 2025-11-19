📌 Explainable EfficientNet System for Gastric Cancer Detection

An AI-powered diagnostic system that classifies H&E-stained gastric histopathology images into Normal or Gastric Cancer using EfficientNet-B3, Reinhard stain normalization, and Grad-CAM explainability.
It includes a complete web interface, PDF reporting, and an AI health chatbot.

🚀 Features

✔️ EfficientNet-B3-based binary classifier

✔️ Grad-CAM visual explanation for interpretability

✔️ Reinhard stain normalization for color consistency

✔️ FastAPI backend for prediction & chatbot

✔️ React-based frontend (image upload, visualization UI)

✔️ Automated PDF report generation

✔️ Accurate model performance (96%+)

✔️ Lightweight and deployable

🏗️ System Architecture
Image Upload → Preprocessing → EfficientNet-B3 Model → Prediction  
                                   ↓  
                              Grad-CAM Heatmap  
                                   ↓  
                              PDF Report + Chatbot  

📦 Tech Stack
Frontend

React.js

HTML, CSS, JavaScript

Backend

FastAPI

TensorFlow / Keras

OpenCV

NumPy, Scikit-learn

ML/AI

EfficientNet-B3

Grad-CAM

Reinhard Stain Normalization

🧪 Performance Metrics
Metric	Score
Accuracy	96.2%
Precision	95.8%
Recall	96.4%
F1-Score	96.1%
AUC-ROC	0.98
📁 Project Structure
project/
│── frontend/
│   ├── Detection.tsx
│   ├── Chatbot.jsx
│   └── public/
│
│── backend/
│   ├── main.py
│   ├── model/
│   │   └── gastric_model.h5
│   ├── utils/
│   │   ├── stain_normalization.py
│   │   └── gradcam.py
│
│── reports/
│── README.md

🔄 Workflow
1️⃣ Upload Image

User uploads a histopathology image via the React interface.

2️⃣ Preprocessing

Reinhard stain normalization

Resize → 224×224

Scaling → [0,1]

3️⃣ Model Prediction

EfficientNet-B3 returns:

Prediction: Normal / Gastric Cancer

Confidence Score

4️⃣ Explainability (Grad-CAM)

Heatmap highlights tissue regions influencing the model’s decision.

5️⃣ PDF Report

Includes:

Original image

Grad-CAM output

Diagnostic prediction

Confidence score

6️⃣ AI Chatbot

Rule-based assistant for medical queries.

⚙️ Installation
Backend Setup
pip install -r requirements.txt
uvicorn main:app --reload

Frontend Setup
cd frontend
npm install
npm start

🧠 Model Training

Dataset: HMU-GC-HE-30K

Epochs: 30

Optimizer: Adam

Loss: Binary Cross-Entropy

Augmentation: Flip, rotation, color normalization

🛠️ Future Enhancements

Multi-class cancer grading

Integration with hospital digital pathology systems

Mobile application

SHAP/LIME explainability

Federated learning for privacy-safe training

✨ Authors

Zainab Nisa J
Dept. of Information Technology
Meenakshi Sundararajan Engineering College