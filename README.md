# Neural Fraud Detector v2

> **v2** of the original [neural-fraud-detector](https://github.com/codezeroexe/neural-fraud-detector).  
> Deep learning credit card fraud detection with an updated, modern dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Quick Start

### Double-Click to Run (Recommended)

1. Download release from [Releases](https://github.com/codezeroexe/neural-fraud-detector-v2/releases)
2. Extract the folder
3. Double-click `Neural Fraud Detector.command` (macOS) or `Neural Fraud Detector.bat` (Windows)
4. Done! Everything happens automatically.

The launcher will:
- Create virtual environment
- Install dependencies
- Download dataset (if not present)
- Launch dashboard in browser

### Manual Start

```bash
git clone https://github.com/codezeroexe/neural-fraud-detector-v2.git
cd neural-fraud-detector-v2
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

---

## Overview

MLP neural network detecting fraudulent transactions with **0.98 ROC-AUC**, handling extreme class imbalance (~0.5% fraud rate) using class weighting.

### Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.98 |
| PR-AUC | 0.62 |
| Recall | 90% |
| Precision | 8.7% |

### Key Differences from v1

- **New Dashboard** — Modern, tabbed interface with interactive charts
- **Light/Dark Theme** — Toggle with localStorage persistence
- **13 EDA Visualizations** — Class imbalance, time patterns, correlations
- **Real-time Prediction** — Transaction form with risk tiers
- **Training History** — Live loss/AUC curves
- **Architecture View** — Layer visualization with parameter counts

---

## Architecture

```
Input (15 features) → Dense(256) → BatchNorm → Dropout(0.3) → ReLU
                   → Dense(128) → BatchNorm → Dropout(0.3) → ReLU
                   → Dense(64)  → BatchNorm → Dropout(0.3) → ReLU
                   → Dense(1)   → Sigmoid
```

**~45K parameters** | Class-weighted training | Adam optimizer

---

## Dashboard Features

### 6 Tabs

| Tab | Content |
|-----|---------|
| **EDA Analysis** | 13 visualizations: class imbalance, amount distributions, time patterns, correlation heatmap |
| **Architecture** | Neural network diagram, layer details, input features |
| **Training** | Loss/AUC curves, epoch stats, learning rate schedule |
| **Tuning** | Hyperparameter search results, trial comparison table |
| **Evaluation** | Confusion matrix, precision/recall/F1, ROC curve |
| **Predict** | Transaction form → fraud probability + risk tier |

### Design

- **Minimalist** — Clean layout, monochromatic palette
- **Responsive** — Desktop and mobile
- **Fast** — Vanilla JS, no frameworks
- **Accessible** — Proper contrast, semantic HTML

---

## Installation

```bash
# Clone
git clone https://github.com/codezeroexe/neural-fraud-detector-v2.git
cd neural-fraud-detector-v2

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from https://www.kaggle.com/datasets/kartik2112/fraud-detection
# Place fraudTrain.csv and fraudTest.csv in project root

# Train model
python fraud_detection.py

# Start dashboard
python app.py
# Open http://127.0.0.1:5000
```

---

## Project Structure

```
v2/
├── app.py                    # Flask dashboard
├── fraud_detection.py        # Training pipeline
├── tune_model.py            # Hyperparameter tuning
├── requirements.txt
├── static/
│   └── styles.css            # Dashboard styles
├── templates/
│   └── index.html            # Dashboard HTML/JS
└── models/                   # Generated (after training)
    ├── fraud_model.keras
    └── preprocessor.pkl
```

---

## Model Usage

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model('fraud_model.keras')
artifacts = joblib.load('preprocessor.pkl')

# Preprocess and predict
X = scaler.transform(...)
prob = model.predict(X)
prediction = "FRAUD" if prob > 0.5 else "LEGITIMATE"
```

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| Backend | Flask |
| Model | TensorFlow/Keras |
| Frontend | Vanilla HTML/CSS/JS |
| Charts | Chart.js |

---

## Future Work

- Ensemble with XGBoost + Random Forest
- Threshold optimization for F1
- Transaction sequence modeling (LSTM)
- SHAP explanations

---

**Dataset**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) on Kaggle  
**Built with**: TensorFlow, scikit-learn, Flask, Chart.js