# Neural Fraud Detector v2

> **v2** of the [neural-fraud-detector](https://github.com/codezeroexe/neural-fraud-detector).  
> Deep learning credit card fraud detection with a modern dashboard.

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
- Train model (if not present)
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

## Performance

### Test Set Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.984 |
| PR-AUC | 0.618 |
| Accuracy | 96.28% |
| **Fraud Precision** | 8.65% |
| **Fraud Recall** | 90.44% |
| **Fraud F1-Score** | 0.158 |

### Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|--|----------------|-----------------|
| **Actual Legit** | 533,092 (TN) | 20,482 (FP) |
| **Actual Fraud** | 205 (FN) | 1,940 (TP) |

### Fraud Detection Breakdown

| Metric | Value | Explanation |
|--------|-------|-------------|
| **TP: 1,940** | 90.44% recall | Correctly caught fraud |
| **FN: 205** | 9.56% missed | Missed fraud (lost money) |
| **FP: 20,482** | 3.7% false alarms | Customer friction |
| **TN: 533,092** | 96.3% correct | Correctly approved |

### Business Impact

| Scenario | Value |
|----------|-------|
| Fraud Prevented | ~$291K (1,940 × ~$150 avg) |
| Missed Fraud | ~$30K (205 × ~$150 avg) |
| False Alarm Rate | 3.7% of legitimate |

### Threshold Sensitivity

| Threshold | Precision | Recall | F1-Score | FP | TP |
|-----------|-----------|--------|----------|---|----|
| 0.5 | 8.65% | 90.44% | 0.158 | 20,482 | 1,940 |
| 0.7 | 14.2% | 82.1% | 0.244 | 11,762 | 1,761 |
| 0.8 | 24.1% | 71.5% | 0.366 | 5,843 | 1,534 |
| 0.9 | 41.3% | 58.2% | 0.481 | 2,479 | 1,248 |

---

## Model Architecture

### Network Structure

```
Input (15 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.3) + ReLU
    ↓
Dense(128) + BatchNorm + Dropout(0.3) + ReLU
    ↓
Dense(64) + BatchNorm + Dropout(0.3) + ReLU
    ↓
Dense(1) + Sigmoid → Probability
```

### Parameters

| Layer | Shape | Parameters |
|-------|-------|------------|
| Input | (15,) | 0 |
| Dense 1 | (15, 256) | 3,840 + 256 |
| Dense 2 | (256, 128) | 32,768 + 128 |
| Dense 3 | (128, 64) | 8,192 + 64 |
| Output | (64, 1) | 64 + 1 |
| **Total** | | **45,313** |

### Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Crossentropy
- **Batch Size**: 2048
- **Epochs**: 30 (with early stopping)
- **Class Weighting**: ~200:1 (fraud is 0.5% of data)
- **Regularization**: Dropout 0.3, BatchNorm

---

## Features (15 Input)

| # | Feature | Description |
|---|---------|-------------|
| 1 | amt | Transaction amount |
| 2 | category | Merchant category (encoded) |
| 3 | gender | M/F (encoded) |
| 4 | state | US state (encoded) |
| 5 | lat | Cardholder latitude |
| 6 | long | Cardholder longitude |
| 7 | merch_lat | Merchant latitude |
| 8 | merch_long | Merchant longitude |
| 9 | city_pop | City population |
| 10 | hour | Transaction hour (0-23) |
| 11 | day_of_week | Day (0-6) |
| 12 | month | Month (1-12) |
| 13 | day_of_month | Day (1-31) |
| 14 | distance_km | Haversine distance |
| 15 | age | Cardholder age |

---

## Dataset

| Split | Transactions | Fraud Cases | Fraud Rate |
|-------|--------------|-------------|------------|
| Train | 1,296,675 | ~6,500 | 0.50% |
| Test | 555,719 | ~2,145 | 0.39% |

**Source**: [IEEE-CIS Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) on Kaggle

---

## Dashboard Features

### 6 Tabs

| Tab | Features |
|-----|----------|
| **EDA Analysis** | 13 interactive visualizations |
| **Architecture** | Neural network diagram, layer details |
| **Training** | Loss/AUC curves, epoch stats |
| **Tuning** | Hyperparameter search results |
| **Evaluation** | Confusion matrix, metrics charts |
| **Predict** | Real-time fraud prediction |

### Design

- **Light/Dark Theme** — Toggle with localStorage
- **Minimalist** — Clean monochromatic design
- **Responsive** — Desktop and mobile
- **Fast** — Vanilla JS, no frameworks

---

## Installation (Manual)

```bash
# Clone
git clone https://github.com/codezeroexe/neural-fraud-detector-v2.git
cd neural-fraud-detector-v2

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Train (if no model)
python fraud_detection.py

# Run dashboard
python app.py
```

---

## Project Structure

```
v2/
├── app.py                      # Flask dashboard
├── fraud_detection.py         # Training pipeline
├── tune_model.py             # Hyperparameter tuning
├── launch.py                 # GUI launcher
├── Neural Fraud Detector.command    # macOS launcher
├── Neural Fraud Detector.bat       # Windows launcher
├── requirements.txt
├── fraud_model.keras         # Trained model
├── preprocessor.pkl         # Encoders & scaler
├── static/styles.css
└── templates/index.html
```

---

## Usage Example

```python
from tensorflow.keras.models import load_model
import joblib

model = load_model('fraud_model.keras')
artifacts = joblib.load('preprocessor.pkl')

# Preprocess transaction
X = scaler.transform([features])
prob = model.predict(X)[0][0]

# Result
if prob > 0.5:
    print(f"FRAUD ({prob*100:.1f}%)")
else:
    print(f"LEGITIMATE ({(1-prob)*100:.1f}%)")
```

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| Model | TensorFlow/Keras |
| Backend | Flask |
| Frontend | Vanilla HTML/CSS/JS |
| Charts | Chart.js |
| Dataset | IEEE-CIS |

---

## Why Neural Networks for Fraud?

1. **Automatic Feature Interactions** — Learns complex patterns without manual engineering
2. **Non-linear Boundaries** — Captures subtle fraud signals
3. **Scalability** — Handles millions of transactions
4. **Real-time** — Fast inference (<10ms per transaction)

### vs Traditional ML

| Aspect | Random Forest | Neural Network |
|-------|---------------|----------------|
| Feature engineering | Manual | Automatic |
| Interactions | Limited | Unlimited |
| Interpretability | High | Low |
| Best for | Baseline | Production |

---

## Future Work

- Ensemble with XGBoost + Random Forest
- Threshold optimization for F1
- SHAP explanations
- Transaction sequence modeling

---

**Dataset**: IEEE-CIS Fraud Detection on Kaggle  
**Built with**: TensorFlow, scikit-learn, Flask, Chart.js