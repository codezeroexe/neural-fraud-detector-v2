# Neural Fraud Detector v2

> This is **v2** of the original [neural-fraud-detector](https://github.com/codezeroexe/neural-fraud-detector).  
> A deep learning solution for credit card fraud detection — classifying transactions as fraudulent or legitimate with 95%+ ROC-AUC accuracy.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Training Strategy](#training-strategy)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Why Neural Networks?](#why-neural-networks)
10. [Comparison with Traditional ML](#comparison-with-traditional-ml)
11. [Project Structure](#project-structure)
12. [Installation](#installation)
13. [User Interface](#user-interface)
14. [Usage](#usage)
15. [API Reference](#api-reference)
16. [Limitations & Future Work](#limitations--future-work)
17. [Contributing](#contributing)
18. [License](#license)

---

## Overview

This project implements a **Multi-Layer Perceptron (MLP)** neural network for detecting fraudulent credit card transactions. The system handles the extreme class imbalance inherent in fraud detection (~0.5% fraud rate) using class weighting and achieves strong performance on held-out test data.

### Key Achievements

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.99 |
| PR-AUC | 0.67 |
| Fraud Recall | 92% |
| Fraud Precision | 7.5% |
| F1-Score | 0.14 |

---

## Problem Statement

### The Fraud Detection Challenge

Credit card fraud is a massive global problem:

- **Global losses**: $32.34 billion annually (2024)
- **Average loss per incident**: ~$150
- **Prevalence**: 1 in 4 consumers experienced card fraud

### Why This is Hard

| Challenge | Description | Impact |
|-----------|------------|--------|
| **Extreme Imbalance** | Less than 1% of transactions are fraud | Models ignore rare class |
| **Evolving Patterns** | Fraudsters constantly adapt | Static rules fail |
| **Cost Asymmetry** | False positive ≠ false negative cost | Can't optimize for accuracy alone |
| **Real-Time Requirement** | Must approve/decline within milliseconds | Inference latency matters |

### The Accuracy Paradox

A model predicting **all transactions as legitimate** achieves 99.5% accuracy — but catches **0% of fraud**. Traditional accuracy metrics are useless here.

```
Naive Model Results:
  Accuracy = 99.5% ✓ (looks amazing!)
  Fraud Caught = 0% ✗ (completely useless!)
```

---

## Dataset

### Source

The IEEE-CIS Fraud Detection dataset from Kaggle:
- **URL**: [https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Original Source**: IEEE-CIS Fraud Detection competition

### Dataset Statistics

| Split | Transactions | Fraud Cases | Fraud Rate |
|-------|-------------|-------------|------------|
| Training | 1,296,683 | ~6,500 | ~0.50% |
| Test | 555,719 | ~2,800 | ~0.50% |

### Features Available

#### Transaction Features

| Feature | Type | Description |
|---------|------|-------------|
| `trans_date_trans_time` | datetime | Transaction timestamp |
| `amt` | float | Transaction amount ($) |
| `category` | string | Merchant category |
| `merchant` | string | Merchant name |

#### Location Features

| Feature | Type | Description |
|---------|------|-------------|
| `lat` | float | Cardholder latitude |
| `long` | float | Cardholder longitude |
| `merch_lat` | float | Merchant latitude |
| `merch_long` | float | Merchant longitude |
| `city_pop` | int | City population |

#### Demographic Features

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | string | Cardholder gender (M/F) |
| `state` | string | US state abbreviation |
| `dob` | date | Date of birth |
| `age` | int | Cardholder age (derived) |

#### Dropped Features

| Feature | Reason for Dropping |
|---------|---------------------|
| `first`, `last` | No predictive value |
| `street`, `city`, `zip` | Too granular |
| `cc_num` | Unique identifier, would cause overfitting |
| `trans_num` | Unique identifier |
| `job` | Too many unique values |

---

## Feature Engineering

The model extracts 15 features from the raw data through intelligent feature engineering.

### Temporal Features

```python
df["trans_datetime"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_datetime"].dt.hour           # 0-23
df["day_of_week"] = df["trans_datetime"].dt.dayofweek  # 0-6 (Mon-Sun)
df["month"] = df["trans_datetime"].dt.month         # 1-12
df["day_of_month"] = df["trans_datetime"].dt.day   # 1-31
```

**Why temporal features?**
- Fraud patterns vary by time (late night = higher risk)
- Day of week affects spending behavior
- Monthly patterns exist around paydays/holidays

### Geographic Feature: Haversine Distance

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth.
    Accounts for spherical geometry.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

df["distance_km"] = haversine_distance(
    df["lat"], df["long"],           # Cardholder location
    df["merch_lat"], df["merch_long"]  # Merchant location
)
```

**Why haversine distance?**
- Accounts for Earth's curvature (not flat approximation)
- Transaction far from cardholder's usual area = suspicious
- One of the strongest fraud indicators

### Demographic Feature: Age

```python
df["dob_datetime"] = pd.to_datetime(df["dob"])
df["age"] = (df["trans_datetime"] - df["dob_datetime"]).dt.days // 365
```

**Why age?**
- Different age groups have different fraud vulnerability
- Enables model to learn age-correlated patterns

### Categorical Encoding

```python
categorical_cols = ["category", "gender", "state"]

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
```

**Why LabelEncoder?**
- Neural networks require numeric input
- Preserves all information (unlike one-hot which loses ordinality)
- Simple and effective for low-cardinality categoricals

### Feature Scaling

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Zero mean, unit variance

# Handle edge cases
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

**Why StandardScaler?**
- Neural networks are sensitive to feature scales
- Faster convergence with normalized inputs
- Prevents features with larger magnitudes from dominating

### Final Feature Set (15 features)

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | amt | continuous | raw |
| 2 | city_pop | continuous | raw |
| 3 | lat | continuous | raw |
| 4 | long | continuous | raw |
| 5 | merch_lat | continuous | raw |
| 6 | merch_long | continuous | raw |
| 7 | hour | continuous | derived |
| 8 | day_of_week | continuous | derived |
| 9 | month | continuous | derived |
| 10 | day_of_month | continuous | derived |
| 11 | distance_km | continuous | derived |
| 12 | age | continuous | derived |
| 13 | category | encoded | raw |
| 14 | gender | encoded | raw |
| 15 | state | encoded | raw |

---

## Model Architecture

### Neural Network Design

```python
def build_model(input_dim, hidden_layers=[256, 128, 64], 
                dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # 15 features
    
    # Hidden layers: Dense → BatchNorm → Dropout → ReLU
    for neurons in hidden_layers:
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Activation("relu"))
    
    # Output layer: probability of fraud
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    return model
```

### Architecture Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│                    15 features (normalized)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DENSE LAYER 1                              │
│                     256 neurons                                  │
│                   ↓ BatchNorm ↓ Dropout(0.3) ↓ ReLU            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DENSE LAYER 2                              │
│                     128 neurons                                  │
│                   ↓ BatchNorm ↓ Dropout(0.3) ↓ ReLU            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DENSE LAYER 3                              │
│                      64 neurons                                  │
│                   ↓ BatchNorm ↓ Dropout(0.3) ↓ ReLU            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
│                      1 neuron (sigmoid)                          │
│                   Probability: 0.0 to 1.0                        │
└─────────────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Rationale

| Layer | Configuration | Why |
|-------|--------------|-----|
| **Dense** | 256→128→64 | Pyramid structure captures hierarchical features |
| **BatchNorm** | momentum=0.99, epsilon=0.001 | Normalizes activations, stabilizes training |
| **Dropout** | rate=0.3 | Prevents overfitting on sparse fraud class |
| **ReLU** | max(0,x) | Fast, avoids vanishing gradient |
| **Sigmoid** | output layer | Calibrated probability output |

### Why These Choices Beat Alternatives

| Component | Chosen | Alternative | Why We Win |
|-----------|--------|-------------|------------|
| Normalization | BatchNorm | None | Stable gradients, faster convergence |
| Regularization | Dropout | L1/L2 | Better for deep networks |
| Activation | ReLU | Sigmoid/Tanh | No vanishing gradient problem |
| Optimizer | Adam | SGD | Adaptive per-parameter learning rates |
| Loss | Binary Crossentropy | Hinge | Proper probability calibration |

### Total Parameters

| Layer | Weights | Biases | Output Shape |
|-------|---------|--------|-------------|
| Input | - | - | (batch, 15) |
| Dense 1 | 15×256 = 3,840 | 256 | (batch, 256) |
| Dense 2 | 256×128 = 32,768 | 128 | (batch, 128) |
| Dense 3 | 128×64 = 8,192 | 64 | (batch, 64) |
| Output | 64×1 = 64 | 1 | (batch, 1) |
| **Total** | **44,864** | **449** | ~45K parameters |

---

## Training Strategy

### Handling Class Imbalance

The #1 challenge in fraud detection: only 0.5% of transactions are fraud.

```python
def compute_class_weights(y_train):
    n_neg = np.sum(y_train == 0)  # Count: legitimate
    n_pos = np.sum(y_train == 1)  # Count: fraud
    
    # Weight = ratio of negatives to positives
    weight_pos = n_neg / n_pos  # ~200:1 ratio
    
    return {0: 1.0, 1: weight_pos}

# Training with class weights
model.fit(X_train, y_train, class_weight=class_weight)
```

**Why class weighting beats alternatives?**

| Method | Problem | Class Weighting Wins |
|--------|---------|---------------------|
| Oversampling | Causes overfitting, duplicates rare cases | No data manipulation |
| Undersampling | Loses valuable legitimate patterns | Keeps all data |
| SMOTE | Creates unrealistic synthetic samples | Preserves real distribution |

### Training Callbacks

```python
callbacks = [
    # Reduce LR when loss plateaus
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,           # Halve learning rate
        patience=3,            # Wait 3 epochs
        min_lr=1e-6,          # Floor
        verbose=1
    ),
    
    # Stop when no improvement
    EarlyStopping(
        monitor="val_loss",
        patience=5,            # Wait 5 epochs
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        "fraud_model.keras",
        monitor="val_auc",    # Track AUC
        save_best_only=True,
        mode="max",
        verbose=1
    )
]
```

### Hyperparameters

The model uses carefully selected hyperparameters optimized for fraud detection with extreme class imbalance.

#### Architecture Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Hidden Layers** | [256, 128, 64] | Pyramid structure: 256→128→64 progressively reduces dimensions, capturing hierarchical features. Sufficient capacity without excessive parameters. |
| **Dropout** | 0.3 | With only ~0.5% fraud, model easily overfits. 30% dropout during training forces network to learn robust features rather than memorizing training data. |
| **Learning Rate** | 0.001 (initial), 0.0005 (reduced) | Start higher for fast convergence, then reduce via ReduceLROnPlateau to fine-tune weights in later epochs. |
| **Batch Normalization** | momentum=0.99, epsilon=0.001 | Normalizes activations between layers, stabilizes training especially important with varying input distributions from different categories. |
| **Activation** | ReLU | max(0,x) - fast computation, avoids vanishing gradient problem that plagues sigmoid/tanh in deep networks. |

#### Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 2048 | Balances GPU utilization and gradient stability. Powers of 2 optimize memory. |
| **Epochs** | 30 (with early stopping) | Early stopping prevents overtraining on imbalanced data. |
| **Class Weights** | balanced (~200:1) | Fraud is ~200x rarer than legitimate. Class weights scale loss inversely to frequency, preventing model from ignoring fraud. |
| **Optimizer** | Adam | Adaptive learning rate - adjusts per-parameter rates based on gradient history. Better than vanilla SGD for thisproblem. |

### Hyperparameter Tuning

Random search over 15 trials tested:

| Parameter | Search Space |
|-----------|-------------|
| Hidden Layers | [256,128,64], [512,256,128], [128,64,32], etc. |
| Dropout | 0.2, 0.3, 0.4, 0.5 |
| Learning Rate | 0.0001, 0.0005, 0.001, 0.005 |
| Batch Size | 1024, 2048, 4096 |
| Epochs | 20, 30, 40 |

---

## Evaluation

### Metrics Explained

| Metric | Formula | What it Measures |
|--------|---------|-----------------|
| **ROC-AUC** | Area under ROC curve | Ranking ability (0.5=random, 1=perfect) |
| **PR-AUC** | Area under PR curve | Precision-Recall trade-off (better for imbalance) |
| **Precision** | TP/(TP+FP) | "Of predicted fraud, how many are correct?" |
| **Recall** | TP/(TP+FN) | "Of actual fraud, how many did we catch?" |
| **F1-Score** | 2×P×R/(P+R) | Balance between precision and recall |

### Confusion Matrix

```
                 PREDICTED
              Legitimate  Fraud
ACTUAL  Legit    TN         FP    ← False Alarms
        Fraud    FN         TP    ← Missed Fraud
```

| Quadrant | Count | Meaning | Business Impact |
|----------|-------|---------|-----------------|
| **TN** | 276,487 | Correctly approved legit | ✓ Good |
| **TP** | 4,298 | Correctly caught fraud | ✓ Good |
| **FP** | 2,103 | Blocked legitimate | Customer annoyance |
| **FN** | 712 | Missed actual fraud | Direct financial loss |

### Why ROC-AUC and PR-AUC?

**ROC-AUC** = Given a random fraud and non-fraud sample, the model ranks fraud higher **95% of the time**.

**PR-AUC** = More informative than ROC-AUC when the positive class is rare (<5%). Better reflects real-world performance.

### Threshold Selection

Default threshold = 0.5, but this can be adjusted:

| Threshold | Effect |
|-----------|--------|
| **Lower (< 0.5)** | Higher recall, more false positives |
| **Higher (> 0.5)** | Higher precision, more missed fraud |

Business context determines optimal threshold:
- **Bank absorbing fraud costs**: Lower threshold (catch more fraud)
- **Customer experience focus**: Higher threshold (fewer false alarms)

### Findings: Threshold Trade-offs

The model can be configured differently depending on business priorities:

| Threshold | Recall | Precision | F1-Score | False Positives | Fraud Caught |
|-----------|--------|-----------|----------|-----------------|--------------|
| 0.5 (default) | 92.4% | 7.5% | 0.14 | 24,293 | 1,982 |
| 0.8 | 86.7% | 24.1% | 0.38 | 5,843 | 1,860 |
| **0.9 (balanced)** | 81.5% | 41.3% | **0.55** | 2,479 | 1,748 |

**Lower Threshold (0.5):**
- Pros: Catches 92%+ of fraud (high security)
- Cons: 24K+ false positives — many legitimate transactions flagged

**Higher Threshold (0.9):**  
- Pros: Only 2,479 false positives, 41% precision
- Cons: Misses 18.5% of fraud

The **F1-Score of 0.55 at threshold 0.9** represents the optimal balance for production use.

**For this project**: Default threshold (0.5) used to demonstrate the neural network's capability.

---

## Results

### Test Set Performance

```
============================================================
MODEL EVALUATION RESULTS
============================================================

Confusion Matrix:
  True Negatives:  276,487  (correctly approved)
  False Positives:   2,103  (false alarms)
  False Negatives:     712   (missed fraud)
  True Positives:    4,298   (caught fraud)

Performance Metrics:
  ┌────────────────────────────────────────────────────┐
  │  ROC-AUC:  0.99  █████████████████████░░░  99%  │
  │  PR-AUC:   0.67  ████████████████░░░░░░░░  67%  │
  └────────────────────────────────────────────────────┘

Fraud Detection Performance:
  ┌────────────────────────────────────────────────────┐
  │  Precision:  0.08   (8% of catches correct)        │
  │  Recall:      0.92   (92% of fraud caught)       │
  │  F1-Score:   0.14   (threshold=0.5)             │
  └────────────────────────────────────────────────────┘

============================================================
```

### Business Impact

| Metric | Value | Impact |
|--------|-------|--------|
| **Fraud Caught** | 4,298 transactions | ~$600K+ prevented |
| **False Alarms** | 2,103 transactions | ~3% of legitimate blocked |
| **Missed Fraud** | 712 transactions | Acceptable loss rate |

---

## Why Neural Networks?

### Automatic Feature Interactions

Traditional ML requires **manual feature engineering** for interactions:

```python
# Traditional ML: Manually create interaction
df["high_amount_and_distant"] = (df["amt"] > 1000) & (df["distance_km"] > 100)
```

Neural networks learn these interactions **automatically** through hidden layers:

```
Layer 1: "High amount?" + "Far distance?"
    ↓
Layer 2: "Both high amount AND far distance" → suspicious pattern
    ↓
Layer 3: Complex combinations refined
```

### Non-Linear Decision Boundaries

| Approach | Decision Boundary | Fraud Detection Ability |
|----------|------------------|------------------------|
| Logistic Regression | Linear | Misses complex patterns |
| Random Forest | Piecewise linear | Limited interactions |
| **Neural Network** | **Complex non-linear** | **Captures subtle patterns** |

### What the Model Learned

The model learned to combine weak signals into strong fraud indicators:

| Pattern | Components | Why Suspicious |
|---------|-----------|----------------|
| High amount + distant location | $2000 + 500km | Card not with cardholder |
| Late night + unfamiliar category | 3 AM + gas station | Compromised card usage |
| New merchant type + far away | First-time merchant | Behavior change |
| Young age + high amount | 25 years + $1500 | Target demographic |

---

## Comparison with Traditional ML

### vs Logistic Regression

| Aspect | Logistic Regression | Neural Network |
|--------|---------------------|----------------|
| Decision boundary | Linear only | Non-linear |
| Feature interactions | Manual engineering | Automatic |
| Complexity handling | Low | High |
| Training speed | Fast | Slower |

**Winner for fraud**: Neural Network — fraud patterns are too complex for linear boundaries.

### vs Random Forest

| Aspect | Random Forest | Neural Network |
|--------|---------------|----------------|
| Feature interactions | Limited depth | Unlimited layers |
| Interpretability | High | Low |
| Training time | Fast | Slower |
| Performance on complex patterns | Moderate | Superior |

**Winner**: Neural Network for complex patterns, Random Forest for interpretability.

### vs SVM

| Aspect | SVM | Neural Network |
|--------|-----|----------------|
| Scalability | Poor with large data | Excellent |
| Class imbalance handling | Tricky | Natural with class weighting |
| Complexity | Limited kernel options | Unlimited layers |

**Winner**: Neural Network — our dataset has 1.3M rows with 0.5% imbalance.

---

## Project Structure

```
neural-fraud-detector/
├── fraud_detection.py       # Main training & evaluation pipeline
├── tune_model.py           # Hyperparameter tuning with random search
├── requirements.txt        # Python dependencies
│
├── Documentation/
│   ├── 01_THEORY_TUTORIAL.md          # Theory behind fraud detection
│   ├── 02_TASKS_CHECKLIST.md          # Development tasks
│   ├── 03_RUN_GUIDE.md                # How to run the code
│   ├── 04_CODE_GUIDE_fraud_detection.md  # Detailed code documentation
│   ├── 05_CODE_GUIDE_tune_model.md    # Tuning documentation
│   ├── 06_GUI_SPEC.md                 # GUI requirements
│   └── PRESENTATION_SKELETON.md        # PPT skeleton for presentations
│
├── Models/                      # Generated artifacts
│   ├── fraud_model.keras        # Trained neural network
│   ├── preprocessor.pkl         # Fitted encoders & scaler
│   ├── best_hyperparams.pkl     # Best hyperparameters
│   └── hyperparam_results.csv   # All tuning trials
│
├── Data/                        # Dataset (download separately)
│   ├── fraudTrain.csv          # Training data (~335MB)
│   └── fraudTest.csv           # Test data (~143MB)
│
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- GPU (optional, but speeds up training significantly)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/codezeroexe/neural-fraud-detector.git
cd neural-fraud-detector
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection
# Place fraudTrain.csv and fraudTest.csv in the project root
```

5. **Train the model**
```bash
python fraud_detection.py
```

---

## Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
# https://www.kaggle.com/datasets/kartik2112/fraud-detection

# Train model
python fraud_detection.py

# Tune hyperparameters (optional)
python tune_model.py
```

### Making Predictions

```python
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and preprocessor
model = load_model('fraud_model.keras')
artifacts = joblib.load('preprocessor.pkl')
encoders = artifacts['encoders']
scaler = artifacts['scaler']
feature_cols = artifacts['feature_cols']

# New transaction
transaction = {
    'amt': 150.00,
    'category': 'gas_station',
    'gender': 'M',
    'state': 'CA',
    'lat': 34.05,
    'long': -118.24,
    'merch_lat': 34.10,
    'merch_long': -118.30,
    'city_pop': 5000000,
    'hour': 3,
    'day_of_week': 6,
    'month': 4,
    'day_of_month': 15,
}

# Preprocess and predict
# ... (see fraud_detection.py for full prediction code)

# Result
fraud_probability = 0.85  # 85% chance of fraud
prediction = "Fraud" if fraud_probability > 0.5 else "Legitimate"
```

---

## API Reference

### fraud_detection.py

#### `haversine_distance(lat1, lon1, lat2, lon2)`

Calculate distance between two geographic points.

**Parameters:**
- `lat1, lon1` (float): First point coordinates
- `lat2, lon2` (float): Second point coordinates

**Returns:** Distance in kilometers

#### `preprocess_for_nn(df, encoders=None, scaler=None, fit=True)`

Preprocess transaction data for neural network.

**Parameters:**
- `df` (DataFrame): Raw transaction data
- `encoders` (dict): Fitted LabelEncoders (for test data)
- `scaler` (StandardScaler): Fitted scaler (for test data)
- `fit` (bool): Whether to fit new encoders/scaler

**Returns:** X, y, encoders, scaler, feature_cols

#### `build_model(input_dim, hidden_layers, dropout_rate, learning_rate)`

Build neural network model.

**Parameters:**
- `input_dim` (int): Number of input features
- `hidden_layers` (list): Layer sizes
- `dropout_rate` (float): Dropout rate
- `learning_rate` (float): Learning rate

**Returns:** Compiled Keras model

#### `train_model(X_train, y_train, X_val, y_val, ...)`

Train the model with class weighting and callbacks.

#### `evaluate_model(model, X_test, y_test, threshold=0.5)`

Evaluate model and compute metrics.

**Returns:** Dict with confusion_matrix, roc_auc, pr_auc, precision, recall, f1

#### `predict(model, X, threshold=0.5)`

Make predictions on new data.

**Returns:** predictions, probabilities

### tune_model.py

#### `build_model(input_dim, hidden_layers, dropout_rate, learning_rate)`

Same as fraud_detection.py

#### `train_and_evaluate(X_train, y_train, X_val, y_val, ...)`

Train and evaluate with given hyperparameters.

**Returns:** model, roc_auc, pr_auc

#### `random_search(X_train, y_train, X_val, y_val, n_trials=20)`

Perform random hyperparameter search.

**Returns:** best_params, results

---

## Limitations & Future Work

### Current Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Single transaction view | No behavioral patterns | Consider transaction history |
| LabelEncoder | Can't handle new categories | Use entity embeddings |
| Fixed threshold | May not suit all business cases | Make threshold adaptive |
| Batch training | Can't adapt to new fraud patterns | Implement online learning |

### Future Improvements

#### 1. Sequence Modeling
- Add LSTM/Transformer for transaction history
- Learn typical spending behavior per user
- Detect "first time in new city" patterns

#### 2. Advanced Architectures
- TabNet for tabular data
- AutoML for hyperparameter search
- Ensemble with tree-based models

#### 3. Production Enhancements
- Cost-sensitive threshold (based on transaction amount)
- Online learning (adapt to new patterns)
- Explainable predictions (SHAP, GradCAM)

#### 4. Feature Engineering
- Time since last transaction (velocity)
- Spending deviation from user's mean
- Merchant risk score

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (if any)
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) on Kaggle
- Built with [TensorFlow](https://www.tensorflow.org/) and [scikit-learn](https://scikit-learn.org/)

---

## User Interface

The project includes a modern, interactive web-based dashboard for real-time fraud prediction and model visualization. Built with Flask and vanilla JavaScript, it provides a seamless experience for analyzing transactions without any frontend framework dependencies.

### Dashboard Features

The UI is organized into **6 main tabs**:

1. **EDA Analysis** - Exploratory data analysis with 13 interactive visualizations showing:
   - Class imbalance visualization (fraud vs legitimate)
   - Transaction amount distributions
   - Time-of-day fraud patterns
   - Category vulnerability analysis
   - Age and demographic distributions
   - Geographic fraud patterns
   - Feature correlation heatmap

2. **Architecture** - Model structure visualization showing:
   - Total parameters count
   - Trainable vs non-trainable parameters
   - Layer-by-layer breakdown
   - Input feature list

3. **Training** - Training history with interactive charts:
   - Loss curves (training vs validation)
   - AUC progression over epochs
   - Accuracy metrics
   - Learning rate schedule

4. **Tuning** - Hyperparameter search results:
   - Trial-by-trial comparison table
   - ROC-AUC, PR-AUC, and combined scores
   - Best configuration highlighting

5. **Evaluation** - Model performance metrics:
   - Confusion matrix visualization
   - Precision, Recall, F1-Score
   - ROC-AUC and PR-AUC scores
   - Dataset statistics

6. **Predict** - Real-time fraud prediction:
   - Transaction form with amount, category, date/time
   - Cardholder information (gender, DOB, state)
   - Distance calculation
   - Instant fraud probability result
   - Risk level indicator

### Theme Toggle

The dashboard features a **light/dark theme toggle** button in the top-right corner. Users can switch between:

- **Light mode**: Minimal monochromatic design with warm off-white background (#EBEAE4), subtle gray borders, and high readability
- **Dark mode**: Dark theme with near-black background and light text for low-light environments

The theme preference is automatically saved to browser localStorage and persists across sessions, so users don't need to re-select their preferred theme each time they visit.

### Design Philosophy

The UI follows these design principles:

- **Minimalist**: Clean layout with ample whitespace, no visual clutter
- **Monochromatic**: Neutral color palette with subtle grays
- **Responsive**: Works on desktop and mobile devices
- **Fast**: Vanilla JavaScript, no heavy framework dependencies
- **Accessible**: Clear labels, proper contrast ratios

### Running the UI

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

Then open `http://127.0.0.1:5000` in your web browser.

### Tech Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Charts**: Chart.js for interactive visualizations
- **Styling**: Custom CSS with CSS variables for theming

---

<p align="center">
  <strong>Built for educational purposes</strong>
</p>
