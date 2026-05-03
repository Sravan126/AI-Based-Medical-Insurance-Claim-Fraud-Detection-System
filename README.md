# 🛡️ Medical Insurance Fraud Detection System

A machine learning-based system to automatically classify medical insurance claims as **Fraudulent** or **Genuine** using structured claim data and multiple classification algorithms.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [License](#license)

---

## 🔍 Overview

Medical insurance fraud results in billions of dollars in losses for insurance companies and healthcare systems every year. This project presents an automated, scalable fraud detection system that leverages machine learning to identify suspicious claims early — reducing reliance on manual audits and rule-based systems.

---

## ❗ Problem Statement

Traditional fraud detection methods are:
- **Manual and time-consuming** — relying heavily on human auditors
- **Rule-based and rigid** — unable to adapt to new fraud patterns
- **Reactive rather than proactive** — fraud is detected after losses occur

This system addresses these gaps by using predictive analytics to flag fraudulent claims **before** they are processed.

---

## 📊 Dataset

The dataset contains structured medical insurance claim records with the following types of attributes:

| Category | Features |
|---|---|
| Patient Details | Age, gender, medical history |
| Billing Information | Claim amount, billing codes |
| Diagnosis Codes | ICD codes, diagnosis categories |
| Treatment Costs | Procedure costs, hospitalization charges |
| Claim History | Previous claims, claim frequency |
| Provider Details | Provider type, provider history |
| **Target Variable** | `Fraud_Label` — `1` (Fraudulent) / `0` (Genuine) |

> **Note:** The dataset may be imbalanced (fraudulent claims are a minority). Appropriate resampling techniques (e.g., SMOTE) are applied during preprocessing.

---

## 📁 Project Structure

```
medical-insurance-fraud-detection/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and encoded dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Data cleaning and feature engineering
│   ├── 03_Modeling.ipynb       # Model training and evaluation
│   └── 04_Results.ipynb        # Comparison and final results
│
├── src/
│   ├── preprocess.py           # Data preprocessing pipeline
│   ├── train.py                # Model training scripts
│   ├── evaluate.py             # Evaluation metrics and plots
│   └── predict.py              # Inference on new claims
│
├── models/
│   └── *.pkl                   # Saved trained models
│
├── outputs/
│   ├── confusion_matrices/     # Confusion matrix plots
│   ├── roc_curves/             # AUC-ROC curve plots
│   └── results_summary.csv     # Comparison of all model metrics
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/medical-insurance-fraud-detection.git
cd medical-insurance-fraud-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
xgboost
lightgbm
joblib
jupyter
```

---

## 🚀 Usage

### 1. Run the Full Pipeline

```bash
python src/preprocess.py       # Clean and encode data
python src/train.py            # Train all models
python src/evaluate.py         # Generate metrics and plots
```

### 2. Predict on New Claims

```python
from src.predict import predict_claim

claim = {
    "patient_age": 45,
    "claim_amount": 12000,
    "diagnosis_code": "Z51.11",
    "previous_claims": 3,
    # ... other features
}

result = predict_claim(claim)
print(result)  # Output: "Fraudulent" or "Genuine"
```

### 3. Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

---

## 🔧 Machine Learning Pipeline

```
Raw Data
   │
   ▼
Data Cleaning          ← Handle missing values, remove duplicates
   │
   ▼
Feature Encoding       ← Label encoding / One-hot encoding for categoricals
   │
   ▼
Normalization          ← MinMaxScaler / StandardScaler for numerical features
   │
   ▼
Imbalance Handling     ← SMOTE / class_weight balancing
   │
   ▼
Train-Test Split       ← Stratified 80/20 split
   │
   ▼
Model Training         ← Multiple classifiers trained and tuned
   │
   ▼
Evaluation             ← Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
   │
   ▼
Best Model Selection   ← Saved as .pkl for deployment
```

---

## 🤖 Models Used

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting for high accuracy |
| XGBoost | Optimized gradient boosting framework |
| LightGBM | Fast, memory-efficient boosting |
| Support Vector Machine (SVM) | Margin-based classifier |
| K-Nearest Neighbors (KNN) | Distance-based classifier |

---

## 📈 Evaluation Metrics

Each model is evaluated using the following metrics:

- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted frauds, how many are truly fraudulent
- **Recall (Sensitivity)** — Of actual frauds, how many were correctly detected
- **F1-Score** — Harmonic mean of Precision and Recall
- **AUC-ROC** — Area under the Receiver Operating Characteristic curve
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN

> ⚠️ In fraud detection, **Recall** is prioritized to minimize False Negatives (missed fraud cases).

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| LightGBM | — | — | — | — | — |

> 📝 Results will be populated after running `src/evaluate.py` on your dataset.

---

## 🛠️ Technologies Used

- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, XGBoost, LightGBM, imbalanced-learn
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## 🔮 Future Work

- [ ] Deploy as a REST API using Flask or FastAPI
- [ ] Build an interactive dashboard for claim monitoring
- [ ] Integrate deep learning models (e.g., Autoencoders for anomaly detection)
- [ ] Incorporate real-time claim stream processing
- [ ] Apply explainability tools (SHAP, LIME) for model interpretability
- [ ] Explore graph-based fraud detection for provider network analysis

---

## 🙋‍♂️ Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

> *"Catching fraud early saves money, protects patients, and keeps healthcare systems fair."*
