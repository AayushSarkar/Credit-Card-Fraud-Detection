# 💳 Credit Card Fraud Detection — Machine Learning & Streamlit Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Model-orange.svg?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF8000.svg?logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced%20Data-critical)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> A high-performance fraud detection system built with **Machine Learning**, **SMOTE**, **Random Forest**, **XGBoost**, and a modern **Streamlit dashboard**.  
> Designed for real-world **fintech fraud analytics**, offering insights, explainability (SHAP), and real-time fraud scoring.

---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Key Features](#key-features)
4. [Tech Stack](#tech-stack)
5. [Dataset](#dataset)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Modeling Approach](#modeling-approach)
8. [Results](#results)
9. [Streamlit Dashboard](#streamlit-dashboard)
10. [How to Run](#how-to-run)
11. [Future Enhancements](#future-enhancements)
12. [Acknowledgements](#acknowledgements)
13. [Authors](#authors)

---

## 🌍 Overview
Credit card fraud is extremely rare (**0.172%** of total transactions), making it a highly imbalanced and challenging problem.  
This project builds an end-to-end **Fraud Detection System** using:

- Supervised ML models  
- SMOTE balancing  
- PCA-based hidden fraud signals  
- A professional Streamlit Dashboard  
- Explainable AI (SHAP)

The system is optimized for **accuracy, interpretability, and real-time detection**.

---

## 💡 Motivation
Banks lose billions annually due to fraudulent transactions.  
Rule-based systems fail because fraud patterns evolve rapidly.

This project aims to:

- Understand hidden PCA fraud signals  
- Solve class imbalance (492 fraud vs 284,315 genuine)  
- Build high-AUC fraud classifiers  
- Offer explainability using SHAP  
- Provide a real-time user-friendly dashboard  

---

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| 🧠 **Random Forest + XGBoost Models** | High AUC fraud prediction |
| ⚖️ **SMOTE Oversampling** | Fixes extreme imbalance |
| 📊 **Analytics Dashboard** | ROC, PR Curve, Confusion Matrix |
| 🔍 **Explainability (SHAP)** | Why the model predicted fraud |
| 🚨 **Real-Time Prediction** | Score individual transactions |
| 📂 **Batch Prediction** | Upload CSV → get results instantly |
| 🎨 **Fintech UI** | Modern purple–blue theme |

---

## 🛠️ Tech Stack

### Languages  
- Python 3.10+

### Libraries  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-Learn  
- XGBoost  
- SMOTE (imbalanced-learn)  
- Streamlit  
- SHAP  
- KaggleHub  
- Joblib  

---

## 📂 Dataset

**Source:** Kaggle — *Credit Card Fraud Detection*  
**Rows:** 284,807  
**Fraud cases:** 492  
**Fraud percentage:** **0.172%**

### Columns
- `Time` — seconds since first transaction  
- `Amount` — transaction amount  
- `V1–V28` — PCA-anonymized features  
- `Class` — 0 = Genuine, 1 = Fraud  

---

## 🔍 Exploratory Data Analysis

### ✔ Key Insights  
- Fraud transactions contain **strong outliers**  
- PCA features **V1, V3, V4, V10, V12, V14** show maximum class separation  
- Fraud amounts follow different patterns  
- Time suggests clustered fraud activity  
- Dataset is **highly imbalanced**, requiring SMOTE  

### ✔ Visualizations  
- Amount & Time histograms  
- Boxplots (Amount, Time, PCA features)  
- PCA distribution analysis  
- Correlation heatmap  
- Class imbalance plot  

---

## 🧮 Modeling Approach

### **1️⃣ Preprocessing**
- StandardScaler on `Amount` & `Time`  
- PCA features unchanged  

# 💳 Credit Card Fraud Detection — Machine Learning & Streamlit Dashboard

A complete end-to-end ML project for detecting fraudulent credit card transactions using  
**SMOTE**, **Random Forest**, **XGBoost**, **Explainable AI (SHAP)**, and a **Streamlit Dashboard**.

---

## 🚀 2️⃣ Train–Test Split
```python
train_test_split(X, y, test_size=0.2, stratify=y)
🧮 3️⃣ SMOTE Oversampling
Balances fraud and genuine transactions in the training dataset.

🤖 4️⃣ Algorithms Used
Model	AUC
Logistic Regression	~0.98
Random Forest	0.9849
XGBoost	~0.9763

📌 Final model saved:
best_fraud_model_rf.joblib

🧠 5️⃣ Explainable AI (XAI)
SHAP force plot shows feature-wise contribution towards prediction.

📈 Results
✔ Performance Summary
AUC: 0.9849

High Precision & Recall

PR Curve optimized for rare-event detection

Low false positives

Very low false negatives

✔ Outputs Generated
ROC Curve

PR Curve

Confusion Matrix

SHAP Feature Impact

🖥️ Streamlit Dashboard
Tabs Included
Tab	Function
Model Performance	Upload test CSV → see evaluation metrics
Single Prediction	Enter basic values → PCA auto-generated → fraud score
Batch Prediction	Upload CSV → get predictions for all rows
Explainability	SHAP force plot for transparency

🎨 UI Highlights
Blue–purple gradient fintech theme

Compact charts

Clean metric cards

Modern layout

⚙️ How to Run
1️⃣ Install Dependencies
bash
Copy code
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn xgboost streamlit shap joblib kagglehub
2️⃣ Download Dataset (Optional)
python
Copy code
import kagglehub
kagglehub.dataset_download("mlg-ulb/creditcardfraud")
3️⃣ Run Streamlit App
bash
Copy code
streamlit run app.py
🌱 Future Enhancements
Feature	Description
📡 API Deployment	Real-time fraud detection API
🧠 Deep Learning	Autoencoders & LSTM models
📊 Monitoring	Track model drift
🔐 PCI-DSS Secure Version	Bank-grade secure version
☁️ Cloud Deployment	HuggingFace / AWS Spaces

🙏 Acknowledgements
Kaggle Contributors

Scikit-Learn Community

XGBoost Developers

Streamlit Team

SHAP Authors
