# AML Mule Risk Scoring

Synthetic dataset generation, feature engineering, and machine learning models for detecting mule accounts in Anti–Money Laundering (AML) systems.

This repository is part of the MSc Artificial Intelligence thesis titled:

> **"AI-Based Risk Scoring System for Mule Account Detection: A Prototype for Preemptive Fraud Detection in Digital Banking."**

---

## 📌 Overview

This project implements a **dual-source risk scoring framework** to identify suspicious banking accounts (mule accounts) using synthetic transaction data. The work includes:

- Synthetic data generation (based on AMLSim patterns)
- Feature engineering using transaction–level behavioural indicators
- Rule-based baseline risk scoring
- Machine learning model comparison (XGBoost / LightGBM)
- Evaluation using precision, recall, PR–AUC, ROC–AUC, and alert volume reduction

The primary objective is to **compare an AI-based model with a rule-based system** and demonstrate improved detection accuracy and lower false-positive alerts.

---

## 📂 Project Structure

aml-mule-risk-scoring/
├── data/
│   ├── raw/             # synthetic accounts + transactions CSV files
│   └── processed/       # engineered account-level feature tables
├── notebooks/
│   └── 01_data_generation_and_eda.ipynb
├── src/
│   ├── data_generation.py     # AML typology transaction generators
│   ├── feature_engineering.py # transform raw txns → features
│   ├── rule_baseline.py       # rule system implementation
│   └── model_training.py      # ML model, training, evaluation
├── README.md
├── requirements.txt
└── .gitignore


---

## 🧪 AML Typologies Modelled

The synthetic generator includes **five classic money laundering patterns**:

1. **Fan-in model** — Many accounts send small amounts to a single mule.
2. **Fan-out model** — A mule distributes funds to many recipients.
3. **Rapid in–out (velocity)** — Fast back-to-back transfers.
4. **Structuring (smurfing)** — Deposits just below the \$10,000 threshold.
5. **Circular layering** — Multi-hop transaction chains to obscure origin.

Normal account behaviour is simulated for comparison.

---

## 🧠 Features

The account-level feature table (`account_features.csv`) includes:

- Flow features (inflow, outflow, net flow)
- Network structure features (fan-in, fan-out)
- Velocity features
- Structuring indicators
- Statistical and anomaly indicators

Total: **17 engineered features** used as ML input.

---

## 🧾 Baseline and ML Models

- **Rule-based baseline:**  
  Manually designed thresholds on fan-in, fan-out, and structuring signals.

- **Machine learning model:**  
  Gradient-boosted binary classifier (XGBoost or LightGBM).

- **Target:**  
  Detect **mule accounts** — `is_mule = 1`.

---

## 📊 Evaluation

Evaluation will compare:

- Precision  
- Recall  
- F1-score  
- ROC–AUC  
- PR–AUC  
- **Alert volume at fixed recall (80%)**  

This metric demonstrates operational impact — a key AML industry expectation.

---
