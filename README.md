# Credit Card Fraud Detection Using Leakage-Aware Statistical ML Pipelines

An end-to-end statistical machine learning project for detecting fraudulent credit card transactions under severe class imbalance (`0.2591%` fraud rate across `41,682` validated records), built with strict train/validation/test isolation to prevent data leakage.

## Overview

Real-world card transaction logs are heavily skewed toward legitimate activity, making overall classification accuracy misleading. In addition, fitting imputers, scalers, or resampling steps across the full dataset before splitting leaks distributional information across partitions.

This repository implements an eight-stage pipeline comparing cost-sensitive **Logistic Regression** (`class_weight='balanced'`) and **Random Forest** (`class_weight='balanced_subsample'`, `120` trees, `max_depth=16`, `min_samples_leaf=2`) using **PR-AUC**, **F1-Score**, **F2-Score**, **ROC-AUC**, **Balanced Accuracy**, and **Matthews Correlation Coefficient (MCC)**.

## Dataset & Feature Engineering

- **Source Records:** `creditcard.csv` (`41,683` raw rows, `31` columns); schema validation removes `1` record with an invalid target label, leaving `41,682` valid rows (`41,574` legitimate / `99.7409%` and `108` fraud / `0.2591%`).
- **Deterministic Features (`32` total predictors):**
  - `Amount_log = ln(1 + max(Amount, 0))` to reduce right-skew in transaction amounts
  - `Hour = floor(Time / 3600) mod 24` to capture cyclic hour-of-day patterns
- **Stratified Split (`RANDOM_STATE = 42`):**
  - **Train (`64%`):** `26,676` transactions (`69` fraud)
  - **Validation (`16%`):** `6,669` transactions (`17` fraud)
  - **Holdout Test (`20%`):** `8,337` transactions (`22` fraud)

## Validation Results (Threshold = 0.50)

| Metric | Random Forest (Balanced Subsample) | Logistic Regression (Balanced) |
| :--- | :---: | :---: |
| **PR-AUC** | **0.9710** | 0.7984 |
| **ROC-AUC** | 0.9999 | 0.9995 |
| **Precision** | **1.0000** | 0.1318 |
| **Recall** | 0.8235 | **1.0000** |
| **F1-Score** | **0.9032** | 0.2329 |
| **F2-Score** | **0.8537** | 0.4315 |
| **Balanced Accuracy** | 0.9118 | **0.9916** |
| **MCC** | **0.9073** | 0.3600 |
| **Fit Time** | `18.2324 s` | `1.2995 s` |

## Project Structure

```text
├── creditcard.csv
├── notebook.ipynb          # End-to-end EDA, pipeline training, and evaluation
├── report.docx             # Academic report (15-20 pages, Times New Roman)
└── README.md
