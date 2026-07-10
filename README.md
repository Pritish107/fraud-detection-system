# Real-Time Fraud Detection System with Explainable AI and Model Drift Monitoring

End-to-end ML system for detecting fraudulent financial transactions, explaining individual
predictions with SHAP, monitoring for data/concept drift, and serving predictions via a
FastAPI REST API.

**Status: work in progress.** This README is updated as each module lands.

## Problem

Fraudulent transactions are a small fraction (<5%) of total volume, so naive accuracy-optimized
models miss most fraud. This project optimizes for precision/recall tradeoffs, explains flagged
transactions, and monitors for concept drift so retraining can be triggered before performance
silently degrades.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle competition).
See [Setup](#setup) for how to download it — requires a free Kaggle account.

## Project structure

```
data/
  raw/            # original Kaggle CSVs (gitignored)
  interim/        # intermediate merged/cleaned data (gitignored)
  processed/      # model-ready feature tables (gitignored)
notebooks/        # exploratory notebooks
src/
  data/           # data download + loading
  features/       # feature engineering
  models/         # training, imbalance handling, evaluation
  explainability/ # SHAP global/local explanations
  drift/          # PSI/KS/evidently drift monitoring
  api/            # FastAPI service
tests/            # pytest suite
models/           # trained model artifacts (gitignored)
reports/
  eda/            # EDA report output
  figures/        # generated plots
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Download the dataset

1. Create a Kaggle API token at https://www.kaggle.com/settings -> API -> "Create New Token"
   (downloads `kaggle.json`).
2. Place it at `C:\Users\<you>\.kaggle\kaggle.json`.
3. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection/rules
   (Kaggle rejects API downloads for competitions you haven't joined).
4. Run:
   ```powershell
   .venv\Scripts\python src/data/download_data.py
   ```

## Methodology

_(filled in as each module is built)_

- **EDA**: TBD
- **Imbalance handling**: TBD — comparing class weighting, SMOTE/ADASYN, and threshold tuning
- **Model**: baseline Logistic Regression, main model LightGBM/XGBoost
- **Explainability**: SHAP global + local explanations
- **Drift monitoring**: PSI/KS statistics and/or evidently, time-split comparison
- **API**: FastAPI service returning fraud probability, decision, and top contributing features

## Results

_(filled in after model training)_

## Limitations

_(filled in as they're discovered)_
