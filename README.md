# Real-Time Fraud Detection System with Explainable AI and Model Drift Monitoring

End-to-end ML system for detecting fraudulent financial transactions: a LightGBM classifier
tuned for precision/recall tradeoffs under severe class imbalance, SHAP-based explanations for
every prediction, a drift-monitoring module that tracks when the model needs retraining, and a
FastAPI service (plus a Streamlit dashboard) to serve it all.

## Problem

Fraudulent transactions are a small fraction of total volume (3.50% in this dataset), so a
naive accuracy-optimized model can score >96% accuracy while catching zero fraud. Beyond
detection, production fraud systems need to explain *why* a transaction was flagged and detect
when fraud patterns have shifted enough that the model needs retraining. This project addresses
all three: imbalance-aware modeling, per-prediction SHAP explanations, and drift monitoring with
a concrete retrain-or-not recommendation.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle competition) —
590,540 transactions, 434 raw features (many anonymized `V*`/`C*`/`D*` engineered columns, plus
card/device/identity metadata), 3.50% fraud rate.

## Project structure

```
data/
  raw/               # original Kaggle CSVs (gitignored — see Setup)
  interim/           # scratch space (gitignored)
  processed/         # example_transactions.json for the API/dashboard (tracked)
src/
  data/              # download + memory-efficient loading (dtype downcasting)
  models/            # time-based split, baseline LR, main LightGBM + imbalance comparison
  explainability/     # SHAP explainer (shared by the report generator and the API)
  drift/             # PSI/KS statistics, out-of-sample performance tracking
  api/               # FastAPI service
  dashboard/         # Streamlit app
tests/               # pytest — API (TestClient) and dashboard (Streamlit AppTest)
models/              # trained model artifacts (gitignored, regenerate via scripts below)
reports/
  eda/, models/, explainability/, drift/   # generated markdown reports
  figures/           # all generated plots
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Download the dataset

1. Create a Kaggle API token at https://www.kaggle.com/settings/api.
   Kaggle currently issues two credential formats — use whichever you're given:
   - **Legacy key** (`{"username": ..., "key": ...}`) → save as `C:\Users\<you>\.kaggle\kaggle.json`
   - **New token** (prefixed `KGAT_...`, under "API Tokens") → save the raw token string as
     a plain-text file at `C:\Users\<you>\.kaggle\access_token` (no JSON, no username) —
     this format needs `kaggle>=1.8.0` (already pinned in requirements.txt)
2. Accept the competition rules at https://www.kaggle.com/c/ieee-fraud-detection/rules
   (Kaggle rejects API downloads for competitions you haven't joined, even with valid credentials).
3. Run:
   ```powershell
   .venv\Scripts\python src\data\download_data.py
   ```

## Running the pipeline

Each stage writes its output (models, reports, figures) to disk, so later stages don't need
earlier ones to be re-run — but the first full pass needs to go in this order:

```powershell
.venv\Scripts\python src\eda.py                                # EDA report
.venv\Scripts\python src\models\train_baseline.py               # baseline Logistic Regression
.venv\Scripts\python src\models\train_main.py                   # main LightGBM + imbalance comparison
.venv\Scripts\python src\explainability\generate_shap_report.py # SHAP global + local explanations
.venv\Scripts\python src\drift\generate_drift_report.py         # drift monitoring report
.venv\Scripts\python src\api\prepare_examples.py                # example transactions for API/dashboard

.venv\Scripts\python -m uvicorn src.api.main:app --reload --port 8000   # API at localhost:8000/docs
.venv\Scripts\python -m streamlit run src\dashboard\app.py               # dashboard at localhost:8501
```

Run tests with `.venv\Scripts\python -m pytest tests/`.

## Methodology and key decisions

**EDA** ([report](reports/eda/eda_report.md)) — confirmed the 3.50% fraud rate, found 214
features >50% missing (mostly `id_*`/`D*`/`V*` blocks — treated as signal via LightGBM's native
NaN handling rather than imputed away), `ProductCD == "C"` at ~11.7% fraud rate vs 3.5% overall,
and fraud rate drifting over the observed window — which motivated a **time-based** train/val/test
split (70/15/15 by `TransactionDT`) instead of random, throughout the project.

**Baseline** ([report](reports/models/baseline_metrics.md)) — Logistic Regression with
`class_weight="balanced"`, one-hot/imputed/scaled features. Val PR-AUC 0.412, test PR-AUC 0.189 —
the val/test gap is an early signal of the same temporal drift the EDA found.

**Imbalance-handling comparison** ([report](reports/models/main_model_comparison.md)) — three
techniques on identical LightGBM hyperparameters, isolating the technique's effect:

| Technique | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| Class weighting (`scale_pos_weight`) | **0.517** | 0.893 |
| SMOTE (SMOTENC, 30% minority ratio) | 0.499 | 0.891 |
| Class weighting + tuned threshold | 0.517 | 0.893 |

Class weighting beat SMOTE on PR-AUC — SMOTENC's synthetic interpolation in ~430 mixed
numeric/categorical dimensions blurs the boundary gradient-boosted trees already handle well via
loss reweighting, and requires imputing away LightGBM's native missing-value handling to make
interpolation possible. The final model uses class weighting, with the decision threshold tuned
on validation to maximize F2 (recall weighted over precision, since missing fraud costs more than
a false alarm) — moving from precision 0.559/recall 0.458 at the default 0.5 cutoff to precision
0.372/recall 0.561 at the tuned threshold of 0.283.

**Explainability** ([report](reports/explainability/shap_report.md)) — SHAP `TreeExplainer` on
the LightGBM model. Global importance is dominated by the anonymized `V*`/`C*` engineered
features, `TransactionAmt`, and card identifiers. The report includes local waterfall
explanations for a confidently caught fraud, a missed fraud, and a false alarm — not just the
aggregate view — matching the "human-readable justification per flagged transaction" requirement.
The `FraudExplainer` class (`src/explainability/explainer.py`) is shared by the report generator,
the API, and the dashboard, so explanations are computed identically everywhere.

**Drift monitoring** ([report](reports/drift/drift_report.md)) — custom PSI/KS implementation
(`src/drift/psi_ks.py`) rather than the `evidently` library, for transparent, version-stable
computation. The most important finding: out-of-sample PR-AUC dropped **35.5%** across the
val+test period, while only 4.6% of raw features showed significant PSI drift and the
*predicted-probability* distribution barely moved (PSI 0.005). That gap matters — it means this
is **concept drift** (the relationship between features and the fraud label changed), not
covariate shift, and a monitoring setup that only watches prediction distributions (the
label-free signal, available immediately) would completely miss it. The retrain-trigger logic
(>10% of features significantly drifted, or >25% relative PR-AUC drop) flags **RETRAIN
RECOMMENDED** on this data.

**API** — FastAPI service (`src/api/main.py`) with `/health`, `/examples`, `/examples/{id}`, and
`POST /predict`, which accepts a flexible feature-name→value payload (missing features become
NaN, matching LightGBM's native handling) and returns fraud probability, decision, and the
top-5 SHAP-contributing features.

**Dashboard** (optional) — Streamlit app (`src/dashboard/app.py`) with a Transaction Explorer
(pick an example, see the prediction and SHAP breakdown) and a Drift Monitoring page (retrain
recommendation, feature drift table, performance-over-time and prediction-drift charts).

## Results summary

| Model | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| Baseline (Logistic Regression) | 0.189 | 0.830 |
| Main (LightGBM, class weighting + tuned threshold) | **0.517** | 0.893 |

## Limitations

- **Anonymized features limit regulatory explainability.** SHAP tells you `V257` was the top
  driver, but Kaggle doesn't disclose what `V257` means — a real production system needs
  interpretable feature names for compliance-facing explanations, not just statistical attribution.
- **Label-free drift monitoring isn't sufficient here.** As found in the drift report, prediction-
  distribution drift didn't flag the performance degradation this dataset exhibits — a production
  system needs a fast feedback loop on delayed ground truth (chargebacks), not just distribution
  watching.
- **No hyperparameter tuning.** LightGBM params were fixed at reasonable defaults to keep the
  imbalance-handling comparison controlled (isolating technique effect, not hyperparameters);
  a production model would benefit from proper HPO on top of the chosen technique.
- **SMOTE comparison required imputation**, discarding LightGBM's native missing-value handling
  for that one experiment — a limitation of the comparison methodology, not of SMOTE generally.
- **Single train/test drift comparison, not continuous monitoring.** The drift module answers
  "has drift happened between these two windows," which is what a scheduled batch job would run;
  it isn't a streaming/real-time drift detector.
- **Threshold tuned for F2, a generic recall-favoring metric** — a real deployment would tune the
  operating point against actual cost figures (chargeback cost vs. investigation cost per flagged
  transaction), which aren't in this dataset.
- **No containerization or CI/CD** — out of scope for this project but the natural next step for
  actual deployment.
