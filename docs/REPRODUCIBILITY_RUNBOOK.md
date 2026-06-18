# Reproducibility Runbook

## Purpose

This runbook explains how to reproduce the Canadian Retail Credit Risk XAI project locally from raw workbook to final governance outputs.

---

## Recommended Environment

- Python: 3.10.x
- OS tested by project owner: Windows / PowerShell
- Package management: `venv` + `pip`

The project pins Python dependencies in `requirements.txt`.

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd canadian-retail-credit-risk-xai
```

---

## 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Add Raw Data Locally

Place the raw workbook here:

```text
data/raw/Credit_Risk_Dataset.xlsx
```

Raw data is intentionally excluded from GitHub.

---

## 4. Run Pipelines in Order

```powershell
python scripts/run_data_pipeline.py
python scripts/run_cleaning_pipeline.py
python scripts/run_portfolio_monitoring_pipeline.py
python scripts/run_feature_engineering_pipeline.py
python scripts/run_model_training_pipeline.py
python scripts/run_threshold_selection_pipeline.py
python scripts/run_explainability_pipeline.py
python scripts/run_governance_pipeline.py
```

---

## 5. Expected Final Results

Notebook 07 / threshold pipeline should show:

```text
Recommended operational model: xgboost_weighted_baseline
Operating threshold: 0.560
Test recall: 62.21%
Test precision: 19.09%
Test review rate: 29.46%
```

Notebook 08 / explainability pipeline should show the same champion model and threshold.

---

## 6. Expected Output Folders

```text
reports/tables/
reports/figures/
reports/governance/
reports/html/
reports/model_artifacts/
```

Important: `reports/model_artifacts/` should stay local and should not be committed.

---

## 7. Final Governance Outputs

Expected final governance markdown files:

```text
reports/governance/09_model_card.md
reports/governance/09_model_validation_summary.md
reports/governance/09_model_monitoring_plan.md
reports/governance/09_stakeholder_brief.md
```

If older unprefixed duplicates exist, make sure they are either updated or excluded from the final GitHub presentation.

---

## 8. Troubleshooting

### Import errors from saved `.joblib` models

Make sure `src/credit_risk/models/thresholding.py` contains compatibility functions used by Notebook 06/07/08.

### Deepchecks or joblib warnings

Warnings related to `joblib_memmapping_folder` or `resource_tracker` are usually cleanup warnings after successful execution. They do not necessarily indicate pipeline failure.

### SHAP shows `feature_0`, `feature_1`, etc.

Use the feature-name-fixed version of `src/credit_risk/explainability/shap_analysis.py`. Final SHAP outputs should show business-readable names such as `interest_rate`, `amount_missing_flag`, and `total_income_pa`.

### Notebook 08 shows stale threshold

Copy or regenerate the final `07_operational_threshold_recommendation_validation.csv` and rerun Notebook 08.

---

## 9. GitHub Safety Check

Before committing:

```bash
git status
```

Do not commit:

```text
data/raw/
data/interim/
data/processed/
reports/model_artifacts/
*.joblib
*.pkl
.env
.venv/
```

Row-level prediction, SHAP, and counterfactual outputs should also be reviewed before committing.

---

## 10. Recommended Validation Command

```powershell
python -m py_compile scripts/run_threshold_selection_pipeline.py
python -m py_compile scripts/run_explainability_pipeline.py
python -m py_compile scripts/run_governance_pipeline.py
```
