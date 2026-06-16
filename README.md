# Canadian Retail Credit Risk Analytics

**Explainable Default Prediction, Portfolio Monitoring & Model Governance**

This project simulates a Canadian retail lending risk analytics workflow: data ingestion, portfolio monitoring, default prediction, explainable machine learning, business threshold selection, and model governance documentation.

## Business problem

Retail lenders need to identify borrowers with elevated default risk while keeping credit decisions explainable, auditable, and useful for business stakeholders. This project frames the model as an **early-warning default risk model** for a retail credit portfolio rather than a purely academic classification exercise.

## What this project demonstrates

- Credit risk data quality review and schema validation
- Borrower and portfolio monitoring KPIs
- Feature engineering with leakage controls
- Baseline and challenger model development
- Random Forest and XGBoost classification
- Imbalanced classification evaluation using PR-AUC, ROC-AUC, recall, precision, F1, MCC, and business-cost metrics
- SHAP-based global and local explanations
- Anchors and counterfactual explanations for stakeholder communication
- Model card, validation summary, and governance-ready documentation

## Repository structure

```text
config/                 Project and model configuration
data/                   Local data folders; raw data is not committed
notebooks/              Business-facing analysis notebooks
src/credit_risk/        Reusable Python package
reports/                Figures, tables, model artifacts, governance outputs
dashboards/             Power BI/Tableau placeholders
scripts/                Pipeline entry points
tests/                  Unit tests for data and modelling logic
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name canadian-credit-risk-xai --display-name "Canadian Credit Risk XAI"
```

Place the source Excel workbook locally at:

```text
data/raw/Credit_Risk_Dataset.xlsx
```

The data file is intentionally excluded from GitHub.

## Current project phase

Phase 1 is focused on project setup, business framing, and schema review.

Next notebooks:

1. `00_project_brief_and_business_context.ipynb`
2. `01_data_ingestion_and_schema_review.ipynb`

## Hiring positioning

This project is designed for Canadian finance roles such as Credit Risk Analyst, Risk Analytics Analyst, Data Analyst - Credit Risk, Model Risk Analyst, and Portfolio Analytics Analyst.
