# Final GitHub Checklist

## 1. Core Files

- [ ] `README.md` reflects final threshold `0.560` and test recall `62.21%`
- [ ] `requirements.txt` uses the Python 3.10-compatible dependency set
- [ ] `pyproject.toml` has Python `>=3.10,<3.11`
- [ ] `.gitignore` excludes raw data, model binaries, row-level predictions, SHAP rows, and secrets
- [ ] `LICENSE` has the correct copyright owner
- [ ] `docs/` contains the final documentation set

---

## 2. Docs Folder

Recommended final docs:

- [ ] `docs/PROJECT_SUMMARY.md`
- [ ] `docs/PROJECT_CASE_STUDY.md`
- [ ] `docs/INTERVIEW_TALKING_POINTS.md`
- [ ] `docs/PORTFOLIO_SNIPPETS.md`
- [ ] `docs/GITHUB_PRESENTATION_GUIDE.md`
- [ ] `docs/REPRODUCIBILITY_RUNBOOK.md`
- [ ] `docs/FINAL_GITHUB_CHECKLIST.md`

---

## 3. Notebooks

- [ ] `notebooks/00_project_brief_and_business_context.ipynb`
- [ ] `notebooks/01_data_ingestion_and_schema_review.ipynb`
- [ ] `notebooks/02_data_quality_assessment.ipynb`
- [ ] `notebooks/03_data_cleaning_and_preprocessing.ipynb`
- [ ] `notebooks/04_credit_risk_eda_and_portfolio_monitoring.ipynb`
- [ ] `notebooks/05_feature_engineering_and_leakage_review.ipynb`
- [ ] `notebooks/06_model_training_and_evaluation.ipynb`
- [ ] `notebooks/07_threshold_selection_and_business_costing.ipynb`
- [ ] `notebooks/08_explainable_ai_shap_anchors_counterfactuals.ipynb`
- [ ] `notebooks/09_model_governance_and_monitoring.ipynb`

---

## 4. Final Metric Consistency

Confirm the same final numbers appear in README, docs, and Notebook 09:

| Item | Final value |
|---|---:|
| Champion operating model | `xgboost_weighted_baseline` |
| Operating threshold | 0.560 |
| Test ROC-AUC | 0.7478 |
| Test PR-AUC | 0.2147 |
| Test recall | 62.21% |
| Test precision | 19.09% |
| Test review rate | 29.46% |
| Test business cost | $5.85M |

---

## 5. Figures

Recommended README figures:

- [ ] `reports/figures/portfolio_target_distribution.png`
- [ ] `reports/figures/default_rate_by_loan_category.png`
- [ ] `reports/figures/08_xai_global_grouped_shap_top_features.png`

Optional supporting figures:

- [ ] `reports/figures/default_rate_by_interest_rate_quantile.png`
- [ ] `reports/figures/08_xai_shap_summary_beeswarm.png`
- [ ] `reports/figures/08_xai_shap_dependence_interest_rate.png`
- [ ] `reports/figures/08_xai_shap_dependence_amount.png`
- [ ] `reports/figures/08_xai_shap_dependence_loan_to_income_ratio.png`

---

## 6. Governance Outputs

Prefer final `09_` governance markdown files:

- [ ] `reports/governance/09_model_card.md`
- [ ] `reports/governance/09_model_validation_summary.md`
- [ ] `reports/governance/09_model_monitoring_plan.md`
- [ ] `reports/governance/09_stakeholder_brief.md`

If older duplicate governance files exist, remove or update them before publishing.

---

## 7. Do Not Commit

- [ ] `data/raw/Credit_Risk_Dataset.xlsx`
- [ ] `data/interim/*.csv`
- [ ] `data/processed/*.csv`
- [ ] `reports/model_artifacts/*.joblib`
- [ ] `.venv/`
- [ ] `.env`
- [ ] `__pycache__/`
- [ ] `.ipynb_checkpoints/`
- [ ] row-level prediction CSVs
- [ ] row-level SHAP CSVs
- [ ] borrower-level counterfactual CSVs

---

## 8. Safe Commit Command

Run:

```bash
git status
```

Then stage safe files:

```bash
git add README.md LICENSE .gitignore requirements.txt pyproject.toml config/ src/ scripts/ notebooks/ docs/ reports/figures/ reports/governance/
git status
git commit -m "Complete Canadian retail credit risk XAI portfolio project"
```

If unsafe files appear, unstage them:

```bash
git restore --staged data/raw data/interim data/processed reports/model_artifacts
git restore --staged reports/tables/*predictions*.csv reports/tables/*shap_values*.csv
```

---

## 9. GitHub Presentation

- [ ] Repository About description is filled out
- [ ] GitHub topics are added
- [ ] README images render
- [ ] Project is pinned on GitHub profile
- [ ] LinkedIn project description is ready
- [ ] Resume bullets are ready
