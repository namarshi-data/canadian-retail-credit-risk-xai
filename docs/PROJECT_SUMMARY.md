# Project Summary

## Canadian Retail Credit Risk Analytics: Explainable Default Prediction, Portfolio Monitoring, Threshold Strategy, and Model Governance

This project is an end-to-end credit risk analytics and machine-learning workflow for a Canadian retail lending portfolio. It is designed as a finance-industry portfolio project that demonstrates credit-risk business understanding, Python analytics, machine learning, explainable AI, threshold strategy, and model governance.

The model is framed as an **early-warning default-risk ranking and manual-review prioritization tool**. It is **not** presented as an automated credit-decline, pricing, or adverse-action engine.

---

## Business Context

Retail lenders need reliable early-warning indicators of borrower default risk. A practical credit-risk solution should support more than model accuracy. It should help answer:

- Which borrower and loan segments show elevated observed default risk?
- Which data-quality issues affect portfolio monitoring and model reliability?
- Which features can be used safely without target leakage?
- Which model ranks borrowers effectively under class imbalance?
- What operating threshold balances default capture and manual-review capacity?
- Which features explain model behaviour globally and locally?
- What governance controls are needed before a model could be used responsibly?

---

## Dataset and Target

| Item | Value |
|---|---:|
| Final modelling records | 134,417 |
| Observed default rate | 9.04% |
| Target variable | `defaulter` |
| Modelling purpose | Default-risk ranking and review prioritization |
| Primary use restriction | Decision support only; not automated decline |

---

## Methodology

1. Multi-sheet Excel ingestion
2. Record-grain validation to prevent many-to-many merge inflation
3. Data-quality review and missingness profiling
4. Cleaning, standardization, and audit-table creation
5. Portfolio monitoring and segment-risk analysis
6. Leakage-reviewed feature engineering
7. Train/validation/test split and preprocessing design
8. Model training using Logistic Regression, Random Forest, and XGBoost
9. Validation-based threshold selection under review-cap and cost constraints
10. Test-set confirmation of the selected operating policy
11. SHAP explainability, local explanations, anchor-style rules, and counterfactual diagnostics
12. Model card, validation summary, risk controls, monitoring limits, and stakeholder brief

---

## Final Model and Threshold Results

| Metric | Validation | Test |
|---|---:|---:|
| Champion operating model | `xgboost_weighted_baseline` | `xgboost_weighted_baseline` |
| Operating threshold | 0.560 | 0.560 |
| ROC-AUC | 0.7512 | 0.7478 |
| PR-AUC | 0.2263 | 0.2147 |
| Recall at threshold | 62.59% | 62.21% |
| Precision at threshold | 19.05% | 19.09% |
| F1 at threshold | 0.2921 | 0.2921 |
| Review rate | 29.71% | 29.46% |
| Illustrative business cost | $5.83M | $5.85M |

The selected threshold captures approximately **62.21% of default cases** on the held-out test set while keeping the review population under the project’s **30% review-rate cap**.

---

## Key Explainability Drivers

Top grouped SHAP drivers include:

- `interest_rate`
- `amount_missing_flag`
- `amount_missing_raw_flag`
- `broad_data_quality_issue_count`
- `total_income_pa`
- `core_data_quality_issue_count`
- `amount`
- `income_to_loan_buffer`
- `high_interest_flag`
- `tenure_years`

These drivers are interpreted with governance caution. For example, data-quality and missingness features can be useful risk signals, but they should not be interpreted as borrower behaviour alone.

---

## Governance and Responsible-Use Position

The final governance layer documents:

- Intended use and out-of-scope use
- Model card
- Validation/test evidence
- Control register
- Risk-limit register
- Monitoring KPI snapshot
- Stakeholder brief
- Monitoring plan
- Leakage and sensitive/proxy-feature controls
- Explainability and counterfactual limitations

The recommended governance decision is to use the model for **portfolio analytics, risk ranking, and manual-review prioritization**, not standalone credit decisioning.

---

## Portfolio Value

This project demonstrates skills relevant to Canadian finance and banking roles:

- Credit risk analytics
- Portfolio monitoring
- Data-quality assessment
- Python modelling
- Imbalanced classification evaluation
- XGBoost and Random Forest development
- Threshold strategy and operational capacity analysis
- SHAP explainability
- Model validation thinking
- Model risk governance and monitoring design
